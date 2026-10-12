# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness oracles for the experimental UBTree inference primitives.

The public contract is scores -> prefix-closed tree -> target-sampled commit
plan. Exhaustive tiny trees and independent per-prefix attention catch search,
masking, and KV-selection errors without a model checkpoint or GPU engine.
"""

import itertools
import math
import os

import pytest
import torch
import torch.nn.functional as F

from benchmarks.spec_decode.ubtree_reference import (
    UBTree,
    build_ubtree,
    follow_target_samples,
    precompute_codebook,
    score_ubtree_edges,
)


def _paths(tree: UBTree) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = [()]
    for node in range(1, tree.token_ids.numel()):
        paths.append(
            paths[int(tree.parent_indices[node])] + (int(tree.token_ids[node]),)
        )
    return paths


@pytest.mark.parametrize("depth,top_k", [(1, 1), (1, 3), (4, 3)])
def test_selector_matches_scalar_definition_with_depth_calibration(depth, top_k):
    """Catch predecessor/successor, anchor, and calibration-index swaps."""
    generator = torch.Generator().manual_seed(7)
    vocab, embed, middle, rank = 19, 5, 7, 4
    embeddings = torch.randn(vocab, embed, generator=generator)
    w1 = torch.randn(middle, embed, generator=generator)
    w2 = torch.randn(rank, middle, generator=generator)
    pred = precompute_codebook(embeddings, w1, w2)
    succ = precompute_codebook(embeddings, -w1, w2)
    ids = torch.stack(
        [torch.randperm(vocab, generator=generator)[:top_k] for _ in range(depth)]
    )
    unary = torch.randn(depth, top_k, generator=generator)
    hidden = torch.randn(depth, rank, generator=generator)
    rho = torch.linspace(-0.4, 0.7, depth)
    kappa = torch.linspace(0.2, -0.6, depth)
    root, edges = score_ubtree_edges(ids, unary, hidden, pred, succ, 17, rho, kappa)

    def codebook(token, sign):
        projected = [
            sum(float(embeddings[token, j] * w1[i, j]) * sign for j in range(embed))
            for i in range(middle)
        ]
        activated = [x / (1 + math.exp(-x)) for x in projected]
        return [
            sum(float(w2[r, i]) * activated[i] for i in range(middle))
            for r in range(rank)
        ]

    for step in range(depth):
        for previous in range(1 if step == 0 else top_k):
            a = 17 if step == 0 else int(ids[step - 1, previous])
            for current in range(top_k):
                b = int(ids[step, current])
                delta = sum(
                    x * y * float(h)
                    for x, y, h in zip(codebook(a, 1), codebook(b, -1), hidden[step])
                )
                expected = (
                    math.exp(float(rho[step])) * float(unary[step, current])
                    + math.exp(float(kappa[step])) * delta
                )
                actual = (
                    root[current] if step == 0 else edges[step - 1, previous, current]
                )
                assert float(actual) == pytest.approx(expected, abs=2e-5, rel=1e-5)


@pytest.mark.parametrize("budget", [0, 1, 4, 13, 50])
def test_wide_search_matches_exhaustive_prefix_ranking(budget):
    """With no frontier pruning, global selection must match enumeration."""
    ids = torch.tensor([[2, 3], [4, 5], [6, 7]])
    root = torch.tensor([0.3, -0.7], dtype=torch.float64)
    edges = torch.tensor(
        [[[0.6, -0.2], [-1.1, 0.2]], [[-0.9, 1.3], [0.1, 0.7]]], dtype=torch.float64
    )
    tree = build_ubtree(
        ids, root, edges, anchor_token_id=1, node_budget=budget, search_width=8
    )

    def log_prob(row, column):
        # Independent scalar normalization; enumerate complete candidate prefixes.
        weights = [math.exp(float(x) - max(row)) for x in row]
        return math.log(weights[column] / sum(weights))

    scores = {}
    for depth in range(1, 4):
        for path in itertools.product(range(2), repeat=depth):
            score = log_prob(root.tolist(), path[0])
            for step in range(1, depth):
                score += log_prob(edges[step - 1, path[step - 1]].tolist(), path[step])
            scores[tuple(int(ids[i, p]) for i, p in enumerate(path))] = score
    expected = sorted(scores, key=lambda path: (-scores[path], len(path)))[:budget]
    assert set(_paths(tree)[1:]) == set(expected)
    for path, score in zip(_paths(tree)[1:], tree.log_scores[1:]):
        assert float(score) == pytest.approx(scores[path])


def test_width_limits_entire_depth_and_global_budget_excludes_anchor():
    ids = torch.tensor([[2, 3], [4, 5], [6, 7]])
    tree = build_ubtree(
        ids,
        torch.zeros(2),
        torch.zeros(2, 2, 2),
        anchor_token_id=1,
        node_budget=20,
        search_width=2,
    )
    assert tree.token_ids.numel() == 7
    assert tree.depths.bincount().tolist() == [1, 2, 2, 2]
    for node, parent in enumerate(tree.parent_indices.tolist()[1:], start=1):
        assert 0 <= parent < node


def test_zero_log_probability_ties_retain_ancestors_before_descendants():
    tree = build_ubtree(
        torch.tensor([[2], [3], [4], [5]]),
        torch.zeros(1),
        torch.zeros(3, 1, 1),
        anchor_token_id=1,
        node_budget=2,
        search_width=1,
    )
    assert tree.token_ids.tolist() == [1, 2, 3]
    assert tree.parent_indices.tolist() == [-1, 0, 1]


def _branched_tree():
    return build_ubtree(
        torch.tensor([[2, 3], [4, 5]]),
        torch.tensor([0.6, 0.4]).log(),
        torch.tensor([[[0.1, 0.9], [0.9, 0.1]]]).log(),
        anchor_token_id=1,
        node_budget=6,
        search_width=4,
    )


def test_tree_attention_matches_independent_autoregressive_prefixes():
    """Each repeated token has its own ancestry; siblings cannot leak keys."""
    tree = _branched_tree()
    paths = _paths(tree)
    assert (2, 4) in paths and (3, 4) in paths
    generator = torch.Generator().manual_seed(9)
    count, prefix, hidden = tree.token_ids.numel(), 3, 8
    q = torch.randn(count, hidden, generator=generator)
    k = torch.randn(prefix + count, hidden, generator=generator)
    v = torch.randn(prefix + count, hidden, generator=generator)
    mask = torch.cat(
        [torch.ones(count, prefix, dtype=torch.bool), tree.attention_mask()], dim=1
    )
    actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    for node, path in enumerate(paths):
        ancestors = [
            i
            for i, other in enumerate(paths)
            if other == path[: len(other)] and len(other) <= len(path)
        ]
        keys = [*range(prefix), *[prefix + i for i in ancestors]]
        expected = (q[node] @ k[keys].T / math.sqrt(hidden)).softmax(-1) @ v[keys]
        torch.testing.assert_close(actual[node], expected)
        assert tree.position_ids(prefix)[node] == prefix + len(path)


@pytest.mark.parametrize(
    "limit,eos,expected,bonus",
    [
        (8, (), [3, 4, 9], True),
        (1, (), [3], False),
        (8, (4,), [3, 4], False),
        (8, (9,), [3, 4, 9], True),
        (0, (), [], False),
    ],
)
def test_target_path_and_kv_plan_stop_at_bonus_eos_or_limit(
    limit, eos, expected, bonus
):
    tree = _branched_tree()
    paths = _paths(tree)
    samples = torch.full_like(tree.token_ids, 18)
    samples[0] = 3
    samples[paths.index((3,))] = 4
    samples[paths.index((3, 4))] = 9
    result = follow_target_samples(
        tree, samples, max_new_tokens=limit, eos_token_ids=eos
    )
    assert result.output_token_ids.tolist() == expected
    assert result.has_bonus_token is bonus
    accepted = [(3,), (3, 4)][: min(len(expected), 2)]
    assert result.kv_row_indices.tolist() == [
        0,
        *[paths.index(path) for path in accepted],
    ]
    assert all(paths[i] in [(), *accepted] for i in result.kv_row_indices.tolist())


def test_root_mismatch_does_not_accept_an_unreachable_branch():
    tree = _branched_tree()
    result = follow_target_samples(
        tree, torch.tensor([8, 4, 5, 4, 5, 4, 5]), max_new_tokens=8
    )
    assert result.output_token_ids.tolist() == [8]
    assert result.kv_row_indices.tolist() == [0]
    assert result.has_bonus_token


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_logits_are_rejected_before_pruning(bad):
    with pytest.raises(ValueError, match="finite"):
        build_ubtree(
            torch.tensor([[1, 2]]),
            torch.tensor([0.0, bad]),
            torch.empty(0, 2, 2),
            anchor_token_id=0,
            node_budget=2,
            search_width=2,
        )


def test_duplicate_candidates_cannot_create_ambiguous_children():
    with pytest.raises(ValueError, match="unique"):
        build_ubtree(
            torch.tensor([[2, 2]]),
            torch.zeros(2),
            torch.empty(0, 2, 2),
            anchor_token_id=0,
            node_budget=2,
            search_width=2,
        )


@pytest.fixture(params=["eager", "sdpa"])
def tiny_qwen_target(request):
    """Actual dense decoder/RoPE/cache; no checkpoint download is needed."""
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.3":
        pytest.skip("the standalone DFlash reference pins Transformers 4.57.3")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(13)
        config = transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
        )
        config._attn_implementation = request.param
        model = transformers.Qwen3ForCausalLM(config).eval()
    yield model
    torch.set_num_threads(previous_threads)


@torch.inference_mode()
@pytest.mark.parametrize("path", [(), (2, 5), (3, 4)])
def test_qwen_tree_logits_and_compacted_kv_match_fresh_causal_decode(
    tiny_qwen_target, path
):
    """Real decoder catches sibling leakage, flattened RoPE, and stale branch KV."""
    from transformers import DynamicCache

    from benchmarks.spec_decode.hf_tree import (
        check_tree_logits,
        commit_cache,
        verify_tree,
    )

    model = tiny_qwen_target
    prefix = torch.tensor([[11, 12, 13]])
    cache = DynamicCache()
    model(prefix, past_key_values=cache, use_cache=True)
    tree = _branched_tree()
    output = verify_tree(model, cache, tree)
    check_tree_logits(model, prefix, tree, output.logits[0], atol=1e-6, rtol=1e-5)
    paths = _paths(tree)
    rows = torch.tensor([0, *[paths.index(path[:i]) for i in range(1, len(path) + 1)]])
    commit_cache(cache, prefix.shape[1], rows)
    committed = torch.cat([prefix, tree.token_ids[rows][None]], dim=1)
    expected_cache = DynamicCache()
    model(committed, past_key_values=expected_cache, use_cache=True)
    assert cache.get_seq_length() == committed.shape[1]
    for actual, expected in zip(cache.layers, expected_cache.layers):
        torch.testing.assert_close(actual.keys, expected.keys, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.values, expected.values, atol=1e-6, rtol=1e-5)

    # Decode a second branched round after compaction, then a fresh bonus token.
    second = verify_tree(model, cache, tree)
    check_tree_logits(model, committed, tree, second.logits[0], atol=1e-6, rtol=1e-5)
    commit_cache(cache, committed.shape[1], rows)
    bonus = torch.tensor([[19]])
    actual = model(bonus, past_key_values=cache, use_cache=True).logits[0, -1]
    full = torch.cat([committed, tree.token_ids[rows][None], bonus], 1)
    expected = model(full, use_cache=False).logits[0, -1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_sampled_tree_path_consumes_only_causal_path_randomness(tiny_qwen_target):
    """Extra branches must not change RNG draws or the target sampling law."""
    from benchmarks.spec_decode.hf_tree import sample_path

    tree = _branched_tree()
    probabilities = torch.zeros(3, 32)
    probabilities[0, [2, 3]] = torch.tensor([0.3, 0.7])
    probabilities[1, [4, 5]] = torch.tensor([0.6, 0.4])
    probabilities[2, [8, 9]] = torch.tensor([0.2, 0.8])
    for seed in range(16):
        generator = torch.Generator().manual_seed(seed)
        reference_generator = torch.Generator().manual_seed(seed)
        result = sample_path(
            tree,
            probabilities[tree.depths].log(),
            temperature=1.0,
            generator=generator,
            limit=8,
            eos=(),
        )
        expected = [
            int(torch.multinomial(p, 1, generator=reference_generator))
            for p in probabilities
        ]
        assert result.output_token_ids.tolist() == expected
        assert result.has_bonus_token
        assert tree.token_ids[result.kv_row_indices].tolist() == [1, *expected[:-1]]
        assert torch.equal(generator.get_state(), reference_generator.get_state())


@torch.inference_mode()
def test_probability_guard_catches_errors_hidden_by_wide_logit_tolerance(
    tiny_qwen_target,
):
    """A BF16 logit allowance must not silently allow a large sampling change."""
    from transformers import DynamicCache

    from benchmarks.spec_decode.hf_tree import check_tree_logits, verify_tree

    model = tiny_qwen_target
    prefix = torch.tensor([[11, 12, 13]])
    cache = DynamicCache()
    model(prefix, past_key_values=cache, use_cache=True)
    tree = _branched_tree()
    logits = verify_tree(model, cache, tree).logits[0].clone()
    logits[:, :16] += 0.7
    with pytest.raises(AssertionError, match="probability TV"):
        check_tree_logits(model, prefix, tree, logits, atol=1.0, rtol=0.0, max_tv=0.05)


@torch.inference_mode()
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_reference_target_decode_matches_independent_causal_sampling(
    tiny_qwen_target, temperature
):
    """Check first-token/output-limit accounting and cached target sampling."""
    from benchmarks.spec_decode.hf_tree import generate, sample_token

    model = tiny_qwen_target
    prompt = torch.tensor([[11, 12, 13]])
    result = generate(
        model,
        None,
        prompt,
        arm="target",
        max_new_tokens=8,
        temperature=temperature,
        seed=71,
    )
    full = prompt.clone()
    generator = torch.Generator().manual_seed(71)
    for _ in range(8):
        logits = model(full, use_cache=False).logits[0, -1]
        token = sample_token(logits, temperature, generator)
        full = torch.cat([full, torch.tensor([[token]])], 1)
    assert result["output_token_ids"] == full[0, prompt.shape[1] :].tolist()
    assert result["decode_tokens"] == result["rounds"] == 7
    assert result["decode_tpot_ms"] == result["decode_ms"] / 7


@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_DFLASH_REFERENCE") != "1",
    reason="opt-in test downloads pinned DFlash Python code, never model weights",
)
@torch.inference_mode()
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_public_dflash_code_cache_and_generation_parity(tiny_qwen_target, temperature):
    """Exercise the real checkpoint interface with tiny random weights on CPU."""
    from transformers import DynamicCache, Qwen3Config
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from benchmarks.spec_decode.dflash_tree import DRAFT, DRAFT_REVISION
    from benchmarks.spec_decode.hf_tree import generate

    draft_class = get_class_from_dynamic_module(
        "dflash.DFlashDraftModel", DRAFT, revision=DRAFT_REVISION
    )

    # Hugging Face resolves the pinned checkpoint class at runtime.
    class CheckedDraft(draft_class):  # type: ignore[valid-type, misc]
        def forward(self, **kwargs):
            if kwargs["past_key_values"].get_seq_length() == 0:
                self.history = kwargs["target_hidden"]
            else:
                self.history = torch.cat([self.history, kwargs["target_hidden"]], 1)
            actual = super().forward(**kwargs)
            full = dict(kwargs)
            full["past_key_values"] = DynamicCache()
            full["target_hidden"] = self.history
            full["position_ids"] = torch.arange(
                self.history.shape[1] + self.block_size
            )[None]
            expected = super().forward(**full)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            return actual

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_target_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        block_size=4,
        dflash_config={"mask_token_id": 31, "target_layer_ids": [0]},
    )
    config._attn_implementation = tiny_qwen_target.config._attn_implementation
    with torch.random.fork_rng():
        torch.manual_seed(51)
        draft = CheckedDraft(config).eval()
    for limit in (1, 2, 12):
        baseline = None
        for arm, budget in [("target", 0), ("chain", 0), ("tree", 0), ("tree", 8)]:
            result = generate(
                tiny_qwen_target,
                draft,
                torch.tensor([[11, 12, 13]]),
                arm=arm,
                max_new_tokens=limit,
                temperature=temperature,
                seed=7,
                budget=budget,
                width=8,
                top_k=16,
                depth=3,
                check_rounds=1,
                atol=1e-6,
                rtol=1e-5,
            )
            if baseline is None:
                baseline = result["output_token_ids"]
            assert result["output_token_ids"] == baseline
    assert baseline is not None
    for arm in ("target", "chain", "tree"):
        result = generate(
            tiny_qwen_target,
            draft,
            torch.tensor([[11, 12, 13]]),
            arm=arm,
            max_new_tokens=12,
            temperature=temperature,
            seed=7,
            depth=3,
            top_k=16,
            eos=(baseline[0],),
        )
        assert result["output_token_ids"] == baseline[:1]
        assert result["rounds"] == 0
