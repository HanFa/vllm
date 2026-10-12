# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness oracles for the experimental UBTree inference primitives.

The public contract is scores -> prefix-closed tree -> target-sampled commit
plan. Exhaustive tiny trees and independent per-prefix attention catch search,
masking, and KV-selection errors without a model checkpoint or GPU engine.
"""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.spec_decode.ubtree import (
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
