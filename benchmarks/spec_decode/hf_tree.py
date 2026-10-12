# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request HF reference for DFlash unigram trees, not a vLLM backend.

The cache adapter targets Transformers 4.57.3 Qwen3 full attention. Python tree
search, path sampling, and KV gathers are intentionally visible in total timing.
"""

import time
from contextlib import contextmanager

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

from benchmarks.spec_decode.ubtree_reference import (
    UBTree,
    UBTreeAcceptance,
    build_ubtree,
)


def unigram_tree(logits, anchor, budget, width, top_k):
    """Use top-K-renormalized unigram scores with no trained bigram selector."""
    scores, ids = logits.float().topk(top_k, dim=-1)
    depth, k = ids.shape
    return build_ubtree(
        ids,
        scores[0],
        scores[1:, None, :].expand(depth - 1, k, k),
        anchor_token_id=anchor,
        node_budget=budget,
        search_width=width,
    )


def chain_tree(logits, anchor):
    """DFlash's greedy single-path proposal; the target may still sample."""
    ids = [anchor, *logits.argmax(-1).cpu().tolist()]
    count = len(ids)
    return UBTree(
        torch.tensor(ids),
        torch.arange(-1, count - 1),
        torch.arange(count),
        torch.zeros(count, dtype=torch.float64),
    )


def verify_tree(target, cache, tree, *, causal_chain=False):
    """Append flattened nodes to scratch KV with prefix-correct attention/RoPE."""
    prefix = cache.get_seq_length()
    device, dtype = target.device, target.dtype
    count = tree.token_ids.numel()
    mask = None
    if not causal_chain and count > 1:
        allowed = torch.cat(
            [torch.ones(count, prefix, dtype=torch.bool), tree.attention_mask()], 1
        ).to(device)
        mask = torch.zeros(count, prefix + count, dtype=dtype, device=device)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        mask = mask[None, None]
    return target(
        tree.token_ids.to(device)[None],
        attention_mask=mask,
        position_ids=tree.position_ids(prefix).to(device)[None],
        # Physical cache rows differ from RoPE positions for branched trees.
        cache_position=torch.arange(prefix, prefix + count, device=device),
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
    )


def commit_cache(cache, prefix, rows):
    """Compact accepted scratch rows in place; preserve all verified prefix KV."""
    if rows.ndim != 1 or not rows.numel() or int(rows[0]) != 0:
        raise ValueError("accepted rows must start with anchor row zero")
    if prefix < 0 or (rows[1:] <= rows[:-1]).any():
        raise ValueError("require nonnegative prefix and increasing accepted rows")
    if not cache.layers or any(
        type(layer) is not DynamicLayer for layer in cache.layers
    ):
        raise ValueError("only full-attention DynamicLayer caches are supported")
    if any(prefix + int(rows[-1]) >= layer.get_seq_length() for layer in cache.layers):
        raise ValueError("accepted row is outside scratch KV")
    end = prefix + rows.numel()
    for layer in cache.layers:
        indices = rows.to(layer.keys.device) + prefix
        # index_select owns its data: destination may overlap source rows.
        layer.keys[..., prefix:end, :].copy_(layer.keys.index_select(-2, indices))
        layer.values[..., prefix:end, :].copy_(layer.values.index_select(-2, indices))
        layer.crop(end)


def sample_token(logits, temperature, generator):
    if temperature == 0:
        return int(logits.argmax(-1))
    probabilities = (logits.float() / temperature).softmax(-1)
    return int(torch.multinomial(probabilities, 1, generator=generator))


def sample_path(tree, logits, *, temperature, generator, limit, eos):
    """Draw only on the reached path, consuming one RNG draw per output token."""
    children = {
        (int(tree.parent_indices[i]), int(tree.token_ids[i])): i
        for i in range(1, tree.token_ids.numel())
    }
    node, bonus = 0, False
    output, rows = [], [0]
    for _ in range(limit):
        token = sample_token(logits[node], temperature, generator)
        output.append(token)
        child = children.get((node, token))
        if child is None:
            bonus = True
        else:
            rows.append(child)
            node = child
        if bonus or token in eos:
            break
    return UBTreeAcceptance(
        torch.tensor(output, dtype=torch.long), torch.tensor(rows), bonus
    )


def context_features(output, layer_ids, rows=None):
    states = [output.hidden_states[i + 1] for i in layer_ids]
    if rows is not None:
        states = [s.index_select(1, rows.to(s.device)) for s in states]
    return torch.cat(states, dim=-1)


def check_tree_logits(target, prefix_ids, tree, logits, *, atol, rtol, max_tv=0.05):
    """Compare every tree node with an independently recomputed causal prefix."""
    paths = []
    max_error = 0.0
    max_probability_tv, argmax_mismatches = 0.0, 0
    for i, token in enumerate(tree.token_ids.tolist()):
        path = ([] if i == 0 else paths[int(tree.parent_indices[i])]) + [token]
        paths.append(path)
        ids = torch.cat([prefix_ids, prefix_ids.new_tensor([path])], dim=1)
        expected = target(ids, use_cache=False, logits_to_keep=1).logits[0, -1]
        max_error = max(
            max_error, float((expected.float() - logits[i].float()).abs().max())
        )
        tv = float(
            (expected.float().softmax(-1) - logits[i].float().softmax(-1)).abs().sum()
            / 2
        )
        max_probability_tv = max(max_probability_tv, tv)
        argmax_mismatches += int(int(expected.argmax()) != int(logits[i].argmax()))
        torch.testing.assert_close(logits[i], expected, atol=atol, rtol=rtol)
        if tv > max_tv:
            raise AssertionError(
                f"node {i}: probability TV at T=1 is {tv}, limit {max_tv}"
            )
    return {
        "nodes": len(paths),
        "max_abs_logit_error": max_error,
        "max_probability_tv_at_temperature_1": max_probability_tv,
        "argmax_mismatches": argmax_mismatches,
    }


def synchronize(device):
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


class StageTimer:
    def __init__(self, device, enabled):
        self.device = device
        self.enabled = enabled
        self.totals = {}

    @contextmanager
    def measure(self, name):
        if self.enabled:
            synchronize(self.device)
            start = time.perf_counter()
        yield
        if self.enabled:
            synchronize(self.device)
            self.totals[name] = (
                self.totals.get(name, 0.0) + (time.perf_counter() - start) * 1000
            )


@torch.inference_mode()
def generate(
    target,
    draft,
    input_ids,
    *,
    arm,
    max_new_tokens,
    temperature=0.0,
    seed=0,
    eos=(),
    budget=32,
    width=12,
    top_k=64,
    depth=15,
    profile=False,
    check_rounds=0,
    atol=0.05,
    rtol=0.01,
    max_tv=0.05,
):
    """Generate with target-only, DFlash chain, or unigram-tree verification.

    Each run starts with fresh target/draft caches and RNG. The already sampled
    anchor is emitted once; subsequent rounds emit matching children and one
    unmatched bonus. TTFT includes prefill/first sampling; decode TPOT divides
    time after that point by N-1, never by N. No top-p/penalty processors are used.
    """
    if arm not in ("target", "chain", "tree"):
        raise ValueError("unknown arm")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("provide one nonempty, unpadded prompt")
    if (
        max_new_tokens < 1
        or temperature < 0
        or not torch.isfinite(torch.tensor(temperature))
    ):
        raise ValueError("require positive output limit and finite temperature >= 0")
    if arm != "target" and (draft is None or not 1 <= depth < draft.block_size):
        raise ValueError("draft depth must fit the trained DFlash block")
    if target.config.model_type != "qwen3" or any(
        kind != "full_attention" for kind in target.config.layer_types
    ):
        raise ValueError("this reference supports dense, full-attention Qwen3 only")
    if target.config._attn_implementation not in ("sdpa", "eager"):
        raise ValueError("tree masks require SDPA or eager attention")
    if (
        input_ids.shape[1] + max_new_tokens + depth
        >= target.config.max_position_embeddings
    ):
        raise ValueError("prompt plus output/draft exceeds target context limit")
    generator = torch.Generator(device=target.device).manual_seed(seed)
    cache, draft_cache = DynamicCache(), DynamicCache()
    timer = StageTimer(target.device, profile)
    if target.device.type == "cuda":
        torch.accelerator.memory.reset_peak_memory_stats(target.device)
    synchronize(target.device)
    start = time.perf_counter()
    output = target(
        input_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=arm != "target",
    )
    tokens = [sample_token(output.logits[0, -1], temperature, generator)]
    hidden = (
        context_features(output, draft.target_layer_ids) if arm != "target" else None
    )
    del output
    synchronize(target.device)
    first = time.perf_counter()
    rounds, verified, matched = 0, 0, 0
    checks = []
    while len(tokens) < max_new_tokens and tokens[-1] not in eos:
        prefix = cache.get_seq_length()
        if arm == "target":
            output = target(
                input_ids.new_tensor([[tokens[-1]]]),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            tokens.append(sample_token(output.logits[0, -1], temperature, generator))
            del output
            rounds += 1
            continue
        with timer.measure("proposer_ms"):
            # Always run the trained block shape, then truncate candidate depth.
            block = input_ids.new_full((1, draft.block_size), draft.mask_token_id)
            block[0, 0] = tokens[-1]
            draft_start = draft_cache.get_seq_length()
            if draft_start + hidden.shape[1] != prefix:
                raise AssertionError("draft context is not the committed target prefix")
            draft_hidden = draft(
                target_hidden=hidden,
                noise_embedding=target.get_input_embeddings()(block),
                position_ids=torch.arange(
                    draft_start, prefix + draft.block_size, device=target.device
                )[None],
                past_key_values=draft_cache,
                use_cache=True,
                is_causal=False,
            )
            logits = target.get_output_embeddings()(draft_hidden[:, 1 : depth + 1])[0]
            draft_cache.crop(prefix)  # Discard all noise/anchor-derived draft KV.
            del draft_hidden
        with timer.measure("tree_build_ms"):
            tree = (
                chain_tree(logits, tokens[-1])
                if arm == "chain"
                else unigram_tree(logits, tokens[-1], budget, width, top_k)
            )
            del logits
        with timer.measure("verify_ms"):
            output = verify_tree(target, cache, tree, causal_chain=arm == "chain")
        if rounds < check_rounds:
            prefix_ids = torch.cat([input_ids, input_ids.new_tensor([tokens[:-1]])], 1)
            checks.append(
                check_tree_logits(
                    target,
                    prefix_ids,
                    tree,
                    output.logits[0],
                    atol=atol,
                    rtol=rtol,
                    max_tv=max_tv,
                )
            )
        with timer.measure("sample_commit_ms"):
            accepted = sample_path(
                tree,
                output.logits[0],
                temperature=temperature,
                generator=generator,
                limit=max_new_tokens - len(tokens),
                eos=eos,
            )
            rows = accepted.kv_row_indices
            commit_cache(cache, prefix, rows)
            hidden = context_features(output, draft.target_layer_ids, rows)
            new_tokens = accepted.output_token_ids.tolist()
            if (
                len(tokens) + len(new_tokens) < max_new_tokens
                and new_tokens[-1] not in eos
            ):
                assert accepted.has_bonus_token
            tokens.extend(new_tokens)
            del output
        rounds += 1
        verified += tree.token_ids.numel() - 1
        matched += rows.numel() - 1
    synchronize(target.device)
    end = time.perf_counter()
    decode_tokens = len(tokens) - 1
    decode_ms = (end - first) * 1000
    return {
        "arm": arm,
        "output_token_ids": tokens,
        "output_tokens": len(tokens),
        "decode_tokens": decode_tokens,
        "ttft_ms": (first - start) * 1000,
        "decode_ms": decode_ms,
        "e2e_ms": (end - start) * 1000,
        "decode_tpot_ms": decode_ms / decode_tokens if decode_tokens else None,
        "rounds": rounds,
        "committed_tokens_per_round": decode_tokens / rounds if rounds else None,
        "matched_draft_tokens": matched,
        "verified_nonroot_nodes": verified,
        "verified_nodes_per_decode_token": verified / decode_tokens
        if decode_tokens
        else None,
        "stage_totals_ms": timer.totals,
        "checked_round_logit_diagnostics": checks,
        "timing_valid_for_comparison": not profile and not check_rounds,
        "peak_allocated_bytes": (
            torch.accelerator.memory.max_memory_allocated(target.device)
            if target.device.type == "cuda"
            else None
        ),
    }
