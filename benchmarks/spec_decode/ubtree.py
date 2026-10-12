# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay UBTree selector inputs and time the CPU reference implementation.

Run from the repository root with `python -m benchmarks.spec_decode.ubtree`.
Synthetic data is opt-in and cannot establish model acceptance or speedup.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from vllm.v1.spec_decode.ubtree import build_ubtree, score_ubtree_edges


def synthetic_inputs(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    depth, top_k, vocab, rank = 15, 64, 256, 32
    return {
        "candidate_ids": torch.stack(
            [torch.randperm(vocab, generator=generator)[:top_k] for _ in range(depth)]
        ),
        "unary_logits": torch.randn(depth, top_k, generator=generator),
        "projected_hidden": torch.randn(depth, rank, generator=generator) * 0.1,
        "predecessor_codebook": torch.randn(vocab, rank, generator=generator),
        "successor_codebook": torch.randn(vocab, rank, generator=generator),
        "anchor_token_id": 7,
        "log_unigram_scale": torch.zeros(depth),
        "log_bigram_scale": torch.zeros(depth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", type=Path, help="torch.save dictionary of selector inputs"
    )
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random tensors; no model benchmark",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--search-width", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--save-input", type=Path, help="Save tensors for deterministic replay"
    )
    args = parser.parse_args()
    if (
        min(args.budgets) < 0
        or min(args.search_width, args.repeats, args.threads) < 1
        or args.warmup < 0
    ):
        parser.error(
            "budgets/warmup must be nonnegative; width/repeats/threads positive"
        )
    torch.set_num_threads(args.threads)
    if args.synthetic:
        inputs = synthetic_inputs(args.seed)
        source_info = {"kind": "synthetic", "seed": args.seed}
    else:
        inputs = torch.load(args.input, map_location="cpu", weights_only=True)
        source_info = {
            "kind": "tensor_replay",
            "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        }
    if args.save_input:
        torch.save(inputs, args.save_input)

    results = []
    with torch.inference_mode():
        for budget in args.budgets:
            timings: dict[str, list[float]] = {
                key: [] for key in ("selector", "tree_build", "metadata", "total")
            }
            for repeat in range(args.warmup + args.repeats):
                start = time.perf_counter()
                root, edges = score_ubtree_edges(**inputs)
                scored = time.perf_counter()
                tree = build_ubtree(
                    inputs["candidate_ids"],
                    root,
                    edges,
                    anchor_token_id=inputs["anchor_token_id"],
                    node_budget=budget,
                    search_width=args.search_width,
                )
                built = time.perf_counter()
                tree.attention_mask()
                tree.position_ids(0)
                finished = time.perf_counter()
                if repeat >= args.warmup:
                    for key, elapsed in zip(
                        timings,
                        (
                            scored - start,
                            built - scored,
                            finished - built,
                            finished - start,
                        ),
                    ):
                        timings[key].append(elapsed * 1000)
            results.append(
                {
                    "node_budget": budget,
                    "search_width": args.search_width,
                    "materialized_nonroot_nodes": tree.token_ids.numel() - 1,
                    "max_depth": int(tree.depths.max()),
                    "median_cpu_reference_ms": {
                        key: statistics.median(values)
                        for key, values in timings.items()
                    },
                    "cpu_reference_ms_samples": timings,
                    "tree": {
                        "token_ids": tree.token_ids.tolist(),
                        "parent_indices": tree.parent_indices.tolist(),
                        "depths": tree.depths.tolist(),
                        "log_scores": tree.log_scores.tolist(),
                    },
                }
            )
    report = {
        "schema_version": 1,
        "benchmark": "ubtree_cpu_reference",
        "source": source_info,
        "includes_target_forward": False,
        "includes_kv_commit": False,
        "includes_proposer_forward": False,
        "includes_codebook_precompute": False,
        "model_speedup_measured": False,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "threads": args.threads,
        },
        "repeats": args.repeats,
        "warmup": args.warmup,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote CPU reference results to {args.output}; no model speedup measured.")


if __name__ == "__main__":
    main()
