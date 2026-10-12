# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-backed DFlash tree ablation. See README_ubtree.md for GPU commands."""

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from benchmarks.spec_decode.hf_tree import generate

TARGET = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
DRAFT_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "bench", "profile"], required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--target-revision", default=TARGET_REVISION)
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--check-rounds", type=int, default=1)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--max-tv", type=float, default=0.05)
    args = parser.parse_args()
    if transformers.__version__ != "4.57.3":
        parser.error("use Transformers 4.57.3 in a dedicated reference environment")
    if (
        min(args.budgets) < 0
        or min(args.width, args.top_k, args.repeats, args.threads) < 1
    ):
        parser.error("require budgets >= 0, width/top-k/repeats/threads >= 1")
    if args.warmup < 0 or args.check_rounds < 1:
        parser.error("require warmup >= 0 and check-rounds >= 1")
    if (
        args.max_new_tokens < 1
        or args.depth < 1
        or not math.isfinite(args.temperature)
        or args.temperature < 0
        or args.atol < 0
        or args.rtol < 0
        or not 0 <= args.max_tv <= 1
    ):
        parser.error("invalid output limit, depth, temperature, or logit tolerance")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no weights have been downloaded")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    raw = args.prompts.read_bytes()
    prompts = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not prompts or any(not isinstance(p.get("prompt"), str) for p in prompts):
        parser.error('prompts must be nonempty JSONL with a string "prompt" per row')
    tokenizer = AutoTokenizer.from_pretrained(
        args.target, revision=args.target_revision
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        revision=args.target_revision,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attention,
        device_map=args.device,
    ).eval()
    # This pinned public checkpoint defines the DFlash forward interface.
    draft = AutoModel.from_pretrained(
        args.draft,
        revision=args.draft_revision,
        code_revision=args.draft_revision,
        trust_remote_code=True,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attention,
        device_map=args.device,
    ).eval()
    if (
        target.config.hidden_size != draft.config.hidden_size
        or target.config.num_hidden_layers != draft.config.num_target_layers
        or target.config.vocab_size != draft.config.vocab_size
    ):
        parser.error("target and drafter dimensions do not match")
    if not 1 <= args.depth < draft.block_size or args.top_k > target.config.vocab_size:
        parser.error("draft depth or top-k exceeds checkpoint dimensions")
    eos = target.generation_config.eos_token_id
    eos = tuple(eos if isinstance(eos, list) else [eos]) if eos is not None else ()
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            enable_thinking=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(target.device)
        for p in prompts
    ]
    arms = [("target", 0), ("chain", 0), *[("tree", b) for b in args.budgets]]
    settings = dict(
        temperature=args.temperature,
        eos=eos,
        width=args.width,
        top_k=args.top_k,
        depth=args.depth,
        atol=args.atol,
        rtol=args.rtol,
        max_tv=args.max_tv,
    )
    for _ in range(args.warmup):
        for arm, budget in arms:
            generate(
                target,
                draft,
                inputs[0],
                arm=arm,
                budget=budget,
                max_new_tokens=min(16, args.max_new_tokens),
                seed=args.seed,
                **settings,
            )
    repository = Path(__file__).resolve().parents[2]
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=repository,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                cwd=repository,
                stderr=subprocess.DEVNULL,
            )
        )
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
        git_dirty = None
    try:
        driver_versions = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()
            if target.device.type == "cuda"
            else None
        )
    except (OSError, subprocess.CalledProcessError):
        driver_versions = None
    report = {
        "schema_version": 1,
        "experiment": "dflash_unigram_tree_hf_reference",
        "ubtree_trained_selector": False,
        "vllm_serving_integration": False,
        "cuda_graphs": False,
        "concurrency": 1,
        "eos_token_ids": eos,
        "target_top_p": 1.0,
        "target_top_k": "disabled",
        "status": "running",
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "prompts_sha256": hashlib.sha256(raw).hexdigest(),
        "environment": {
            "git_revision": git_revision,
            "git_dirty": git_dirty,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "driver_versions": driver_versions,
            "gpu": torch.cuda.get_device_name(target.device)
            if target.device.type == "cuda"
            else None,
            "target_revision": target.config._commit_hash,
            "draft_revision": draft.config._commit_hash,
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    jobs = [
        (repeat, i, arm, budget)
        for repeat in range(args.repeats)
        for i in range(len(inputs))
        for arm, budget in arms
    ]
    random.Random(args.seed).shuffle(jobs)
    try:
        for repeat, i, arm, budget in jobs:
            result = generate(
                target,
                draft,
                inputs[i],
                arm=arm,
                budget=budget,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed + repeat * len(inputs) + i,
                profile=args.mode == "profile",
                check_rounds=args.check_rounds if args.mode == "check" else 0,
                **settings,
            )
            result.update(
                {
                    "prompt_index": i,
                    "prompt_id": prompts[i].get("id", str(i)),
                    "prompt_tokens": inputs[i].shape[1],
                    "repeat": repeat,
                    "budget": budget,
                    "text": tokenizer.decode(
                        result["output_token_ids"], skip_special_tokens=True
                    ),
                }
            )
            report["results"].append(result)
            print(
                f"{len(report['results'])}/{len(jobs)}: {arm} B={budget} prompt={i}",
                flush=True,
            )
            save()
        if args.mode == "check" or args.temperature == 0:
            baselines = {
                (r["prompt_index"], r["repeat"]): r["output_token_ids"]
                for r in report["results"]
                if r["arm"] == "target"
            }
            for r in report["results"]:
                if r["output_token_ids"] != baselines[r["prompt_index"], r["repeat"]]:
                    raise AssertionError(
                        f"target sequence mismatch: {r['arm']} B={r['budget']} "
                        f"prompt={r['prompt_index']} repeat={r['repeat']}; "
                        "inspect numerical/logit differences before benchmarking"
                    )
            report["sequence_parity_passed"] = True
        report["status"] = "complete"
        save()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
