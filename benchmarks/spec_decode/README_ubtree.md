# UBTree feasibility prototype

Tracks [HanFa/vllm#2](https://github.com/HanFa/vllm/issues/2).
Algorithm reference: [UBTree, equations 4–12 and Appendix A](https://arxiv.org/html/2609.39972v1).

## Implemented

`vllm/v1/spec_decode/ubtree.py` provides an executable correctness reference:

- Bias-free, two-layer SiLU codebook precomputation from target embeddings.
- Parallel bigram edge scoring with trained depth-dependent unigram/bigram scales.
- Per-predecessor normalization, bounded depth-wise expansion, and global pruning.
- Deterministic ancestor-first ties and distinct nodes for repeated tokens under different parents.
- Tree-local attention masks and depth-based position IDs.
- Traversal of target samples, including EOS, output limits, bonus-token accounting, and the ordered KV rows to retain.

This is not a registered `vllm serve` method. Search and traversal use Python/CPU
operations and synchronize if given CUDA inputs. They are correctness oracles for
subsequent kernels, not a performance implementation. There is no assumed official
UBTree checkpoint mapping, vLLM target backend, batching, or CUDA graph support.
Candidate scores must be finite; masked-out candidates must be removed before
using this initial reference. Official UBTree weights and performance results
remain outstanding; GPU reference correctness results are recorded below.

An additional **DFlash unigram-tree ablation** now runs a real Qwen3 target with
public DFlash weights through Transformers. It implements tree attention, target
sampling and actual accepted-path KV compaction. It has no trained UBTree selector
and cannot establish UBTree's paper-level benefit. See the GPU procedure below.

## Reference tests

The tests enumerate tiny candidate trees independently, compare calibrated scores
against scalar formulas, and compare tree attention with independent causal
prefix attention. They also check branch isolation and the KV commit plan.

In a configured vLLM development environment:

```bash
.venv/bin/python -m pytest tests/v1/spec_decode/test_ubtree.py -q
```

The primitive tests need PyTorch and pytest. Real-decoder tests additionally use
Transformers 4.57.3 with tiny random Qwen3 weights. Without a vLLM engine build,
they can be run without the engine-wide fixtures:

```bash
.venv/bin/python -m pytest --noconftest tests/v1/spec_decode/test_ubtree.py -q
```

The latter is a standalone reference check; it does not validate engine integration
or replace full model tests once a serving path exists.

## Reproducible reference microbenchmark

From the repository root:

```bash
.venv/bin/python -m benchmarks.spec_decode.ubtree \
  --synthetic --budgets 16 32 64 --search-width 12 \
  --warmup 3 --repeats 20 --seed 0 --threads 1 \
  --save-input /tmp/ubtree-input.pt --output /tmp/ubtree-reference.json

.venv/bin/python -m benchmarks.spec_decode.ubtree \
  --input /tmp/ubtree-input.pt --budgets 16 32 64 \
  --output /tmp/ubtree-replay.json
```

The JSON contains raw CPU timing samples, medians, environment details, and
constructed trees. Synthetic mode is explicitly labeled. No target, proposer,
or KV-commit latency is measured; no output acceptance or model speedup is claimed.
Timings include validation and CPU reference overhead, so they are not estimates
of an optimized GPU implementation's cost.

For real tensor replay, use `torch.save` with a dictionary containing exactly:

| Key | Meaning / shape |
| --- | --- |
| `candidate_ids` | Target vocabulary IDs `[depth, K]`, unique at each depth |
| `unary_logits` | Proposer logits at those candidates `[depth, K]` |
| `projected_hidden` | Trained hidden projection `P(h_i)`, `[depth, rank]` |
| `predecessor_codebook` | Precomputed phi, `[vocabulary, rank]` |
| `successor_codebook` | Precomputed psi, `[vocabulary, rank]` |
| `anchor_token_id` | Python integer for the already sampled root token |
| `log_unigram_scale` | Trained rho, `[depth]`; the scorer exponentiates it |
| `log_bigram_scale` | Trained kappa, `[depth]`; the scorer exponentiates it |

This is an experimental tensor-replay schema, not a Hugging Face checkpoint
format. Keep the source model/checkpoint revisions and prefix alongside real dumps.

## Target verification contract

The compact tree always includes root row 0. Its budget counts only non-root
nodes. Every row may attend to the verified context, its ancestors, and itself;
sibling branches must be invisible. Position IDs are anchor position plus depth.

For each row, the target supplies the next-token sample under that exact prefix.
`follow_target_samples` walks from the root and retains only matching children.
It does not search for the longest among independently sampled full paths.
The first unmatched sample becomes the bonus token. `kv_row_indices` contains
the anchor and matched descendants in order; the bonus has no computed KV row.
A backend must gather those rows safely into the committed prefix, discard the
other branches, and feed a nonterminal bonus in the next round. This module only
returns the plan. Target sampling (including history-dependent processors) and
cache mutation remain the backend's responsibility. The standalone adapter in
`hf_tree.py` implements this contract for full-attention Qwen3 DynamicCache.

## DFlash experiment on one GPU

Use a dedicated reference environment; its Transformers pin should not replace
the dependencies of a working vLLM server environment. From this checkout:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend=auto \
  torch 'transformers==4.57.3' 'accelerate>=1,<2' 'datasets>=3,<4' pytest
```

The checkpoint's Python helper imports `datasets`, even for inference. The
harness loads that public code with `trust_remote_code=True` at a pinned revision.
No vLLM engine build is required for the standalone reference.

| Component | Checkpoint | Pinned revision |
| --- | --- | --- |
| Target | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| DFlash | `z-lab/Qwen3-8B-DFlash-b16` | `9b41424b7109f9c5413454f481b09a82b85333f4` |

Defaults are BF16, one CUDA device, SDPA, one CPU thread, thinking disabled, and fresh caches per
request. Both models stay resident for all three reference arms. Initial weight
downloads require about 19 GB of disk; weights alone need roughly 19 GB of GPU
memory, with additional space needed for KV, activations and attention masks.
Choose context lengths according to available VRAM. The included three prompts
are smoke inputs, not the held-out benchmark in issue #2.

For strict correctness on a 32 GB GPU, start with Qwen3-4B in FP32. This is a
diagnostic target/precision substitution, not the planned Qwen3-8B BF16 performance
experiment. These commands include the root-only budget and use the matching
public DFlash checkpoint:

```bash
for ubtree_temperature in 0 1; do
  .venv/bin/python -m benchmarks.spec_decode.dflash_tree \
    --mode check --target Qwen/Qwen3-4B \
    --target-revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --draft z-lab/Qwen3-4B-DFlash-b16 \
    --draft-revision b74e3a329c4d963783143b1e970d95b002be72bd \
    --dtype float32 --atol 0.0001 --rtol 0.0001 --max-tv 0.0001 \
    --prompts benchmarks/spec_decode/ubtree_prompts.jsonl \
    --budgets 0 16 --max-new-tokens 32 --repeats 1 \
    --temperature "$ubtree_temperature" \
    --output "/tmp/dflash-tree-qwen4b-fp32-t${ubtree_temperature}.json"
done
```

Check mode compares every node's logits in the first round against an independent
causal prefix, then requires complete generated-token parity across all arms.
Only reached nodes consume sampling RNG, allowing the same per-token draws in
each arm. Reports also include maximum probability total variation (TV) at T=1
and argmax mismatch counts. `--max-tv` supplies an additional diagnostic bound,
independent of the raw logit tolerance. BF16/kernel numerical differences can
still change a sampled token; investigate failures using FP32 and smaller models.
This coupling check is not a statistical distribution-equivalence study.
`--check-rounds` increases the number of rounds checked node by node. Logit checks
are expensive, and check-mode timing is explicitly invalid for performance use.

After the chosen configuration passes its correctness gates, collect
uninstrumented reference timings and a separate profile. The commands below
select the default Qwen3-8B BF16 pair; its sampled numerical gate remains open
(see below), so these are prepared commands, not a qualified performance result:

```bash
.venv/bin/python -m benchmarks.spec_decode.dflash_tree \
  --mode bench --prompts benchmarks/spec_decode/ubtree_prompts.jsonl \
  --budgets 16 32 64 --max-new-tokens 256 --repeats 3 --temperature 1 \
  --output /tmp/dflash-tree-timings.json

.venv/bin/python -m benchmarks.spec_decode.dflash_tree \
  --mode profile --prompts benchmarks/spec_decode/ubtree_prompts.jsonl \
  --budgets 16 32 64 --max-new-tokens 256 --repeats 1 --temperature 1 \
  --output /tmp/dflash-tree-profile.json
```

Each invocation runs target-only, DFlash greedy-chain proposals, and DFlash
unigram trees for every budget. The target samples at the requested temperature
in every arm. Candidate scores are renormalized over top-K at each depth; no
bigram scores or learned depth calibration are used. `--depth` caps candidate
depth while preserving the drafter's trained 16-token forward shape. Repeat the
baseline sweep with depths 1, 3, 7 and 15 before choosing a tuned chain baseline.
There are no repetition penalties, top-p/top-k target filters or structured
output processors in this reference.

Raw JSON records checkpoint/software revisions, dataset hash, randomized run
order, output tokens/text, TTFT, decode TPOT, round yield, verified nodes per
committed token and peak allocated GPU memory. Decode timing starts after the
first token and divides by the remaining N-1 tokens. Profile mode adds synchronized
proposer/build/verify/sample-and-commit stage timings; it is not valid for speedup
comparison. Tree mask preparation is included in verification time. Reports
label all results as a Transformers unigram-tree reference. Python search, GPU/CPU
transfers, host sampling and DynamicCache copies are included in total time.
Greedy benchmark/profile runs additionally require sequence parity before being
marked complete. Sampled runs require a separate correctness qualification;
check-mode timings must never be used to claim speedup.

### Numerical validation status

Initial GPU validation uses an RTX 5090 (32 GB), driver 575.57.08, PyTorch
2.9.1+cu128 and Transformers 4.57.3. The strict Qwen3-4B FP32 greedy and sampled
smoke tests pass per-prefix checks and full sequence parity for all 24 combinations
of two temperatures, three prompts and four arms (target, chain, root-only tree,
B=16 tree). Maximum checked logit error is 9.54e-5, probability TV at T=1 is
1.32e-5, and no checked node has an argmax mismatch. Peak allocated memory is
17.04 GiB. Each run generates at most 32 tokens; this is a correctness smoke test,
not a benchmark or a quality-equivalence study.

Qwen3-8B BF16 greedy sequence parity passes on the same three 32-token smoke
inputs with explicitly calibrated logit tolerances (`--atol 0.75 --rtol 0.02`).
The stricter default logit tolerance fails even for the ordinary DFlash chain.
An independent numerical control found ordinary BF16 cached versus full-prefix
decoding differences up to 0.375 in logits and about 0.031 in probability TV;
all 27 control prefixes retained their top prediction. These are numerical
diagnostics, not quality-equivalence guarantees.

The sampled BF16 run fails full sequence parity on the chat prompt. Its first-round
per-prefix probability checks pass, which does not establish later-round sampling
equivalence. The failed check remains a gate: do not report sampled BF16 support
or infer UBTree speedup from these smoke runs. A root-only tree now uses the
ordinary causal attention path, avoiding an unnecessary custom mask; broader
BF16 sampled qualification remains outstanding. FP16 produced non-finite draft
scores in a diagnostic probe and is not offered as a supported alternative.

For model-free interface regression checks, the opt-in test downloads only the
pinned checkpoint's Python source and uses tiny random weights:

```bash
VLLM_TEST_DFLASH_REFERENCE=1 .venv/bin/python -m pytest --noconftest \
  tests/v1/spec_decode/test_ubtree.py -q
```

## Optimized vLLM baselines

Run these in a separate, working vLLM GPU environment at the same fork revision.
For each arm, start a fresh server, warm it up, and run the identical JSONL prompts.
Keep the target revision, BF16, context limit and sampling settings fixed:

```bash
vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --dtype bfloat16 --max-model-len 8192 --no-enable-prefix-caching \
  --generation-config vllm
```

Use that command for target-only. For DFlash add:

```text
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3-8B-DFlash-b16","revision":"9b41424b7109f9c5413454f481b09a82b85333f4","num_speculative_tokens":15}'
```

For DSpark add instead:

```text
--speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_8b_block7","revision":"03326e5043815da1f81b109078b2889737c26017","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Benchmark each server, changing the result filename per arm/repeat:

```bash
vllm bench serve --model Qwen/Qwen3-8B \
  --backend openai --endpoint /v1/completions \
  --dataset-name custom --dataset-path benchmarks/spec_decode/ubtree_prompts.jsonl \
  --num-prompts -1 --custom-output-len 256 --disable-shuffle --no-oversample \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --extra-body '{"stop_token_ids":[151645,151643]}' \
  --temperature 1 --top-p 1 --top-k -1 --max-concurrency 1 \
  --save-result --save-detailed --result-filename target-c1-repeat1.json
```

Sweep supported draft lengths before selecting the strongest baseline. Repeat at
concurrency 8 and 32 if concurrency 1 warrants more work. HTTP serving timings
and the Transformers reference have different overheads: report them separately.
The reference diagnoses acceptance/cost; a vLLM UBTree speedup claim still requires
an implementation in the same engine and the trained UBTree checkpoint.

## Next milestones

1. Run the checkpoint-backed correctness checks above on a GPU, then collect
   DFlash unigram-tree acceptance and cost alongside tuned vLLM baselines.
2. Obtain official trained UBTree weights/reference code and establish parity.
3. Connect the verified tree/KV contract to vLLM's model runner and attention
   backend; optimize tree search and commit, then add CUDA graphs.
4. Run the held-out experiment and statistical decision criteria in issue #2.
