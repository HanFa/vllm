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
checkpoint mapping, target backend, cache mutation, batching, or CUDA graph support.
Candidate scores must be finite; masked-out candidates must be removed before
using this initial reference. Official weights and GPU results remain outstanding.

## Reference tests

The tests enumerate tiny candidate trees independently, compare calibrated scores
against scalar formulas, and compare tree attention with independent causal
prefix attention. They also check branch isolation and the KV commit plan.

In a configured vLLM development environment:

```bash
.venv/bin/python -m pytest tests/v1/spec_decode/test_ubtree.py -q
```

These particular tests need PyTorch, pytest, and vLLM's lightweight import
dependencies (including NumPy and regex). On a CPU host without a vLLM engine
build, they can be run without the engine-wide fixtures:

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
cache mutation remain the backend's responsibility.

## Next milestones

1. Obtain official trained weights/reference code and establish checkpoint parity.
2. Connect target tree attention on one dense model and test target logits and KV
   commit against ordinary autoregressive decoding.
3. Measure uninstrumented end-to-end decoding against the tuned same-target
   baselines in issue #2, separately from profiling.
4. Optimize only after the acceptance-versus-round-cost experiment demonstrates
   headroom; then add CUDA graphs and serving integration.
