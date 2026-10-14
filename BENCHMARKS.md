# Benchmarks

GPU: **NVIDIA RTX A5000**. All numbers in fp32. Reproduce with the commands shown.

## Milestone 2 — KV cache (single sequence)

`PYTHONPATH=. python bench/bench_single.py` — 2026-10-13, NVIDIA RTX A5000

Context = 256 tokens, decode = 64 tokens.

| metric | value |
| --- | --- |
| prefill (256 tok) | 27.5 ms |
| decode cached, end-to-end (CUDA graph) | 189.9 tok/s |
| decode cached, eager KV cache | 46.1 tok/s |
| per-token compute, cached step (graphed) | 210.7 tok/s |
| per-token compute, uncached step (graphed) | 37.1 tok/s |
| **cached / uncached speedup (fair, both graphed)** | **5.69x** |

Acceptance (cached decode >= 5x uncached at 256-token context): **PASS**.

### How the speedup is measured (and why it is fair)

The acceptance ratio compares the per-token *compute* of a cached decode step
(project only the new token, attend against the cached K/V history) against an
uncached step (recompute the full 256-token forward, as you must with no cache).

Both sides are CUDA-graph captured and timed as pure replays. This matters
because for a 0.5B model an eager decode step is dominated by CPU kernel-launch
overhead (~430 launches/step), not compute — eager cached decode is only ~1.5x
eager uncached, which would understate the cache. Graph-capturing *both* sides
removes that launch overhead equally, so the remaining 5.69x is purely the
algorithmic work the KV cache saves (1 new token vs 256 recomputed tokens).

The CUDA-graph decoder is numerically identical to eager cached decode, verified
token-for-token in `tests/test_cache.py::test_graph_decode_matches_eager`. The
uncached step uses a fixed 256-token context, so the ratio is conservative (real
uncached generation grows the context and gets slower).

Reproducibility: two consecutive runs agreed within <2% on every number
(speedup 5.69x then 5.65x).

## Milestone 3 — continuous batching (throughput vs concurrency)

`PYTHONPATH=. python bench/bench_concurrent.py` — 2026-10-13, NVIDIA RTX A5000

Prompt = 11 tokens, max_tokens = 64, max_batch = concurrency. Throughput is total
tokens/sec across all sequences.

| concurrency | throughput (tok/s) | speedup vs 1 |
| --- | --- | --- |
| 1 | 45.5 | 1.00x |
| 4 | 171.6 | 3.77x |
| 8 | 324.6 | 7.13x |
| 16 | 585.4 | 12.86x |
| 32 | 917.3 | 20.14x |

Acceptance (throughput@16 >= 3x throughput@1): **PASS** (12.86x).

Continuous batching (not the fallback static batching): the scheduler admits
waiting requests into free rows each step and evicts/compacts finished ones, so
rows are reused mid-run. The strong scaling comes from this model being
launch-bound per step — folding more sequences into one batched step adds little
step cost, so tokens/sec rises nearly linearly with the batch until compute
starts to bind around concurrency 32.

Reproducibility: a second run gave 46.2 / 176.2 / 329.9 / 582.9 / 944.4 tok/s
(throughput@16 = 12.62x), within ~3% of the above on every level.

## Milestone 4 — Triton fused RMSNorm

`PYTHONPATH=. python bench/bench_kernels.py` — 2026-10-13, NVIDIA RTX A5000

Fused Triton RMSNorm vs the PyTorch implementation it replaces (one program per
row; square/mean/rsqrt/scale fused into a single launch). Per-call latency, timed
with CUDA events (min over 5 blocks of 500 calls):

| shape (rows x hidden) | PyTorch (us) | Triton (us) | speedup |
| --- | --- | --- | --- |
| 1 x 896 | 55.6 | 35.7 | 1.56x |
| 32 x 896 | 56.2 | 35.1 | 1.60x |
| 256 x 896 | 59.2 | 36.0 | 1.64x |
| 4096 x 896 | 166.0 | 45.2 | 3.68x |

The kernel wins at every size by collapsing several PyTorch kernel launches into
one; the gap widens with row count as the PyTorch path becomes bandwidth-bound
while the fused kernel stays efficient. Correctness is verified within atol 1e-4
over 10+ shapes in `tests/test_kernels.py`, and the kernel is enabled by default
(`engine.model.USE_TRITON_RMSNORM`), so all earlier milestone tests run through
it and still pass (6 passed).

Reproducibility: a second run gave 1.55 / 1.56 / 1.59 / 3.67x, within ~3% of the
above on every shape.

## Milestone 5 / 6 — this engine vs vLLM (same model, prompt, concurrency)

`PYTHONPATH=. python bench/bench_concurrent.py` and
`PYTHONPATH=. python bench/bench_vllm.py` — M5: 2026-10-13, M6: 2026-10-14,
NVIDIA RTX A5000.

Same model (Qwen2.5-0.5B), same prompt ("Once upon a time, in a land far away,"),
same concurrency levels, max_tokens = 64, total tokens/sec across all sequences.
vLLM runs its default bf16 with paged attention. The M5 engine column is fp32
eager; the M6 engine column is the same engine run in bf16 with the CUDA-graph
decode path inside the scheduler (see Milestone 6 below).

| concurrency | engine M5 fp32 (tok/s) | engine M6 bf16+graph (tok/s) | vLLM bf16 (tok/s) | M6 gap to vLLM |
| --- | --- | --- | --- | --- |
| 1 | 45.5 | 278.0 | 415.4 | 1.5x |
| 4 | 171.6 | 747.7 | 1436.7 | 1.9x |
| 8 | 324.6 | 1140.4 | 2839.9 | 2.5x |
| 16 | 585.4 | 1524.5 | 5499.1 | 3.6x |
| 32 | 917.3 | 1812.0 | 9039.7 | 5.0x |

Reproducibility: engine M6 second run gave 274.6 / 730.5 / 1091.8 / 1560.7 /
1890.4 tok/s (within ~4%); vLLM second run gave 399.2 / 1418.2 / 2811.7 / 5423.5 /
9403.6 tok/s (within ~4%). All benchmark numbers in this file reproduce across
two runs within 10%.

## Milestone 6 — closing the gap with vLLM (bf16 + CUDA-graph scheduler)

Two changes, both applied to the existing engine:

- **bf16 end to end.** `Qwen2Model.from_pretrained(..., dtype=torch.bfloat16)`
  runs the whole forward (and RoPE tables) in bf16. The fp32 path is kept as a
  flag; the logit test still loads fp32 and passes < 1e-3. The Triton RMSNorm was
  adjusted to round the normalized value to the input dtype before scaling by the
  weight, so it is bit-identical to the PyTorch reference in bf16 too.
- **CUDA-graph decode inside the scheduler.** The batched continuous-batching
  decode step is captured as a CUDA graph (one per batch size, cached) and
  replayed, collapsing ~hundreds of kernel launches per step into one. The graph
  runs a fixed `max_batch` rows (inactive rows are harmless dummies) and attends
  over the full `max_len` cache with a per-row mask, so shapes stay static.

Effect (concurrency 1 → 32): throughput rose from 45.5/171.6/324.6/585.4/917.3
(fp32 eager) to 278.0/747.7/1140.4/1524.5/1812.0 (bf16+graph). The biggest win is
at low concurrency — concurrency 1 went 6.1x faster and the gap to vLLM shrank
from ~9x to ~1.5x — because that regime was entirely launch-bound and the graph
removes that overhead. At high concurrency vLLM's paged attention and larger
effective batching pull ahead (gap ~5x at 32); closing that would need paged KV
and a batched prefill, which are out of scope.

The M3 acceptance still holds in this configuration: throughput@16 = 5.48x
throughput@1 (>= 3x). Correctness: the fp32 batching test still matches
standalone greedy token-for-token through the graph path, and a bf16 test
teacher-forces every engine-produced token against a single-pass reference and
requires it to be the greedy choice within a loose logit tolerance (bf16
near-ties stay within ~0.4; tol 2.0).
