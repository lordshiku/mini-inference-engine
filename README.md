# mini-inference-engine

A small, from-scratch inference engine for **Qwen2.5-0.5B**, built in plain
PyTorch + Triton to understand how modern LLM serving actually works.

- **From-scratch forward pass** — Qwen2.5 (embeddings, RMSNorm, RoPE,
  grouped-query attention, SwiGLU MLP, tied LM head) implemented directly from
  the safetensors weights, with no `transformers` at runtime. Logits match the
  `transformers` reference to < 1e-3 (fp32).
- **KV cache** — incremental decode that only computes attention for the new
  token, plus a CUDA-graph decode path that makes the cache's per-token win
  visible (~5.7x vs recomputing the full context).
- **Continuous batching** — a scheduler with a request queue that admits,
  decodes, evicts, and compacts sequences every step at ragged lengths; batched
  output is token-for-token identical to single-sequence greedy.
- **Triton kernel** — a fused RMSNorm kernel that replaces the PyTorch version
  everywhere (up to ~3.7x faster, verified to atol 1e-4).
- **Honest benchmarks** — single-sequence, concurrency, kernel, and a
  head-to-head against **vLLM** on the same model/prompts/concurrency.

## Architecture

```
engine/
  model.py        Qwen2.5 forward pass (RMSNorm, RoPE, GQA + causal mask, SwiGLU, tied head)
  cache.py        KVCache (single sequence) and BatchedKVCache (continuous batching)
  generate.py     sampling (greedy / temperature / top-p) and greedy generation
  graph_decode.py CUDA-graph single-sequence decode over a static KV cache
  scheduler.py    Engine: request queue + continuous-batching run loop
  kernels.py      Triton fused RMSNorm
tests/            one test file per milestone
bench/            benchmark scripts; results in BENCHMARKS.md
```

The forward pass stores weights as plain tensors in a dict and wires explicit
building blocks together, favouring readability over abstraction. Decode reuses
the keys/values in the cache; continuous batching keeps one preallocated batched
cache with per-row lengths so ragged-length sequences share one batched step.

**Model config:** hidden 896, 24 layers, 14 query heads, 2 KV heads (GQA),
head_dim 64, SwiGLU intermediate 4864, RoPE theta 1e6, tied embeddings, vocab
151936. Weights live in `weights/qwen2.5-0.5b/`.

## Running the tests

```bash
pytest                      # all milestones (requires CUDA + the weights)
pytest tests/test_logits.py # M1: logits vs transformers (< 1e-3, fp32)
pytest tests/test_cache.py  # M2: cached == uncached; graph decode == eager
pytest tests/test_batching.py  # M3: batched == standalone greedy, 8 prompts
pytest tests/test_kernels.py   # M4: Triton RMSNorm vs PyTorch (atol 1e-4)
```

## Running the benchmarks

```bash
PYTHONPATH=. python bench/bench_single.py       # M2: prefill + cached vs uncached decode
PYTHONPATH=. python bench/bench_concurrent.py   # M3: throughput at concurrency 1..32
PYTHONPATH=. python bench/bench_kernels.py       # M4: Triton RMSNorm vs eager
PYTHONPATH=. python bench/bench_vllm.py          # M5: same workload through vLLM
```

All results, with GPU name and date, are in [BENCHMARKS.md](BENCHMARKS.md).

## This engine vs vLLM

Qwen2.5-0.5B, prompt "Once upon a time, in a land far away,", max_tokens 64,
total tokens/sec across all sequences, on an **NVIDIA RTX A5000**. This engine is
fp32; vLLM uses its default bf16 + paged attention.

| concurrency | this engine, fp32 (tok/s) | vLLM, bf16 (tok/s) |
| --- | --- | --- |
| 1 | 45.5 | 416.8 |
| 4 | 171.6 | 1427.4 |
| 8 | 324.6 | 2788.1 |
| 16 | 585.4 | 5434.0 |
| 32 | 917.3 | 9592.8 |

vLLM is ~9–10x faster in absolute throughput, but both scale with concurrency in
the same way — the batching win is real here too.

## Not implemented

Deliberately out of scope (see SPEC.md non-goals):

- **Paged KV cache** — this engine uses a contiguous preallocated per-row cache,
  not vLLM-style paged blocks, so memory is not shared/packed across sequences.
- **Speculative decoding** — no draft model or token acceptance; every token is
  produced by a single forward step.
- Also absent: HTTP server, quantization, multi-GPU, and training.
