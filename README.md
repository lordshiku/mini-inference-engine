# mini-inference-engine

I wanted to understand what actually happens between "here are some model weights" and "here are tokens coming out at N tokens/sec", so I built the serving stack for Qwen2.5-0.5B myself and measured it against vLLM on the same GPU.

Results on an RTX A5000, same model, prompt, and concurrency levels:

| concurrency | this engine (tok/s) | vLLM (tok/s) | ratio |
| --- | --- | --- | --- |
| 1 | 278 | 415 | 0.67 |
| 4 | 748 | 1437 | 0.52 |
| 8 | 1140 | 2840 | 0.40 |
| 16 | 1525 | 5499 | 0.28 |
| 32 | 1812 | 9040 | 0.20 |

To run it yourself, see [QUICKSTART.md](QUICKSTART.md).

Both bf16. At concurrency 1 the engine is within 1.5x of vLLM. The gap widens with concurrency because vLLM has paged attention and batched prefill and this engine doesn't (see "What's missing" below). Everything in this table reproduces within 10% across runs; the full set of numbers, including the fp32 baseline before the CUDA-graph work, is in [BENCHMARKS.md](BENCHMARKS.md).

## What's in it

**Forward pass from the raw weights.** Qwen2.5's architecture (RMSNorm, RoPE, grouped-query attention with 14 Q heads over 2 KV heads, SwiGLU MLP, tied embeddings) implemented in plain PyTorch straight from the safetensors file. `transformers` is used only in tests, as the reference: logits match it to within 1e-3 in fp32 on every test prompt.

**KV cache.** Decode steps only compute attention for the new token against cached keys/values. Measured per-step speedup is 5.7x versus recomputing a 256-token context.

**CUDA-graph decode.** This turned out to be the biggest single win. A decode step for a model this small is ~430 kernel launches, and in eager mode the launch overhead dominates the actual math: the KV cache only showed a 1.5x gain until the launches were captured into a graph and replayed. Going from eager fp32 to bf16 + graph took single-stream throughput from 45 to 278 tok/s.

**Continuous batching.** A scheduler with a request queue. Each step it admits waiting requests into free rows of a preallocated batched cache, runs one decode step across all active sequences at ragged lengths, evicts the ones that finished, and compacts. One CUDA graph is captured per batch size. Batched greedy output is token-for-token identical to running each prompt alone, which the tests check on 8 prompts.

**Fused RMSNorm in Triton.** Replaces the PyTorch RMSNorm everywhere in the model. Up to 3.7x faster at large row counts, checked against the PyTorch version to 1e-4 across random shapes.

## Layout

```
engine/
  model.py        forward pass
  cache.py        KVCache (single sequence) and BatchedKVCache
  graph_decode.py CUDA-graph decode over a static cache
  scheduler.py    Engine: request queue + continuous batching loop
  generate.py     greedy / temperature / top-p sampling
  kernels.py      Triton RMSNorm
tests/            one file per milestone
bench/            benchmark scripts
```

Weights go in `weights/qwen2.5-0.5b/` (model config: hidden 896, 24 layers, head_dim 64, vocab 151936, RoPE theta 1e6).

## Running it

Needs a CUDA GPU and the weights.

```bash
pytest                          # everything
pytest tests/test_logits.py     # forward pass vs transformers
pytest tests/test_cache.py      # cached == uncached; graph == eager
pytest tests/test_batching.py   # batched == standalone
pytest tests/test_kernels.py    # Triton RMSNorm vs PyTorch

PYTHONPATH=. python bench/bench_single.py       # prefill + cached vs uncached decode
PYTHONPATH=. python bench/bench_concurrent.py   # throughput at concurrency 1..32
PYTHONPATH=. python bench/bench_kernels.py      # Triton vs eager RMSNorm
PYTHONPATH=. python bench/bench_vllm.py         # same workload through vLLM
```

## What's missing

Scoped out on purpose to keep this to a week:

- **Paged KV cache.** The batched cache is contiguous and preallocated per row, so memory isn't packed across sequences. This is most of the gap to vLLM at high concurrency.
- **Batched prefill.** New requests are prefilled one at a time before joining the batch.
- **Speculative decoding**, quantization, an HTTP server, multi-GPU.

## How it was built

Spec-first: [SPEC.md](SPEC.md) defines six milestones, each with an acceptance test it had to pass before the next one started (logit match, cache == no-cache, batched == standalone, kernel allclose, benchmarks reproducible within 10%). Claude Code did most of the implementation against that spec; a separate fresh-context reviewer agent checked each milestone's diff against the spec before it was committed. The reviewer caught one real problem, in the KV-cache benchmark, where the headline speedup was crediting CUDA graphs rather than the cache. That got reworked into a fair comparison. The decisions and reviewer findings for each milestone are in [DEVLOG.md](DEVLOG.md).
