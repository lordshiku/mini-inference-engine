# Mini Inference Engine — Spec

From-scratch inference engine for Qwen2.5-0.5B. Weights at `weights/qwen2.5-0.5b/`.
GPU: NVIDIA RTX A5000 (or whatever `torch.cuda.get_device_name(0)` reports — record it in BENCHMARKS.md).

Work milestone by milestone. A milestone is done only when its acceptance test passes.
Do not start the next milestone until the current one is done. Commit after each milestone.

## Repo layout
- `engine/model.py` — Qwen2.5 forward pass in plain PyTorch (no `transformers` at runtime)
- `engine/cache.py` — KV cache
- `engine/scheduler.py` — continuous batching
- `engine/kernels.py` — Triton kernels
- `engine/generate.py` — sampling (greedy, temperature, top-p)
- `tests/` — one test file per milestone
- `bench/` — benchmark scripts; results go in `BENCHMARKS.md`

## Milestone 1 — Forward pass, verified
Load Qwen2.5-0.5B from safetensors. Implement: tokenizer (use the `tokenizers` library), embeddings,
RMSNorm, RoPE, grouped-query attention with causal mask, SwiGLU MLP, final norm, LM head. Tied embeddings if the config says so.
Acceptance: `tests/test_logits.py` compares logits on 5 varied prompts against `transformers`
AutoModelForCausalLM. Pass = max abs diff < 1e-3 in fp32 on every prompt.

## Milestone 2 — KV cache + benchmark harness
Add a KV cache so decode steps only compute attention for the new token.
Acceptance: (a) `tests/test_cache.py`: greedy generation of 64 tokens with cache == without cache,
token-for-token, on 5 prompts. (b) `bench/bench_single.py` reports prefill time, decode tokens/sec,
cached vs uncached; cached decode must be >= 5x uncached at 256-token context. Write results to BENCHMARKS.md.

## Milestone 3 — Continuous batching
Scheduler with a request queue. Each decode step: admit waiting requests up to `max_batch`, run one step
for all active sequences (ragged lengths, right-padding + attention mask), evict finished sequences, repeat.
Expose `Engine.submit(prompt, max_tokens) -> future` and `Engine.run()`.
Acceptance: (a) `tests/test_batching.py`: for each of 8 prompts, greedy output under batching == greedy output
alone. (b) `bench/bench_concurrent.py` reports throughput (total tokens/sec) at concurrency 1, 4, 8, 16, 32.
Throughput at 16 must be >= 3x throughput at 1. Add table to BENCHMARKS.md.
Fallback if stuck > 3 hours: static batching with eviction of finished sequences; note this in BENCHMARKS.md.

## Milestone 4 — Triton kernel
Fused RMSNorm in Triton replacing the PyTorch implementation. Stretch: RoPE kernel.
Acceptance: (a) `tests/test_kernels.py`: `torch.allclose` vs PyTorch RMSNorm, atol 1e-4, over 10 random shapes.
(b) `bench/bench_kernels.py`: kernel vs eager timing; record speedup in BENCHMARKS.md.
(c) All earlier tests still pass with the kernel enabled.

## Milestone 5 — vLLM comparison + README
Run `bench/bench_vllm.py`: same model, same prompts, same concurrency levels as M3, through vLLM.
Put both in one table in BENCHMARKS.md with the GPU named.
README.md: what this is (5 bullets), architecture, how to run tests and benchmarks, the comparison table,
a "Not implemented" section listing paged KV cache and speculative decoding.
Acceptance: `pytest` passes; every benchmark number in BENCHMARKS.md is reproducible across two runs within 10%.

## Non-goals
HTTP server, paged attention, speculative decoding, quantization, multi-GPU, training.# Mini Inference Engine — Spec

From-scratch inference engine for Qwen2.5-0.5B. Weights at `weights/qwen2.5-0.5b/`.
GPU: NVIDIA RTX A5000 (or whatever `torch.cuda.get_device_name(0)` reports — record it in BENCHMARKS.md).

Work milestone by milestone. A milestone is done only when its acceptance test passes.
Do not start the next milestone until the current one is done. Commit after each milestone.

## Repo layout
- `engine/model.py` — Qwen2.5 forward pass in plain PyTorch (no `transformers` at runtime)
- `engine/cache.py` — KV cache
- `engine/scheduler.py` — continuous batching
- `engine/kernels.py` — Triton kernels
- `engine/generate.py` — sampling (greedy, temperature, top-p)
- `tests/` — one test file per milestone
- `bench/` — benchmark scripts; results go in `BENCHMARKS.md`

## Milestone 1 — Forward pass, verified
Load Qwen2.5-0.5B from safetensors. Implement: tokenizer (use the `tokenizers` library), embeddings,
RMSNorm, RoPE, grouped-query attention with causal mask, SwiGLU MLP, final norm, LM head. Tied embeddings if the config says so.
Acceptance: `tests/test_logits.py` compares logits on 5 varied prompts against `transformers`
AutoModelForCausalLM. Pass = max abs diff < 1e-3 in fp32 on every prompt.

## Milestone 2 — KV cache + benchmark harness
Add a KV cache so decode steps only compute attention for the new token.
Acceptance: (a) `tests/test_cache.py`: greedy generation of 64 tokens with cache == without cache,
token-for-token, on 5 prompts. (b) `bench/bench_single.py` reports prefill time, decode tokens/sec,
cached vs uncached; cached decode must be >= 5x uncached at 256-token context. Write results to BENCHMARKS.md.

## Milestone 3 — Continuous batching
Scheduler with a request queue. Each decode step: admit waiting requests up to `max_batch`, run one step
for all active sequences (ragged lengths, right-padding + attention mask), evict finished sequences, repeat.
Expose `Engine.submit(prompt, max_tokens) -> future` and `Engine.run()`.
Acceptance: (a) `tests/test_batching.py`: for each of 8 prompts, greedy output under batching == greedy output
alone. (b) `bench/bench_concurrent.py` reports throughput (total tokens/sec) at concurrency 1, 4, 8, 16, 32.
Throughput at 16 must be >= 3x throughput at 1. Add table to BENCHMARKS.md.
Fallback if stuck > 3 hours: static batching with eviction of finished sequences; note this in BENCHMARKS.md.

## Milestone 4 — Triton kernel
Fused RMSNorm in Triton replacing the PyTorch implementation. Stretch: RoPE kernel.
Acceptance: (a) `tests/test_kernels.py`: `torch.allclose` vs PyTorch RMSNorm, atol 1e-4, over 10 random shapes.
(b) `bench/bench_kernels.py`: kernel vs eager timing; record speedup in BENCHMARKS.md.
(c) All earlier tests still pass with the kernel enabled.

## Milestone 5 — vLLM comparison + README
Run `bench/bench_vllm.py`: same model, same prompts, same concurrency levels as M3, through vLLM.
Put both in one table in BENCHMARKS.md with the GPU named.
README.md: what this is (5 bullets), architecture, how to run tests and benchmarks, the comparison table,
a "Not implemented" section listing paged KV cache and speculative decoding.
Acceptance: `pytest` passes; every benchmark number in BENCHMARKS.md is reproducible across two runs within 10%.

## Non-goals
HTTP server, paged attention, speculative decoding, quantization, multi-GPU, training.# Mini Inference Engine — Spec

From-scratch inference engine for Qwen2.5-0.5B. Weights at `weights/qwen2.5-0.5b/`.
GPU: NVIDIA RTX A5000 (or whatever `torch.cuda.get_device_name(0)` reports — record it in BENCHMARKS.md).

Work milestone by milestone. A milestone is done only when its acceptance test passes.
Do not start the next milestone until the current one is done. Commit after each milestone.

## Repo layout
- `engine/model.py` — Qwen2.5 forward pass in plain PyTorch (no `transformers` at runtime)
- `engine/cache.py` — KV cache
- `engine/scheduler.py` — continuous batching
- `engine/kernels.py` — Triton kernels
- `engine/generate.py` — sampling (greedy, temperature, top-p)
- `tests/` — one test file per milestone
- `bench/` — benchmark scripts; results go in `BENCHMARKS.md`

## Milestone 1 — Forward pass, verified
Load Qwen2.5-0.5B from safetensors. Implement: tokenizer (use the `tokenizers` library), embeddings,
RMSNorm, RoPE, grouped-query attention with causal mask, SwiGLU MLP, final norm, LM head. Tied embeddings if the config says so.
Acceptance: `tests/test_logits.py` compares logits on 5 varied prompts against `transformers`
AutoModelForCausalLM. Pass = max abs diff < 1e-3 in fp32 on every prompt.

## Milestone 2 — KV cache + benchmark harness
Add a KV cache so decode steps only compute attention for the new token.
Acceptance: (a) `tests/test_cache.py`: greedy generation of 64 tokens with cache == without cache,
token-for-token, on 5 prompts. (b) `bench/bench_single.py` reports prefill time, decode tokens/sec,
cached vs uncached; cached decode must be >= 5x uncached at 256-token context. Write results to BENCHMARKS.md.

## Milestone 3 — Continuous batching
Scheduler with a request queue. Each decode step: admit waiting requests up to `max_batch`, run one step
for all active sequences (ragged lengths, right-padding + attention mask), evict finished sequences, repeat.
Expose `Engine.submit(prompt, max_tokens) -> future` and `Engine.run()`.
Acceptance: (a) `tests/test_batching.py`: for each of 8 prompts, greedy output under batching == greedy output
alone. (b) `bench/bench_concurrent.py` reports throughput (total tokens/sec) at concurrency 1, 4, 8, 16, 32.
Throughput at 16 must be >= 3x throughput at 1. Add table to BENCHMARKS.md.
Fallback if stuck > 3 hours: static batching with eviction of finished sequences; note this in BENCHMARKS.md.

## Milestone 4 — Triton kernel
Fused RMSNorm in Triton replacing the PyTorch implementation. Stretch: RoPE kernel.
Acceptance: (a) `tests/test_kernels.py`: `torch.allclose` vs PyTorch RMSNorm, atol 1e-4, over 10 random shapes.
(b) `bench/bench_kernels.py`: kernel vs eager timing; record speedup in BENCHMARKS.md.
(c) All earlier tests still pass with the kernel enabled.

## Milestone 5 — vLLM comparison + README
Run `bench/bench_vllm.py`: same model, same prompts, same concurrency levels as M3, through vLLM.
Put both in one table in BENCHMARKS.md with the GPU named.
README.md: what this is (5 bullets), architecture, how to run tests and benchmarks, the comparison table,
a "Not implemented" section listing paged KV cache and speculative decoding.
Acceptance: `pytest` passes; every benchmark number in BENCHMARKS.md is reproducible across two runs within 10%.

## Non-goals
HTTP server, paged attention, speculative decoding, quantization, multi-GPU, training.
