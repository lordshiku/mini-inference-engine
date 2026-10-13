"""Milestone 5 benchmark: same workload as bench_concurrent.py, through vLLM.

Runs the identical model, prompt, concurrency levels (1, 4, 8, 16, 32), and
max_tokens as the M3 continuous-batching benchmark, but served by vLLM, to put
this engine next to a production inference engine. Reports total tokens/sec per
concurrency level. vLLM is a reference baseline (allowed outside engine/).

Note: vLLM uses its default optimized dtype (bf16) and paged attention; this
engine's numbers are fp32. The comparison shows where a simple from-scratch
engine lands relative to a heavily optimized one, not a dtype-matched contest.
"""

import argparse
import os
import time

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "qwen2.5-0.5b")
PROMPT = "Once upon a time, in a land far away,"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    llm = LLM(
        model=MODEL_DIR,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        max_model_len=512,
        disable_log_stats=True,
    )
    # ignore_eos so every sequence emits exactly max_tokens, matching our bench.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    # Warmup.
    llm.generate([PROMPT] * 2, SamplingParams(temperature=0.0, max_tokens=8,
                                              ignore_eos=True), use_tqdm=False)

    results = {}
    for c in args.concurrency:
        prompts = [PROMPT] * c
        start = time.perf_counter()
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        results[c] = total / elapsed

    print(f"GPU: {gpu}")
    print(f"prompt='{PROMPT}', max_tokens={args.max_tokens}\n")
    print("| concurrency | vLLM throughput (tok/s) | speedup vs 1 |")
    print("| --- | --- | --- |")
    base = results[args.concurrency[0]]
    for c in args.concurrency:
        print(f"| {c} | {results[c]:.1f} | {results[c] / base:.2f}x |")


if __name__ == "__main__":
    main()
