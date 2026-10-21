## Quick start

#read the readme first!

Needs a CUDA GPU. Install the engine's dependencies and pull the weights (no HF account needed):

```bash
pip install torch safetensors tokenizers triton huggingface_hub
pip install transformers pytest   # only for the tests / reference comparison
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='weights/qwen2.5-0.5b')"
```

Run everything from the repo root with `PYTHONPATH=.`.

**Generate text for one prompt** (KV-cached greedy decode):

```python
import torch
from engine.model import Qwen2Model, load_tokenizer
from engine.generate import generate_greedy

model = Qwen2Model.from_pretrained("weights/qwen2.5-0.5b", device="cuda", dtype=torch.bfloat16)
tok = load_tokenizer("weights/qwen2.5-0.5b")

ids = torch.tensor([tok.encode("Once upon a time,").ids], device="cuda")
out = generate_greedy(model, ids, max_new_tokens=64)
print(tok.decode(out))
```

**Same thing through the CUDA-graph decoder** (fastest single-stream path):

```python
from engine.graph_decode import GraphDecoder

dec = GraphDecoder(model, max_len=256)       # prompt + new tokens must fit in max_len
out = dec.generate(ids, max_new_tokens=64)
print(tok.decode(out))
```

**Many prompts at once** through the continuous-batching engine:

```python
from engine.scheduler import Engine

engine = Engine(model, tokenizer=tok, max_batch=8, max_len=512)
prompts = ["The capital of France is", "def fibonacci(n):", "Q: What is the tallest mountain?\nA:"]
futures = [engine.submit(p, max_tokens=64) for p in prompts]
engine.run()                                  # blocks until every request has finished
for p, f in zip(prompts, futures):
    print(p, "->", tok.decode(f.result()))
```

`submit` takes a string or a list of token ids and returns a `concurrent.futures.Future`; `run()` drives the admit → decode → evict loop until the queue is empty and resolves each future to the generated token ids. `Engine` runs the batched decode step through a captured CUDA graph by default; pass `use_cuda_graph=False` for the eager path.

Sampling other than greedy (temperature / top-p) is available as `engine.generate.sample_next(logits, temperature, top_p)` for wiring into your own loop.
