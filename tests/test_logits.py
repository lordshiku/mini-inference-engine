"""Milestone 1 acceptance test.

Compare our from-scratch forward pass against the transformers reference on 5
varied prompts. Pass if max abs logit diff < 1e-3 in fp32 on every prompt.
`transformers` is the reference here (allowed in tests only).
"""

import os

import pytest
import torch

from engine.model import Qwen2Model, load_tokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "qwen2.5-0.5b")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
    "import numpy as np\n",
    "Q: What is 2 + 2?\nA:",
]


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda"


@pytest.fixture(scope="module")
def ours(device):
    return Qwen2Model.from_pretrained(MODEL_DIR, device=device, dtype=torch.float32)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODEL_DIR)


@pytest.fixture(scope="module")
def reference(device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    return model


def test_logits_match(ours, reference, tokenizer, device):
    max_diff_overall = 0.0
    for prompt in PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        with torch.no_grad():
            ours_logits = ours.forward(ids)
            ref_logits = reference(ids).logits
        diff = (ours_logits - ref_logits).abs().max().item()
        max_diff_overall = max(max_diff_overall, diff)
        assert diff < 1e-3, f"prompt {prompt!r}: max abs diff {diff} >= 1e-3"
    print(f"\nmax abs logit diff across all prompts: {max_diff_overall:.2e}")
