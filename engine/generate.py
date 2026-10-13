"""Token sampling and single-sequence generation.

Provides greedy / temperature / top-p sampling and a greedy generation loop that
can run either with a KV cache (fast, incremental) or without one (recompute the
whole sequence each step), used to prove the cache is correct.
"""

import torch

from engine.cache import KVCache


def sample_next(logits, temperature=0.0, top_p=1.0):
    """Pick the next token id from the last-position logits (shape (B, vocab)).

    temperature == 0 means greedy (argmax). Otherwise scale logits by the
    temperature and, if top_p < 1, restrict sampling to the smallest set of
    tokens whose probabilities sum to at least top_p (nucleus sampling).
    """
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens up to and including the one that crosses top_p.
        remove = cumulative - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, choice).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate_greedy(model, input_ids, max_new_tokens, use_cache=True,
                    stop_on_eos=True):
    """Greedily generate up to `max_new_tokens` tokens for one prompt.

    With `use_cache`, the prompt is prefilled once and each step feeds only the
    new token. Without it, the full sequence is re-run every step. Returns the
    list of generated token ids (excluding the prompt).
    """
    eos = model.cfg.eos_token_id
    generated = []

    if use_cache:
        cache = KVCache(model.cfg.num_hidden_layers)
        logits = model.forward(input_ids, cache=cache, start_pos=0)
        for i in range(max_new_tokens):
            next_id = torch.argmax(logits[:, -1, :], dim=-1)
            token = next_id.item()
            generated.append(token)
            if stop_on_eos and token == eos:
                break
            if i < max_new_tokens - 1:
                logits = model.forward(next_id.view(1, 1), cache=cache,
                                       start_pos=cache.seq_len)
    else:
        seq = input_ids
        for _ in range(max_new_tokens):
            logits = model.forward(seq, cache=None, start_pos=0)
            next_id = torch.argmax(logits[:, -1, :], dim=-1)
            token = next_id.item()
            generated.append(token)
            if stop_on_eos and token == eos:
                break
            seq = torch.cat([seq, next_id.view(1, 1)], dim=1)

    return generated
