"""KV cache for incremental decoding.

During decode we only project the single new token; its key/value vectors are
appended here so attention can run against the whole history without recomputing
keys/values for past tokens.
"""

import torch


class KVCache:
    """Per-layer key/value history for one sequence (or a fixed batch).

    Stores keys/values with the model's number of key/value heads (before GQA
    repetition). Keys/values grow along the time dimension as tokens are decoded.
    """

    def __init__(self, num_layers):
        """Create an empty cache with one slot per decoder layer."""
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def update(self, layer, k, v):
        """Append new keys/values for `layer` and return the full history.

        k, v have shape (B, num_kv_heads, T_new, head_dim). Returns the cached
        keys/values covering all tokens seen so far for this layer.
        """
        if self.k[layer] is None:
            self.k[layer] = k
            self.v[layer] = v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    @property
    def seq_len(self):
        """Number of tokens currently stored (0 if the cache is empty)."""
        return 0 if self.k[0] is None else self.k[0].shape[2]
