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


class BatchedKVCache:
    """Preallocated KV cache for continuous batching: one row per active sequence.

    Keys/values live in fixed buffers of shape (num_layers, max_batch,
    num_kv_heads, max_len, head_dim). Each row has its own fill length, so
    sequences at different (ragged) positions share one batched tensor. New
    tokens are scattered into each row's current position, and attention masks
    out the unfilled tail. Rows can be compacted when sequences finish so the
    active rows always form a dense prefix [0, num_active).
    """

    def __init__(self, num_layers, max_batch, num_kv_heads, max_len, head_dim,
                 device, dtype):
        """Allocate the key/value buffers and the per-row length counter."""
        self.k = torch.zeros(num_layers, max_batch, num_kv_heads, max_len,
                             head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(num_layers, max_batch, num_kv_heads, max_len,
                             head_dim, device=device, dtype=dtype)
        self.lengths = torch.zeros(max_batch, dtype=torch.long, device=device)

    def write_prefill(self, row, k_per_layer, v_per_layer, length):
        """Load a freshly prefilled sequence's K/V (all layers) into `row`.

        k_per_layer[l] / v_per_layer[l] have shape (1, num_kv_heads, length,
        head_dim), as produced by a single-sequence prefill.
        """
        for l in range(self.k.shape[0]):
            self.k[l, row, :, :length, :] = k_per_layer[l][0]
            self.v[l, row, :, :length, :] = v_per_layer[l][0]
        self.lengths[row] = length

    def write_token(self, layer, rows, positions, k, v):
        """Scatter one new token's K/V for the given rows at their positions.

        k, v have shape (B, num_kv_heads, head_dim); rows/positions are (B,) long
        tensors selecting each sequence's slot and its current length index.
        """
        self.k[layer, rows, :, positions, :] = k
        self.v[layer, rows, :, positions, :] = v

    def compact(self, survivor_rows):
        """Move surviving rows to a dense prefix, preserving order.

        `survivor_rows` is the list of current row indices that remain active,
        in the order they should occupy rows 0, 1, 2, .... Returns the number of
        survivors. Row data is copied only when a survivor actually changes row.
        """
        for new_row, old_row in enumerate(survivor_rows):
            if new_row != old_row:
                self.k[:, new_row] = self.k[:, old_row]
                self.v[:, new_row] = self.v[:, old_row]
                self.lengths[new_row] = self.lengths[old_row]
        return len(survivor_rows)
