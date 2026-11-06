"""Mellum numerical semantics and ATOM KV adapter; Step 0 uses masked SDPA.

No native attention kernel reads this adapter's private logical NHD cache view.
Switching the attention backend requires rebuilding the server and KV cache.
"""

import os

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.mellum.modeling_mellum import (
    MellumRMSNorm as HFRMSNorm,
    MellumRotaryEmbedding as HFRotaryEmbedding,
    repeat_kv,
)

from atom.model_ops.paged_attention import PagedAttention
from atom.utils.forward_context import get_forward_context


class MellumRMSNorm(HFRMSNorm):
    """Stable norm boundary, currently using the native HF torch implementation."""


class MellumRoPE(nn.Module):
    """Native HF per-layer-type RoPE; compute frequencies once per model forward."""

    def __init__(self, config):
        super().__init__()
        target = torch.get_default_device()
        with torch.device("cpu"):
            self.native = HFRotaryEmbedding(config)
        self.native.to(device=target)

    def _apply(self, fn, recurse=True):
        # Preserve FP32 frequencies even when a caller casts the whole model.
        frequencies = {name: value.clone() for name, value in self.native.named_buffers()}
        result = super()._apply(fn, recurse=recurse)
        for name, value in frequencies.items():
            target = getattr(self.native, name)
            self.native.register_buffer(name, value.to(device=target.device, dtype=torch.float32), persistent=False)
        return result

    def forward(self, hidden_states, positions):
        return {
            kind: self.native(hidden_states, positions.reshape(1, -1), kind)
            for kind in self.native.layer_types
        }


def causal_visibility(query_positions, key_positions, window=None):
    visible = key_positions[None, :] <= query_positions[:, None]
    if window is not None:
        visible &= key_positions[None, :] > query_positions[:, None] - window
    return visible


def masked_mellum_sdpa(query, key, value, query_positions, key_positions, window, scale):
    """[T,Hq,D], [S,Hkv,D] -> [T,Hq,D], with absolute position semantics."""
    groups = query.shape[1] // key.shape[1]
    keys = repeat_kv(key.transpose(0, 1)[None], groups)
    values = repeat_kv(value.transpose(0, 1)[None], groups)
    visible = causal_visibility(query_positions, key_positions, window)
    additive = torch.where(visible, 0.0, float("-inf")).to(query.dtype)[None, None]
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1)[None], keys, values,
        attn_mask=additive, dropout_p=0.0, scale=scale, is_causal=False,
    )
    return output[0].transpose(0, 1)


class MellumKVCacheView:
    """Private logical NHD view over ATOM-owned pages, independent of native layout."""

    def __init__(self, cache, num_kv_heads, head_dim):
        if cache.k_cache.dtype != torch.bfloat16 or cache.v_cache.dtype != torch.bfloat16:
            raise ValueError("Mellum Step 0 requires bf16 cache tensors")
        if cache.k_cache.numel() != cache.v_cache.numel():
            raise ValueError("Mellum K/V allocation sizes differ")
        self.key = cache.k_cache.view(-1, num_kv_heads, head_dim)
        self.value = cache.v_cache.view_as(self.key)
        self.block_size = self.key.shape[0] // cache.k_cache.shape[0]

    def write(self, slots, key, value):
        valid = slots >= 0
        self.key[slots[valid]] = key[valid]
        self.value[slots[valid]] = value[valid]

    def read(self, block_table, length):
        positions = torch.arange(length, device=self.key.device)
        slots = block_table[positions // self.block_size].long() * self.block_size + positions % self.block_size
        return self.key[slots], self.value[slots], positions


class MellumSDPAAttention(PagedAttention):
    """ATOM metadata/cache adapter; query and key have already been rotated."""

    def __init__(self, *args, per_layer_sliding_window=None, **kwargs):
        super().__init__(*args, per_layer_sliding_window=per_layer_sliding_window, **kwargs)
        self.window = per_layer_sliding_window

    def forward(self, query, key, value, positions=None, **kwargs):
        context = get_forward_context()
        if context.context.is_dummy_run:
            return torch.zeros_like(query)
        q = query.reshape(-1, self.num_heads, self.head_dim)
        k = key.reshape(-1, self.num_kv_heads, self.head_dim)
        v = value.reshape_as(k)
        meta = context.attn_metadata
        cache = MellumKVCacheView(
            context.kv_cache_data[f"layer_{self.layer_num}"], self.num_kv_heads, self.head_dim
        )
        cache.write(meta.slot_mapping[: len(k)].long(), k, v)
        cu = meta.cu_seqlens_q.tolist()
        lengths = meta.context_lens.tolist()
        output = torch.zeros_like(q)
        for index, length in enumerate(lengths):
            start, end = cu[index], cu[index + 1]
            if end == start or length == 0:
                continue
            if meta.block_tables is None:
                if length != end - start:
                    raise ValueError("Cached prefill requires a block table")
                keys, values = k[start:end], v[start:end]
                key_positions = positions[start:end]
            else:
                keys, values, key_positions = cache.read(meta.block_tables[index], length)
            output[start:end] = masked_mellum_sdpa(
                q[start:end], keys, values, positions[start:end], key_positions,
                self.window, self.scale,
            )
        return output.reshape_as(query)


def make_mellum_attention(config, atom_config, index):
    backend = os.environ.get("MELLUM_ATTN_BACKEND", "sdpa")
    if backend != "sdpa":
        raise NotImplementedError(f"Mellum attention backend {backend!r} is not implemented")
    return MellumSDPAAttention(
        config.num_attention_heads, config.head_dim, config.head_dim ** -0.5,
        config.num_key_value_heads, kv_cache_dtype=atom_config.kv_cache_dtype,
        layer_num=index, rotary_emb=None,
        per_layer_sliding_window=(
            config.sliding_window if config.layer_types[index] == "sliding_attention" else None
        ),
        prefix=f"model.layers.{index}.self_attn.attn",
    )
