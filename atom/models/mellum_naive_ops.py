# SPDX-License-Identifier: Apache-2.0
# Numerical conventions follow Transformers v5.10.1 modeling_mellum.py
# (Hugging Face / JetBrains, Apache-2.0); ATOM cache integration is local.
"""Unfused torch operations for the single-GPU Mellum baseline."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from atom.utils.forward_context import get_forward_context


class NaiveRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class NaiveRoPE(nn.Module):
    """FP32 per-type frequencies; ordinary torch rotation with YaRN support."""

    def __init__(self, config):
        super().__init__()
        target = torch.get_default_device()
        self.kinds = sorted(set(config.layer_types))
        self.factors = {}
        for kind in self.kinds:
            params = config.rope_parameters[kind]
            with torch.device('cpu'):
                if params['rope_type'] == 'default':
                    inv = 1.0 / (params['rope_theta'] ** (
                        torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
                    factor = 1.0
                else:
                    inv, factor = ROPE_INIT_FUNCTIONS[params['rope_type']](
                        config, device=torch.device('cpu'), layer_type=kind)
            self.register_buffer(kind + '_inv_freq', inv.to(device=target, dtype=torch.float32), persistent=False)
            self.factors[kind] = factor

    def _apply(self, fn, recurse=True):
        # model.to(dtype=bf16) must not round the frequency buffers.
        original = {name: value.clone() for name, value in self.named_buffers()}
        result = super()._apply(fn, recurse=recurse)
        for name, value in original.items():
            self.register_buffer(name, value.to(device=getattr(self, name).device,
                                               dtype=torch.float32), persistent=False)
        return result

    def forward(self, x, positions):
        result = {}
        with torch.autocast(device_type=x.device.type, enabled=False):
            for kind in self.kinds:
                inv = getattr(self, kind + '_inv_freq').to(x.device)
                frequencies = (inv[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)
                angles = torch.cat((frequencies, frequencies), dim=-1)
                result[kind] = ((angles.cos() * self.factors[kind]).to(x.dtype)[0, :, None, :],
                                (angles.sin() * self.factors[kind]).to(x.dtype)[0, :, None, :])
        return result


def rotate(query, key, cos, sin):
    def half(x):
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    return query * cos + half(query) * sin, key * cos + half(key) * sin


def visibility(query_positions, key_positions, window):
    mask = key_positions[None, :] <= query_positions[:, None]
    if window is not None:
        mask &= key_positions[None, :] > query_positions[:, None] - window
    return mask


def masked_sdpa(query, key, value, query_positions, key_positions, window, scale):
    """Explicit additive causal/SWA mask, including cached query offsets."""
    groups = query.shape[1] // key.shape[1]

    def repeat(x):
        x = x.transpose(0, 1)[None]
        batch, heads, length, width = x.shape
        return x[:, :, None].expand(batch, heads, groups, length, width).reshape(
            batch, heads * groups, length, width)

    mask = torch.zeros((len(query_positions), len(key_positions)), dtype=query.dtype,
                       device=query.device)
    mask.masked_fill_(~visibility(query_positions, key_positions, window), float('-inf'))
    output = F.scaled_dot_product_attention(query.transpose(0, 1)[None], repeat(key),
        repeat(value), attn_mask=mask[None, None], dropout_p=0.0, is_causal=False, scale=scale)
    return output[0].transpose(0, 1)


def naive_sdpa(query, key, value, query_positions, key_positions, window, scale):
    """Match HF's mask-free SDPA calls when their visibility is equivalent.

    Cached multi-token queries need an offset causal mask: SDPA's causal flag
    alone uses the top-left diagonal. Keep the explicit path for those queries
    and for SWA at/above its boundary (HF skips only when KV length < window).
    """
    local_skip = window is None or len(key_positions) < window
    if local_skip and window is not None:
        local_skip = bool((query_positions[-1] - key_positions[0] < window).item())
    same_positions = len(query_positions) == len(key_positions) and torch.equal(
        query_positions, key_positions)
    causal = same_positions and bool(torch.all(key_positions[1:] > key_positions[:-1]).item())
    decode = len(query_positions) == 1 and bool(torch.all(key_positions <= query_positions[0]).item())
    if not (local_skip and (causal or decode)):
        return masked_sdpa(query, key, value, query_positions, key_positions, window, scale)
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1)[None], key.transpose(0, 1)[None], value.transpose(0, 1)[None],
        attn_mask=None, dropout_p=0.0, is_causal=len(query_positions) > 1,
        scale=scale, enable_gqa=True)
    return output[0].transpose(0, 1)


class TorchPagedKV:
    """Torch indexing into ATOM's native BF16 packed K / HDS V page layout."""

    def __init__(self, cache, heads, width):
        self.key, self.value = cache.k_cache, cache.v_cache
        if self.key.dtype != torch.bfloat16 or self.value.dtype != torch.bfloat16:
            raise ValueError('Naive baseline requires BF16 KV pages')
        if self.key.ndim != 5 or self.value.ndim != 4:
            raise ValueError('Unsupported ATOM KV layout for naive baseline')
        self.block_size = self.key.shape[3]
        self.pack = self.key.shape[4]
        self.heads, self.width = heads, width
        if self.key.shape[1:3] != (heads, width // self.pack):
            raise ValueError('Key page shape does not match model dimensions')
        if self.value.shape[1:] != (heads, width, self.block_size):
            raise ValueError('Value page shape does not match model dimensions')

    def write(self, slots, key, value):
        valid = slots >= 0
        slots = slots[valid].long()
        blocks, offsets = slots // self.block_size, slots % self.block_size
        self.key[blocks, :, :, offsets, :] = key[valid].reshape(
            -1, self.heads, self.width // self.pack, self.pack)
        self.value[blocks, :, :, offsets] = value[valid]

    def read(self, block_table, length):
        positions = torch.arange(length, device=self.key.device)
        blocks = block_table[positions // self.block_size].long()
        offsets = positions % self.block_size
        keys = self.key[blocks, :, :, offsets, :].reshape(length, self.heads, self.width)
        values = self.value[blocks, :, :, offsets]
        return keys, values, positions


class NaiveSDPAAttention(nn.Module):
    """Use existing ATOM allocation/metadata; no native attention kernel calls."""

    def __init__(self, config, index):
        super().__init__()
        # AiterAttentionMetadataBuilder recognizes this existing binding protocol.
        self.base_attention = None
        self.use_mla = False
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        self.layer_num = index
        self.window = config.sliding_window if config.layer_types[index] == 'sliding_attention' else None
        self.kv_cache_dtype = 'bf16'
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None
        self.max_model_len = 0

    def forward(self, query, key, value, positions):
        context = get_forward_context()
        if context.context.is_dummy_run:
            # Allocation has not happened yet during ATOM's memory profiling.
            return torch.zeros_like(query)
        meta = context.attn_metadata
        cache = TorchPagedKV(context.kv_cache_data[f'layer_{self.layer_num}'],
                             self.num_kv_heads, self.head_dim)
        cache.write(meta.slot_mapping[:len(key)], key, value)
        boundaries = meta.cu_seqlens_q.tolist()
        output = torch.zeros_like(query)
        for index, length in enumerate(meta.context_lens.tolist()):
            start, end = boundaries[index:index + 2]
            if start == end or length == 0:
                continue
            if meta.block_tables is None:
                if length != end - start:
                    raise ValueError('Cached prefill requires an ATOM block table')
                keys, values, key_positions = key[start:end], value[start:end], positions[start:end]
            else:
                keys, values, key_positions = cache.read(meta.block_tables[index], length)
            output[start:end] = naive_sdpa(query[start:end], keys, values,
                positions[start:end], key_positions, self.window, self.scale)
        return output
