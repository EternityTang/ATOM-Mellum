# SPDX-License-Identifier: MIT
"""Mellum-local correctness path on ROCm, retaining the Qwen backbone/loader.

The CK API choice follows vLLM's AITER provider. Separate BF16 residual adds,
RoPE and external HF routing follow its Transformers-backend organization.
SDPA is a local correctness fallback, not a port of vLLM's attention kernels.
All replacements retain the original checkpoint Parameters and names.
"""

import torch
from torch import nn
from torch.nn import functional as F

from atom.model_ops.layernorm import RMSNorm
from atom.models.mellum_naive_ops import NaiveRoPE, NaiveSDPAAttention, rotate
from atom.models.mellum_torch_experts import (
    MellumTorchExpertsForCausalLM,
    TorchExpertMethod,
    torch_expert_forward,
)
from atom.utils import envs


PROFILES = {
    "ck_native": dict(norm="ck", qkv="packed", attention="native", routing="hf"),
    "torch_native": dict(norm="torch", qkv="packed", attention="native", routing="hf"),
    "torch_sdpa_packed": dict(norm="torch", qkv="packed", attention="sdpa", routing="hf"),
    "correctness": dict(norm="torch", qkv="split", attention="sdpa", routing="hf"),
}


class MellumPrecisionNorm(nn.Module):
    def __init__(self, original, settings):
        super().__init__()
        self.weight = original.weight
        self.eps = original.eps
        self.settings = settings

    def forward(self, x, residual=None, x_scale=None):
        if x_scale is not None:
            raise ValueError("Mellum precision norm does not accept quantized inputs")
        # Materialize the BF16 residual boundary before either norm provider.
        values = x if residual is None else x + residual
        if self.settings["norm"] == "ck":
            from aiter import rms_norm

            shape = values.shape
            output = rms_norm(
                values.reshape(-1, shape[-1]).contiguous(), self.weight, self.eps
            ).view(shape)
        else:
            fp32 = values.float()
            normalized = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + self.eps)
            # Same standalone norm convention as vLLM IR native / HF Mellum.
            output = self.weight * normalized.to(values.dtype)
        return output if residual is None else (output, values)


class MellumPrecisionAttention(nn.Module):
    def __init__(self, original, config, index, rope, settings):
        super().__init__()
        for name in ("qkv_proj", "o_proj", "q_norm", "k_norm", "attn"):
            setattr(self, name, getattr(original, name))
        self.num_heads = original.num_heads
        self.num_kv_heads = original.num_kv_heads
        self.head_dim = original.head_dim
        self.q_size, self.kv_size = original.q_size, original.kv_size
        self.layer_num = index
        self.kind = config.layer_types[index]
        self.window = config.sliding_window if self.kind == "sliding_attention" else None
        self.scale = self.head_dim ** -0.5
        self.rope = rope
        self.settings = settings
        # RoPE is applied once, before cache insertion, for BOTH attention paths.
        self.attn.impl.rotary_emb = None
        if self.attn.impl.q_norm is not None or self.attn.impl.k_norm is not None:
            raise ValueError("Disable fused QK norm/RoPE/cache for Mellum precision")

    def forward(self, positions, hidden_states, **model_kwargs):
        weight, bias = self.qkv_proj.weight, self.qkv_proj.bias
        if self.settings["qkv"] == "split":
            sizes = (self.q_size, self.kv_size, self.kv_size)
            offset = 0
            outputs = []
            for size in sizes:
                outputs.append(F.linear(hidden_states, weight[offset:offset + size],
                    None if bias is None else bias[offset:offset + size]))
                offset += size
            query, key, value = outputs
        else:
            query, key, value = F.linear(hidden_states, weight, bias).split(
                (self.q_size, self.kv_size, self.kv_size), dim=-1)
        query = self.q_norm(query.view(-1, self.num_heads, self.head_dim))
        key = self.k_norm(key.view(-1, self.num_kv_heads, self.head_dim))
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query, key = rotate(query, key, *self.rope(hidden_states, positions)[self.kind])
        if self.settings["attention"] == "sdpa":
            # Reuse the validated BF16 page read/write and causal/SWA boundary code.
            output = NaiveSDPAAttention.forward(self, query, key, value, positions)
        else:
            output = self.attn(query=query.flatten(1), key=key.flatten(1),
                value=value.flatten(1), positions=positions, **model_kwargs)
        output = output.reshape(len(hidden_states), -1).contiguous()
        return F.linear(output, self.o_proj.weight, self.o_proj.bias)


class MellumPrecisionExpertMethod(TorchExpertMethod):
    def __init__(self, moe, settings):
        super().__init__(moe)
        self.settings = settings

    def apply(self, layer, x, router_logits, top_k, renormalize, **options):
        if options.get("expert_map") is not None or options.get("apply_router_weight_on_input", False):
            raise ValueError("Mellum precision requires local, output-weighted experts")
        if options.get("use_grouped_topk", False) or options.get("custom_routing_function") is not None:
            raise ValueError("Mellum precision requires ordinary softmax top-k routing")
        weights, indices = router_logits.float().softmax(-1).topk(top_k, dim=-1)
        if renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(router_logits.dtype)
        return torch_expert_forward(x, layer.w13_weight, layer.w2_weight, indices, weights)


class MellumRocmPrecisionForCausalLM(MellumTorchExpertsForCausalLM):
    def __init__(self, atom_config, prefix=""):
        if envs.ATOM_MELLUM_RNE_NORM:
            raise ValueError("Mellum ROCm precision owns norm selection; disable the RNE override")
        if (atom_config.kv_cache_dtype != "bf16" or atom_config.torch_dtype != torch.bfloat16
                or atom_config.kv_cache_block_size != 16 or atom_config.enable_prefix_caching):
            raise ValueError("Mellum precision requires BF16 weights/KV, block16, prefix cache off")
        super().__init__(atom_config, prefix)
        before = {name: id(value) for name, value in self.named_parameters()}
        self.precision_settings = {}
        self.configure_precision_profile(envs.ATOM_MELLUM_ROCM_PRECISION_PROFILE)
        count = 0
        for module in list(self.modules()):
            for name, child in list(module.named_children()):
                if type(child) is RMSNorm:
                    if child.use_fused_quant or child.x_pad_to_multiple or child.tp_size != 1:
                        raise ValueError("Unsupported Mellum precision norm configuration")
                    setattr(module, name, MellumPrecisionNorm(child, self.precision_settings))
                    count += 1
        self.precision_norm_count = count
        rope = NaiveRoPE(self.config)
        for index, layer in enumerate(self.model.layers):
            layer.self_attn = MellumPrecisionAttention(
                layer.self_attn, self.config, index, rope, self.precision_settings)
            experts = layer.mlp.experts
            experts.quant_method = MellumPrecisionExpertMethod(
                experts.moe_config, self.precision_settings)
        after = {name: id(value) for name, value in self.named_parameters()}
        if before != after or count != 4 * self.config.num_hidden_layers + 1:
            raise RuntimeError("Mellum precision must preserve all checkpoint Parameter identities")
        self.precision_parameter_identity_preserved = True

    def configure_precision_profile(self, name):
        """Select a profile before inference; callers must not change it in-flight."""
        if name not in PROFILES:
            raise ValueError(f"Unknown Mellum precision profile: {name}")
        self.precision_settings.clear()
        self.precision_settings.update(PROFILES[name])
        self.precision_profile = name
