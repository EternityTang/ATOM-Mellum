"""Single-GPU Mellum adapter with replaceable, unfused Step 0 modules.

Model semantics: Transformers 5.10.1 models/mellum (Apache-2.0).
Integration reference: vLLM v0.23.0 models/mellum.py (Apache-2.0).
ATOM owns scheduling, cache pages, request metadata and sampling.
"""

import copy
import os

import torch
from torch import nn
from transformers import MellumConfig
from transformers.models.mellum.modeling_mellum import (
    MellumExperts as HFExperts,
    MellumMLP as HFDenseMLP,
    MellumTopKRouter as HFRouter,
    apply_rotary_pos_emb,
)

from atom.models.mellum_ops import (
    MellumRMSNorm,
    MellumRoPE,
    make_mellum_attention,
)
from atom.models.utils import make_empty_intermediate_tensors_factory
from atom.utils.forward_context import get_forward_context


def validate_mellum_config(config):
    if not isinstance(config, MellumConfig):
        raise TypeError("Mellum requires native Transformers MellumConfig")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if config.head_dim <= 0:
        raise ValueError("Mellum head_dim must be explicitly positive")
    for name, allowed in (
        ("layer_types", {"full_attention", "sliding_attention"}),
        ("mlp_layer_types", {"sparse", "dense"}),
    ):
        values = getattr(config, name)
        if len(values) != config.num_hidden_layers or not set(values) <= allowed:
            raise ValueError(f"Invalid per-layer {name}: {values}")
    if not 0 < config.num_experts_per_tok <= config.num_experts:
        raise ValueError("Invalid expert top-k")
    for kind in set(config.layer_types):
        if kind not in config.rope_parameters:
            raise ValueError(f"Missing RoPE parameters for {kind}")
    if "sliding_attention" in config.layer_types and config.sliding_window <= 0:
        raise ValueError("Sliding attention requires a positive window")


@torch.no_grad()
def load_mellum_expert_weight(param, weight, name, shard_id, expert_id):
    """ATOM's existing expert-loader callback; no generic loader modification."""
    if shard_id in ("gate", "up"):
        width = param.shape[1] // 2
        offset = 0 if shard_id == "gate" else width
        target = param[expert_id, offset : offset + width]
    elif shard_id == "down":
        target = param[expert_id]
    else:
        raise ValueError(f"Unknown Mellum expert shard {shard_id}: {name}")
    if target.shape != weight.shape:
        raise ValueError(f"Mellum weight shape mismatch: {name}: {weight.shape} != {target.shape}")
    target.copy_(weight)
    param.mellum_loaded_slices.add((shard_id, expert_id))


class MellumMoE(nn.Module):
    """Stable [tokens, hidden] interface; torch expert loop is the baseline."""

    def __init__(self, config, require_full_load=True):
        super().__init__()
        backend = os.environ.get("MELLUM_MOE_BACKEND", "torch")
        if backend != "torch":
            raise NotImplementedError(f"Mellum MoE backend {backend!r} is not implemented")
        if config._experts_implementation != "eager":
            raise ValueError("Step 0 must explicitly select eager HF experts")
        self.backend = backend
        self.gate = HFRouter(config)
        self.experts = HFExperts(config)
        self.num_experts = config.num_experts
        self.require_full_load = require_full_load
        for param in (self.experts.gate_up_proj, self.experts.down_proj):
            param.weight_loader = load_mellum_expert_weight
            param.mellum_loaded_slices = set()

    def forward(self, hidden_states):
        _, weights, indices = self.gate(hidden_states)
        return self.experts(hidden_states, indices, weights)

    def process_weights_after_loading(self):
        if not self.require_full_load:
            return
        expected_gate_up = {(part, e) for part in ("gate", "up") for e in range(self.num_experts)}
        expected_down = {("down", e) for e in range(self.num_experts)}
        for param, expected in (
            (self.experts.gate_up_proj, expected_gate_up),
            (self.experts.down_proj, expected_down),
        ):
            if param.mellum_loaded_slices != expected:
                raise RuntimeError(f"Incomplete Mellum expert loading: {expected - param.mellum_loaded_slices}")


class MellumAttention(nn.Module):
    def __init__(self, config, atom_config, index):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        h, d = config.hidden_size, self.head_dim
        self.q_proj = nn.Linear(h, self.num_heads * d, bias=config.attention_bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * d, bias=config.attention_bias)
        self.v_proj = nn.Linear(h, self.num_kv_heads * d, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * d, h, bias=config.attention_bias)
        self.q_norm = MellumRMSNorm(d, config.rms_norm_eps)
        self.k_norm = MellumRMSNorm(d, config.rms_norm_eps)
        self.layer_type = config.layer_types[index]
        self.attn = make_mellum_attention(config, atom_config, index)

    def forward(self, hidden_states, positions, position_embeddings):
        count = hidden_states.shape[0]
        q = self.q_norm(self.q_proj(hidden_states).view(1, count, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(hidden_states).view(1, count, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden_states).view(count, self.num_kv_heads * self.head_dim)
        q, k = apply_rotary_pos_emb(
            q.transpose(1, 2), k.transpose(1, 2), *position_embeddings
        )
        output = self.attn(
            q.transpose(1, 2).reshape(count, -1),
            k.transpose(1, 2).reshape(count, -1),
            v,
            positions=positions,
        )
        return self.o_proj(output)


class MellumDecoderLayer(nn.Module):
    def __init__(self, config, atom_config, index):
        super().__init__()
        self.self_attn = MellumAttention(config, atom_config, index)
        self.input_layernorm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[index] == "sparse":
            self.mlp = MellumMoE(config, require_full_load=not getattr(atom_config, "load_dummy", False))
        else:
            self.mlp = HFDenseMLP(config)

    def forward(self, hidden_states, positions, position_embeddings):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), positions, position_embeddings
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class MellumModel(nn.Module):
    def __init__(self, config, atom_config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            MellumDecoderLayer(config, atom_config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = MellumRoPE(config)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None):
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, positions)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, positions, position_embeddings[layer.self_attn.layer_type]
            )
        return self.norm(hidden_states)


class MellumForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, atom_config, prefix=""):
        super().__init__()
        if atom_config.tensor_parallel_size != 1 or getattr(atom_config, "pipeline_parallel_size", 1) != 1:
            raise ValueError("Mellum Step 0 supports single-GPU inference only")
        if atom_config.kv_cache_dtype != "bf16":
            raise ValueError("Mellum Step 0 requires unquantized bf16 KV")
        self.config = copy.deepcopy(atom_config.hf_config)
        validate_mellum_config(self.config)
        self.config._experts_implementation = "eager"
        self.config._attn_implementation = "sdpa"
        self.model = MellumModel(self.config, atom_config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def get_expert_mapping(self):
        # Include .weight in source fragments: HF's packed targets have no .weight suffix.
        mapping = []
        for eid in range(self.config.num_experts):
            for source, target, shard in (
                ("gate_proj", "gate_up_proj", "gate"),
                ("up_proj", "gate_up_proj", "up"),
                ("down_proj", "down_proj", "down"),
            ):
                mapping.append((f"experts.{target}", f"experts.{eid}.{source}.weight", eid, shard))
        return mapping

    def get_input_embeddings(self, input_ids):
        return self.model.get_input_embeddings(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        context = get_forward_context()
        if context.context.is_prefill and not context.context.is_draft:
            hidden_states = hidden_states[context.attn_metadata.cu_seqlens_q[1:] - 1].contiguous()
        return self.lm_head(hidden_states)
