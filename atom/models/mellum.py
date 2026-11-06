# SPDX-License-Identifier: MIT
# Integration pattern: vLLM v0.23.0 model_executor/models/mellum.py (Apache-2.0).
"""Mellum integration reusing ATOM's Qwen3-MoE and native SWA attention.

Only constructors differ: per-layer attention/RoPE and MLP selection. Forward,
MoE kernels, norms, cache management and checkpoint loaders remain ATOM's.
SWA is connected through the same per_layer_sliding_window interface used by
GPT-OSS and MiMo; their model-specific sinks, biases and V scaling are not used.
"""

import torch
from torch import nn
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope

from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import QKVParallelLinear, RowParallelLinear
from atom.models.qwen3_moe import (
    ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION,
    ENABLE_ALLREDUCE_RMSNORM_FUSION,
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from atom.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile


class MellumAttention(Qwen3MoeAttention):
    """Qwen3-MoE forward with native per-layer SWA and AITER RoPE."""

    def __init__(self, atom_config, layer_num, prefix):
        nn.Module.__init__(self)
        config = atom_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size:
            raise ValueError("Query heads must be divisible by TP size")
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated KV heads")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.layer_num = layer_num
        self.kv_cache_dtype = atom_config.kv_cache_dtype
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, self.total_num_heads,
            self.total_num_kv_heads, bias=config.attention_bias,
            quant_config=atom_config.quant_config, prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size, bias=False,
            quant_config=atom_config.quant_config,
            reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
            prefix=f"{prefix}.o_proj",
        )
        kind = config.layer_types[layer_num]
        rope = dict(config.rope_parameters[kind])
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope["rope_theta"], rope_scaling=rope,
        )
        if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:
            joint = torch.cat((self.rotary_emb.cos_cache, self.rotary_emb.sin_cache), dim=-1)
            self.rotary_emb.register_buffer(
                "cos_sin_cache", joint.view(joint.size(0), self.head_dim), persistent=False
            )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype, layer_num=layer_num,
            rotary_emb=self.rotary_emb,
            per_layer_sliding_window=(config.sliding_window if kind == "sliding_attention" else None),
            prefix=f"{prefix}.attn",
            q_norm=self.q_norm if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION else None,
            k_norm=self.k_norm if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION else None,
        )


class MellumDecoderLayer(Qwen3MoeDecoderLayer):
    """Reuse the Qwen residual/norm/MoE forward; select Mellum layer types."""

    def __init__(self, atom_config, layer_num=0, prefix=""):
        nn.Module.__init__(self)
        self.atom_config = atom_config
        config = atom_config.hf_config
        self.layer_idx = layer_num
        self.hidden_size = config.hidden_size
        self.self_attn = MellumAttention(atom_config, layer_num, f"{prefix}.self_attn")
        if config.mlp_layer_types[layer_num] == "sparse":
            self.mlp = Qwen3MoeSparseMoeBlock(
                config, quant_config=atom_config.quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3MoeMLP(
                config.hidden_size, config.intermediate_size, config.hidden_act,
                quant_config=atom_config.quant_config,
                reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION and layer_num > 0,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )


@support_torch_compile
class MellumModel(Qwen3MoeModel):
    """Construct Mellum layers while inheriting the complete Qwen backbone API."""

    def __init__(self, atom_config, prefix=""):
        nn.Module.__init__(self)
        self.config = atom_config.hf_config
        config = self.config
        if len(config.layer_types) != config.num_hidden_layers or len(config.mlp_layer_types) != config.num_hidden_layers:
            raise ValueError("Mellum requires one attention/MLP type per layer")
        if not set(config.layer_types) <= {"sliding_attention", "full_attention"}:
            raise ValueError("Unsupported Mellum attention layer type")
        if not set(config.mlp_layer_types) <= {"sparse", "dense"}:
            raise ValueError("Unsupported Mellum MLP layer type")
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if get_pp_group().is_first_rank else PPMissingLayer()
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: MellumDecoderLayer(atom_config, layer_num, prefix),
            prefix=f"{prefix}.layers", layer_num_offset=0,
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                    fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION)
            if get_pp_group().is_last_rank else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )


class MellumForCausalLM(Qwen3MoeForCausalLM):
    packed_modules_mapping = dict(Qwen3MoeForCausalLM.packed_modules_mapping)

    def __init__(self, atom_config, prefix=""):
        nn.Module.__init__(self)
        self.atom_config = atom_config
        self.config = atom_config.hf_config
        self.model = MellumModel(atom_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = (
            ParallelLMHead(self.config.vocab_size, self.config.hidden_size,
                           bias=False, prefix=maybe_prefix(prefix, "lm_head"))
            if get_pp_group().is_last_rank else PPMissingLayer()
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
