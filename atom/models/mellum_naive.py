# SPDX-License-Identifier: Apache-2.0
# Torch math follows Transformers v5.10.1 modeling_mellum.py (Apache-2.0):
# https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/mellum/modeling_mellum.py
"""Independent, uncompiled single-GPU Mellum baseline inside the ATOM engine.

Plain torch experts and explicit-mask SDPA. No FusedMoE, native attention,
model compilation, quantization, prefix cache or speculative decoding.
"""

import torch
from torch import nn
from torch.nn import functional as F

from atom.models.mellum_naive_ops import NaiveRMSNorm, NaiveRoPE, NaiveSDPAAttention, rotate
from atom.models.utils import make_empty_intermediate_tensors_factory
from atom.utils.forward_context import get_forward_context


def validate_naive_config(config):
    if config.tensor_parallel_size != 1 or config.parallel_config.data_parallel_size != 1:
        raise ValueError('Mellum naive baseline supports a single inference GPU only')
    if config.enable_expert_parallel or config.enable_dp_attention:
        raise ValueError('Mellum naive baseline does not support parallel experts/attention')
    if not config.enforce_eager or config.compilation_config.level != 0:
        raise ValueError('Mellum naive baseline requires --enforce-eager --level 0')
    if config.speculative_config is not None:
        raise ValueError('Mellum naive baseline requires MTP/speculative decoding OFF')
    if config.kv_cache_dtype != 'bf16' or config.torch_dtype != torch.bfloat16:
        raise ValueError('Mellum naive baseline requires BF16 weights and KV')
    if config.kv_cache_block_size != 16:
        raise ValueError('Mellum naive baseline currently supports 16-token KV pages')
    if config.enable_prefix_caching or config.enable_tbo or config.enable_tbo_decode:
        raise ValueError('Mellum naive baseline requires prefix cache and TBO OFF')
    if config.online_quant_config or config.kv_transfer_config:
        raise ValueError('Mellum naive baseline does not support quantization or P/D transfer')
    hf = config.hf_config
    if hf.hidden_act != 'silu' or hf.num_attention_heads % hf.num_key_value_heads:
        raise ValueError('Unsupported Mellum activation or GQA dimensions')
    if len(hf.layer_types) != hf.num_hidden_layers or len(hf.mlp_layer_types) != hf.num_hidden_layers:
        raise ValueError('Mellum requires one attention/MLP type per layer')
    if not set(hf.layer_types) <= {'sliding_attention', 'full_attention'}:
        raise ValueError('Unsupported Mellum attention type')
    if not set(hf.mlp_layer_types) <= {'sparse', 'dense'}:
        raise ValueError('Unsupported Mellum MLP type')
    if not 0 < hf.num_experts_per_tok <= hf.num_experts:
        raise ValueError('Invalid expert top-k')


@torch.no_grad()
def load_expert(param, weight, name, shard_id, expert_id):
    if shard_id in ('gate', 'up'):
        width = param.shape[1] // 2
        offset = 0 if shard_id == 'gate' else width
        target = param[expert_id, offset:offset + width]
    elif shard_id == 'down':
        target = param[expert_id]
    else:
        raise ValueError(f'Unknown expert shard {shard_id}: {name}')
    if target.shape != weight.shape:
        raise ValueError(f'Expert weight shape mismatch: {name}: {weight.shape} != {target.shape}')
    target.copy_(weight)
    param.loaded_slices.add((shard_id, expert_id))


class TorchExperts(nn.Module):
    def __init__(self, config, require_full_load):
        super().__init__()
        self.num_experts = config.num_experts
        width, inner = config.hidden_size, config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(config.num_experts, 2 * inner, width))
        self.down_proj = nn.Parameter(torch.empty(config.num_experts, width, inner))
        self.require_full_load = require_full_load
        for param in (self.gate_up_proj, self.down_proj):
            param.weight_loader = load_expert
            param.loaded_slices = set()

    def forward(self, hidden, indices, weights):
        result = torch.zeros_like(hidden)
        expert_mask = F.one_hot(indices, num_classes=self.num_experts).permute(2, 1, 0)
        active = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
        for expert in active:
            expert = expert[0]
            top_position, tokens = torch.where(expert_mask[expert])
            gate, up = F.linear(hidden[tokens], self.gate_up_proj[expert]).chunk(2, dim=-1)
            states = F.linear(F.silu(gate) * up, self.down_proj[expert])
            states = states * weights[tokens, top_position, None]
            result.index_add_(0, tokens, states.to(result.dtype))
        return result

    def process_weights_after_loading(self):
        if not self.require_full_load:
            return
        expected = ({(part, e) for part in ('gate', 'up') for e in range(self.num_experts)},
                    {('down', e) for e in range(self.num_experts)})
        for param, slices in zip((self.gate_up_proj, self.down_proj), expected):
            if param.loaded_slices != slices:
                raise RuntimeError(f'Incomplete naive expert loading: {slices - param.loaded_slices}')


class TorchMoE(nn.Module):
    def __init__(self, config, require_full_load):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = TorchExperts(config, require_full_load)
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob

    def forward(self, hidden):
        logits = self.gate(hidden)
        weights, indices = logits.float().softmax(-1).topk(self.top_k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        return self.experts(hidden, indices, weights.to(logits.dtype))


class TorchMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TorchAttention(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.heads, self.kv_heads, self.width = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.width, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.width, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.width, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.heads * self.width, config.hidden_size, bias=config.attention_bias)
        self.q_norm = NaiveRMSNorm(self.width, config.rms_norm_eps)
        self.k_norm = NaiveRMSNorm(self.width, config.rms_norm_eps)
        self.kind = config.layer_types[index]
        self.attn = NaiveSDPAAttention(config, index)

    def forward(self, hidden, positions, embeddings):
        query = self.q_norm(self.q_proj(hidden).view(-1, self.heads, self.width))
        key = self.k_norm(self.k_proj(hidden).view(-1, self.kv_heads, self.width))
        value = self.v_proj(hidden).view(-1, self.kv_heads, self.width)
        query, key = rotate(query, key, *embeddings[self.kind])
        output = self.attn(query, key, value, positions)
        return self.o_proj(output.reshape(len(hidden), -1).contiguous())


class TorchDecoder(nn.Module):
    def __init__(self, config, index, require_full_load):
        super().__init__()
        self.self_attn = TorchAttention(config, index)
        self.input_layernorm = NaiveRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NaiveRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = (TorchMoE(config, require_full_load) if config.mlp_layer_types[index] == 'sparse'
                    else TorchMLP(config))

    def forward(self, hidden, positions, embeddings):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), positions, embeddings)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TorchBackbone(nn.Module):
    def __init__(self, config, require_full_load):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(TorchDecoder(config, i, require_full_load)
                                    for i in range(config.num_hidden_layers))
        self.norm = NaiveRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = NaiveRoPE(config)

    def forward(self, input_ids, positions, inputs_embeds=None):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        embeddings = self.rotary_emb(hidden, positions)
        for layer in self.layers:
            hidden = layer(hidden, positions, embeddings)
        return self.norm(hidden)


class MellumNaiveForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, atom_config, prefix=''):
        super().__init__()
        validate_naive_config(atom_config)
        self.atom_config = atom_config
        self.config = atom_config.hf_config
        self.model = TorchBackbone(self.config, not atom_config.load_dummy)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ['hidden_states', 'residual'], self.config.hidden_size)

    def get_expert_mapping(self):
        return [(f'experts.{target}', f'experts.{eid}.{source}.weight', eid, shard)
                for eid in range(self.config.num_experts)
                for source, target, shard in [('gate_proj', 'gate_up_proj', 'gate'),
                    ('up_proj', 'gate_up_proj', 'up'), ('down_proj', 'down_proj', 'down')]]

    def get_input_embeddings(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden):
        context = get_forward_context()
        if context.context.is_prefill and not context.context.is_draft:
            hidden = hidden[context.attn_metadata.cu_seqlens_q[1:] - 1].contiguous()
        return self.lm_head(hidden)
