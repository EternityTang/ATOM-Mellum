"""Mellum Step 0 for ATOM: single GPU, plain torch, no fused MoE/compile/MTP.

Architecture and arithmetic follow HF Mellum (source commit recorded in
/workspace/reference/SOURCE.txt). Engine scheduling and page allocation are ATOM.
"""
import torch
import os
from pathlib import Path
from torch import nn
from torch.nn import functional as F

from atom.models.mellum_ops import MellumRoPE, MellumRMSNorm, MellumSDPAAttention
from atom.models.utils import make_empty_intermediate_tensors_factory
from atom.utils.forward_context import get_forward_context


class MellumExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, d = config.hidden_size, config.moe_intermediate_size
        # Named checkpoint parameters share a packed allocation. This matches
        # the HF expert GEMM dimensions without retaining a duplicate weight copy.
        self.register_buffer('_gate_up', torch.empty(2 * d, h), persistent=False)
        self.gate_proj = nn.Linear(h, d, bias=False)
        self.up_proj = nn.Linear(h, d, bias=False)
        self.gate_proj.weight = nn.Parameter(self._gate_up[:d])
        self.up_proj.weight = nn.Parameter(self._gate_up[d:])
        self.down_proj = nn.Linear(d, h, bias=False)

    def forward(self, x):
        gate, up = F.linear(x, self._gate_up).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MellumMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.normalize = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([MellumExpert(config) for _ in range(config.num_experts)])

    def forward(self, x):
        weights = F.softmax(self.gate(x), dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(weights, self.top_k, dim=-1)
        if self.normalize:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(x.dtype)
        output = torch.zeros_like(x)
        for eid, expert in enumerate(self.experts):
            # HF traverses its [expert, top-k-slot, token] mask in slot-major
            # order. Preserve that row order through GEMM to avoid bf16 drift.
            slots, rows = torch.where(indices.transpose(0, 1) == eid)
            if rows.numel():
                output.index_add_(0, rows, expert(x[rows]) * weights[rows, slots, None])
        return output


class MellumAttention(nn.Module):
    def __init__(self, config, atom_config, index):
        super().__init__()
        self.qheads, self.kheads, self.dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.qheads * self.dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kheads * self.dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kheads * self.dim, bias=False)
        self.o_proj = nn.Linear(self.qheads * self.dim, config.hidden_size, bias=False)
        self.q_norm = MellumRMSNorm(self.dim, config.rms_norm_eps)
        self.k_norm = MellumRMSNorm(self.dim, config.rms_norm_eps)
        kind = config.layer_types[index]
        params = config.rope_parameters[kind]
        rotary = MellumRoPE(self.dim, config.max_position_embeddings, params['rope_theta'], params)
        self.attn = MellumSDPAAttention(self.qheads, self.dim, self.dim ** -0.5, self.kheads,
            kv_cache_dtype=atom_config.kv_cache_dtype, layer_num=index, rotary_emb=rotary,
            per_layer_sliding_window=config.sliding_window if kind == 'sliding_attention' else None,
            prefix=f'model.layers.{index}.self_attn.attn')

    def forward(self, x, positions):
        q = self.q_norm(self.q_proj(x).view(-1, self.qheads, self.dim))
        k = self.k_norm(self.k_proj(x).view(-1, self.kheads, self.dim))
        v = self.v_proj(x).view(-1, self.kheads, self.dim)
        output = self.attn(q.reshape(len(x), -1), k.reshape(len(x), -1), v.reshape(len(x), -1), positions=positions)
        return self.o_proj(output)


class MellumDecoderLayer(nn.Module):
    def __init__(self, config, atom_config, index):
        super().__init__()
        self.self_attn = MellumAttention(config, atom_config, index)
        self.input_layernorm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MellumMoE(config)

    def forward(self, x, positions):
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class MellumModel(nn.Module):
    def __init__(self, config, atom_config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MellumDecoderLayer(config, atom_config, i) for i in range(config.num_hidden_layers)])
        self.norm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(['hidden_states', 'residual'], config.hidden_size)

    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            x = layer(x, positions)
        return self.norm(x)


class MellumForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, atom_config, prefix=''):
        super().__init__()
        assert atom_config.tensor_parallel_size == 1, 'Step 0 is single GPU'
        self.config = atom_config.hf_config
        self.model = MellumModel(self.config, atom_config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        dump = os.getenv('MELLUM_DUMP_DIR')
        if dump:
            wanted = ['input_layernorm', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_norm', 'self_attn.k_norm', 'self_attn', 'self_attn.o_proj', 'post_attention_layernorm', 'mlp.gate', 'mlp', '']
            names = {'model.layers.0' + ('.' + x if x else '') for x in wanted}
            for name, module in self.named_modules():
                if name in names:
                    module.register_forward_hook(self._dump_hook(dump, name))

    @staticmethod
    def _dump_hook(directory, name):
        done = False
        def hook(module, args, output):
            nonlocal done
            context = get_forward_context().context
            if done or context.is_dummy_run or not context.is_prefill:
                return
            done = True
            Path(directory).mkdir(parents=True, exist_ok=True)
            packet = {'input': args[0].detach().cpu(), 'output': output.detach().cpu()}
            weight = getattr(module, 'weight', None)
            if weight is not None:
                packet['weight'] = weight.detach().cpu()
            torch.save(packet, Path(directory) / (name + '.pt'))
        return hook

    def get_input_embeddings(self, input_ids):
        return self.model.get_input_embeddings(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        context = get_forward_context()
        if context.context.is_prefill and not context.context.is_draft:
            hidden_states = hidden_states[context.attn_metadata.cu_seqlens_q[1:] - 1].contiguous()
        return self.lm_head(hidden_states)
