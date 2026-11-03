# Apache-2.0. Backported inference components from upstream Mellum (see SOURCE.txt).
# No installed package changes. Cache/mask driver and checkpoint packing are local adapters.
from __future__ import annotations
from collections.abc import Callable
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from safetensors import safe_open
GradientCheckpointingLayer=nn.Module
class MellumRotaryEmbedding(nn.Module):

    def __init__(self, config: MellumConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = sorted(set(config.layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue
            self.rope_type[layer_type] = rope_params['rope_type']
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != 'default':
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            setattr(self, f'{layer_type}_inv_freq', nn.Buffer(curr_inv_freq, persistent=False))
            setattr(self, f'{layer_type}_original_inv_freq', nn.Buffer(curr_inv_freq.clone(), persistent=False))
            setattr(self, f'{layer_type}_attention_scaling', curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(config: MellumConfig, device=None, layer_type: str | None=None, **kwargs) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters[layer_type]['rope_theta']
        partial_rotary_factor = config.rope_parameters[layer_type].get('partial_rotary_factor', 1.0)
        head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        return (inv_freq.to(device), attention_factor)

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type):
        inv_freq = getattr(self, f'{layer_type}_inv_freq')
        attention_scaling = getattr(self, f'{layer_type}_attention_scaling')
        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

class MellumRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps: float=1e-06) -> None:
        """
        MellumRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None, scaling: float, dropout: float=0.0, **kwargs: Unpack[TransformersKwargs]):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return (attn_output, attn_weights)

class MellumAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = MellumRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MellumRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == 'sliding_attention' else None

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor | None, past_key_values: Cache | None=None, **kwargs: Unpack[FlashAttentionKwargs]) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)

class MellumExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

class MellumTopKRouter(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return (router_logits, router_scores, router_indices)

class MellumSparseMoeBlock(nn.Module):

    def __init__(self, config: MellumConfig):
        super().__init__()
        self.experts = MellumExperts(config)
        self.gate = MellumTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

class MellumMLP(nn.Module):

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MellumDecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MellumAttention(config, layer_idx)
        if config.mlp_layer_types[layer_idx] == 'sparse':
            self.mlp = MellumSparseMoeBlock(config)
        else:
            self.mlp = MellumMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = MellumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MellumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_values: Cache | None=None, use_cache: bool | None=False, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class ReferenceMellum(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.config=PretrainedConfig.from_dict(json.loads((Path(model_path)/'config.json').read_text()))
        self.config._attn_implementation='sdpa'
        c=self.config
        with torch.device('meta'):
            self.embed_tokens=nn.Embedding(c.vocab_size,c.hidden_size)
            self.layers=nn.ModuleList([MellumDecoderLayer(c,i) for i in range(c.num_hidden_layers)])
            self.norm=MellumRMSNorm(c.hidden_size,c.rms_norm_eps)
            self.lm_head=nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        self.to_empty(device='cuda')
        self.to(dtype=torch.bfloat16)
        self.rotary_emb=MellumRotaryEmbedding(c).to('cuda')
        params=dict(self.named_parameters()); seen=set(); count=0
        for shard in sorted(Path(model_path).glob('*.safetensors')):
            with safe_open(shard,framework='pt',device='cpu') as f:
                for name in f.keys():
                    target=name.removeprefix('model.')
                    t=f.get_tensor(name)
                    if '.mlp.experts.' in target:
                        prefix,rest=target.split('.mlp.experts.')
                        eid,part,_=rest.split('.')
                        if part=='down_proj':
                            key=prefix+'.mlp.experts.down_proj'; params[key].data[int(eid)].copy_(t)
                        else:
                            key=prefix+'.mlp.experts.gate_up_proj'; offset=0 if part=='gate_proj' else c.moe_intermediate_size
                            params[key].data[int(eid),offset:offset+c.moe_intermediate_size].copy_(t)
                    else:
                        key=target; params[key].data.copy_(t)
                    seen.add(key); count+=1
        assert set(params)==seen,(set(params)-seen,seen-set(params))
        self.eval()
        print('REFERENCE loaded all',count,'checkpoint tensors; parameters',len(seen),flush=True)
    @torch.inference_mode()
    def step(self,ids,cache=None):
        if cache is None: cache=DynamicCache()
        start=cache.get_seq_length()
        pos=torch.arange(start,start+ids.shape[1],device=ids.device)[None]
        x=self.embed_tokens(ids)
        ro={typ:self.rotary_emb(x,pos,typ) for typ in set(self.config.layer_types)}
        keypos=torch.arange(start+ids.shape[1],device=ids.device)
        masks={}
        for typ in ro:
            valid=keypos[None,:]<=pos[0,:,None]
            if typ=='sliding_attention': valid &= keypos[None,:]>pos[0,:,None]-self.config.sliding_window
            masks[typ]=torch.where(valid,0.0,float('-inf')).to(x.dtype)[None,None]
        for i,layer in enumerate(self.layers):
            typ=self.config.layer_types[i]
            x=layer(x,attention_mask=masks[typ],position_embeddings=ro[typ],past_key_values=cache)
        return self.lm_head(self.norm(x[:,-1:])),cache
    @torch.inference_mode()
    def greedy(self,ids,n):
        cache=None; output=[]
        for i in range(n):
            logits,cache=self.step(ids,cache)
            ids=logits[:,-1].argmax(-1)[:,None]
            output.append(ids.item())
        return output
