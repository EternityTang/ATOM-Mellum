# Step 0: explicit PyTorch RoPE and masked SDPA. No fused attention or KV truncation.
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from atom.model_ops.paged_attention import PagedAttention
from atom.utils.forward_context import get_forward_context

class MellumRoPE(nn.Module):
    def __init__(self, head_size, max_position, base, rope_scaling):
        super().__init__()
        p = dict(rope_scaling)
        if p['rope_type'] == 'default':
            inv = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float32, device='cpu') / head_size))
            scale = 1.0
        else:
            cfg = PretrainedConfig(head_dim=head_size, hidden_size=2304, num_attention_heads=32,
                                   max_position_embeddings=max_position, rope_parameters=p)
            inv, scale = ROPE_INIT_FUNCTIONS[p['rope_type']](cfg, device='cpu')
        self.register_buffer('inv_freq', inv.to(device=torch.get_default_device()), persistent=False)
        self.scale = scale
    def forward(self, positions, q, k):
        f = positions.float()[:, None] * self.inv_freq.float()[None, :]
        angle = torch.cat((f, f), -1)
        c = (angle.cos() * self.scale).to(q.dtype)[:, None, :]
        s = (angle.sin() * self.scale).to(q.dtype)[:, None, :]
        def rotate(x):
            a,b=x.chunk(2,-1)
            return x*c + torch.cat((-b,a),-1)*s
        return rotate(q), rotate(k)

def get_rope(head_size, rotary_dim, max_position, base, rope_scaling):
    return MellumRoPE(head_size, max_position, base, rope_scaling)

class MellumSDPAAttention(PagedAttention):
    def __init__(self, *args, per_layer_sliding_window=None, **kwargs):
        super().__init__(*args, per_layer_sliding_window=per_layer_sliding_window, **kwargs)
        self.window = per_layer_sliding_window
    def forward(self, query, key, value, positions=None, **kwargs):
        ctx = get_forward_context()
        if ctx.context.is_dummy_run:
            return torch.zeros_like(query)
        q=query.reshape(-1,self.num_heads,self.head_dim)
        k=key.reshape(-1,self.num_kv_heads,self.head_dim)
        v=value.reshape_as(k)
        q,k=self.impl.rotary_emb(positions,q,k)
        cache=ctx.kv_cache_data[f'layer_{self.layer_num}']
        # Step 0 owns this logical layout for both writes and reads.
        # The engine only allocates pages; no other kernel reads their contents.
        kc=cache.k_cache.view(-1,self.num_kv_heads,self.head_dim)
        vc=cache.v_cache.view_as(kc)
        meta=ctx.attn_metadata
        slots=meta.slot_mapping[:len(k)].long()
        valid=slots>=0
        kc[slots[valid]]=k[valid]
        vc[slots[valid]]=v[valid]
        cu=meta.cu_seqlens_q.tolist()
        lengths=meta.context_lens.tolist()
        block_size=cache.k_cache.shape[-2]
        output=torch.zeros_like(q)
        for i,length in enumerate(lengths):
            start,end=cu[i],cu[i+1]
            if end==start or length==0: continue
            pos=torch.arange(length,device=q.device)
            if meta.block_tables is None:
                assert length == end-start
                keys=k[start:end].transpose(0,1)[None]
                vals=v[start:end].transpose(0,1)[None]
            else:
                ids=meta.block_tables[i,pos//block_size].long()*block_size+pos%block_size
                keys=kc[ids].transpose(0,1)[None]
                vals=vc[ids].transpose(0,1)[None]
            query_pos=positions[start:end]
            mask=pos[None,:]<=query_pos[:,None]
            if self.window is not None:
                mask &= pos[None,:] > query_pos[:,None]-self.window
            # Explicit mask, unquantized cache, no MTP, eager only.
            # HF's masked SDPA path explicitly repeats KV heads; enabling native
            # GQA here selects a different backend and changes bf16 numerics.
            groups = self.num_heads // self.num_kv_heads
            keys = keys[:, :, None].expand(1, self.num_kv_heads, groups, length, self.head_dim).reshape(1, self.num_heads, length, self.head_dim)
            vals = vals[:, :, None].expand(1, self.num_kv_heads, groups, length, self.head_dim).reshape(1, self.num_heads, length, self.head_dim)
            additive = torch.where(mask, 0.0, float('-inf')).to(q.dtype)[None,None]
            out=F.scaled_dot_product_attention(q[start:end].transpose(0,1)[None],keys,vals,
                    attn_mask=additive,dropout_p=0.0,scale=self.scale,is_causal=False)
            output[start:end]=out[0].transpose(0,1)
        return output.reshape_as(query)


class MellumRMSNorm(nn.Module):
    """Match HF Mellum's bf16 cast-before-weight order; single-GPU Step 0."""
    def __init__(self, hidden_size, eps=1e-6, fused_allreduce=False, **kwargs):
        super().__init__()
        if fused_allreduce:
            raise ValueError('Step 0 MellumRMSNorm supports single GPU only')
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x, residual=None):
        value = x if residual is None else x + residual
        norm = value.float()
        norm = norm * torch.rsqrt(norm.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        out = self.weight * norm.to(value.dtype)
        return out if residual is None else (out, value)
