"""Materialize identical BF16 K/V rows for torch SDPA from paged storage."""
import torch
import triton
import triton.language as tl
from atom.models.mellum_naive_ops import TorchPagedKV
from atom.utils.forward_context import get_forward_context
from torch.nn import functional as F

@triton.jit
def _gather(k, v, blocks, keys, values, L, H:tl.constexpr,
            D:tl.constexpr, P:tl.constexpr, PACK:tl.constexpr,
            K0, K1, K2, K3, K4, V0, V1, V2, V3):
    page = tl.program_id(0)
    row = tl.program_id(1)
    head = tl.program_id(2)
    # Physical page size matches the number of positions handled by this program.
    physical = tl.load(blocks + row * L + page * P).to(tl.int64)
    pos = tl.arange(0, P)[:, None]
    dim = tl.arange(0, D)[None, :]
    valid = page * P + pos < L
    ko = physical*K0 + head*K1 + (dim//PACK)*K2 + pos*K3 + (dim%PACK)*K4
    vo = physical*V0 + head*V1 + dim*V2 + pos*V3
    out = ((row*L + page*P + pos)*H + head)*D + dim
    tl.store(keys+out, tl.load(k+ko, valid, 0), valid)
    tl.store(values+out, tl.load(v+vo, valid, 0), valid)

def batched(layer, q, k, v, groups, group_masked=False):
    ctx = get_forward_context()
    cache = TorchPagedKV(ctx.kv_cache_data[f'layer_{layer.layer_num}'],layer.num_kv_heads,layer.head_dim)
    cache.write(ctx.attn_metadata.slot_mapping[:len(k)],k,v)
    single = len(groups)==1
    out = None if single else torch.empty_like(q)
    for which,pos,blocks,offset in groups:
        length = len(pos)
        n,h,d = len(which),layer.num_kv_heads,layer.head_dim
        keys = q.new_empty((n,length,h,d))
        vals = torch.empty_like(keys)
        _gather[(triton.cdiv(length,cache.block_size),n,h)](
            cache.key,cache.value,blocks,keys,vals,length,h,d,cache.block_size,
            cache.pack,*cache.key.stride(),*cache.value.stride())
        keys,vals = keys.transpose(1,2),vals.transpose(1,2)
        queries = q.contiguous() if single else q[which]
        masked = layer.window is not None and length>=layer.window
        if masked:
            mask = q.new_zeros(length).masked_fill_(pos<length-layer.window,float('-inf')).view(1,1,1,length)
            queries = queries.reshape(n,h,-1,d)
            value = F.scaled_dot_product_attention(queries,keys,vals,attn_mask=mask,dropout_p=0.0,is_causal=False,scale=layer.scale)
        else:
            value = F.scaled_dot_product_attention(queries.unsqueeze(2),keys,vals,dropout_p=0.0,is_causal=False,scale=layer.scale,enable_gqa=True)
        value = value.reshape(n,layer.num_heads,d)
        if single:return value
        out.index_copy_(0,which,value)
    return out
