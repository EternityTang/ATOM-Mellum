"""Opt-in decode paths for single-GPU BF16 Mellum; no vLLM runtime dependency."""
import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.moe.moe_op import fused_moe
from atom.models.mellum_naive_ops import TorchPagedKV
from atom.utils.forward_context import get_forward_context
from paged_kernel import kernel_paged_attention_2d

def paged_read(query,cache,table,lengths,cu,window,scale):
    out=torch.empty_like(query)
    k,v=cache.key,cache.value
    n,h,d=query.shape
    kv=k.size(1)
    kernel_paged_attention_2d[(n,kv)](
        out,query,k,v,None,table,lengths,None,scale,None,None,1.0,
        h,h//kv,max(16,triton.next_power_of_2(h//kv)),table.stride(0),
        query.stride(0),query.stride(1),out.stride(0),out.stride(1),
        16,cache.block_size,d,triton.next_power_of_2(d),False,window or 0,cache.pack,
        *k.stride(),*v.stride(),False,cu,False,False,0)
    return out

def decode_paged(layer,q,k,v):
    fc=get_forward_context()
    meta=fc.attn_metadata
    cache=TorchPagedKV(fc.kv_cache_data[f'layer_{layer.layer_num}'],layer.num_kv_heads,layer.head_dim)
    cache.write(meta.slot_mapping[:len(k)],k,v)
    return paged_read(q,cache,meta.block_tables,meta.context_lens,meta.cu_seqlens_q,layer.window,layer.scale)

def grouped_experts(x,w13,w2,ids,weights):
    n,top=ids.shape
    num_experts,wide,_=w13.shape
    block=16
    config=dict(BLOCK_SIZE_M=block,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64,GROUP_SIZE_M=1)
    max_ids=n*top+num_experts*(block-1)
    # Unwritten expert padding lanes must point past the last assignment.
    sorted_ids=torch.full((max_ids,),n*top,dtype=torch.int32,device=x.device)
    blocks=torch.empty((triton.cdiv(max_ids,block),),dtype=torch.int32,device=x.device)
    count=torch.empty((1,),dtype=torch.int32,device=x.device)
    moe_align_block_size_triton(ids.int(),num_experts,block,sorted_ids,blocks,count)
    gateup=x.new_empty((n,top,wide))
    def gemm(a,b,c,stride):
        fused_moe(a,b,c,None,None,None,weights,ids,sorted_ids,blocks,count,
                  False,stride,tl.bfloat16,False,False,False,config=config)
    gemm(x,w13,gateup,top)
    gate,up=gateup.chunk(2,dim=-1)
    mid=(F.silu(gate)*up).reshape(n*top,-1)
    down=x.new_empty((n,top,x.shape[-1]))
    gemm(mid,w2,down,1)
    # Preserve the HF BF16 weighting boundary and ascending-expert accumulation.
    weighted=down*weights.unsqueeze(-1)
    order=ids.argsort(dim=-1)
    ordered=weighted.gather(1,order.unsqueeze(-1).expand_as(weighted))
    result=ordered[:,0].clone()
    for slot in range(1,top):
        result=result+ordered[:,slot]
    return result

def install(mode):
    from atom.models import mellum_rocm_precision as p
    attention=p.NaiveSDPAAttention.forward
    experts=p.torch_expert_forward
    from batched_sdpa import batched,build_groups
    groups=None
    def attend(layer,q,k,v,pos):
        nonlocal groups
        fc=get_forward_context()
        if ('batched' in mode() or 'mixed' in mode()) and not fc.context.is_prefill and not fc.context.is_dummy_run and fc.attn_metadata.max_seqlen_q==1:
            if layer.layer_num==0:
                groups=build_groups(fc.attn_metadata)
            return batched(layer,q,k,v,groups,group_masked='mixed' in mode())
        if 'paged' in mode() and not fc.context.is_prefill and not fc.context.is_dummy_run and fc.attn_metadata.max_seqlen_q==1:
            return decode_paged(layer,q,k,v)
        return attention(layer,q,k,v,pos)
    def dispatch(x,w13,w2,ids,weights):
        fc=get_forward_context()
        if 'grouped' in mode() and not fc.context.is_prefill and not fc.context.is_dummy_run:
            return grouped_experts(x,w13,w2,ids,weights)
        if 'packed' in mode() and not fc.context.is_prefill and not fc.context.is_dummy_run:
            from naive_alignment_ops import packed_loop_experts
            from types import SimpleNamespace
            layer=SimpleNamespace(num_experts=w13.shape[0],gate_up_proj=w13,down_proj=w2)
            return packed_loop_experts(layer,x,ids,weights)
        return experts(x,w13,w2,ids,weights)
    p.NaiveSDPAAttention.forward=attend
    p.torch_expert_forward=dispatch
