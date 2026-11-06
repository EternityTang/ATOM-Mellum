"""Bitwise K/V head replication with contiguous source and destination."""
import torch,triton
import triton.language as tl
@triton.jit
def _repeat(K,V,KO,VO,L,H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
    chunk=tl.program_id(0);dst_head=tl.program_id(1)
    offset=chunk*B+tl.arange(0,B)
    src_head=dst_head//(H*G)*H+(dst_head%(H*G))//G
    source=src_head*L*D+offset;target=dst_head*L*D+offset
    valid=offset<L*D
    tl.store(KO+target,tl.load(K+source,valid,0),valid)
    tl.store(VO+target,tl.load(V+source,valid,0),valid)
def repeat(k,v,groups=8):
    assert k.is_contiguous() and v.is_contiguous()
    n,h,l,d=k.shape
    ko=torch.empty((n,h*groups,l,d),device=k.device,dtype=k.dtype);vo=torch.empty_like(ko)
    _repeat[(triton.cdiv(l*d,4096),n*h*groups)](k,v,ko,vo,l,h,groups,d,4096,num_warps=8)
    return ko,vo
