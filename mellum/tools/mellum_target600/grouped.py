"""Grouped expert projections with BF16 activation, weighting and ordered reduction."""
import torch
import triton
import triton.language as tl
from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.moe.moe_op import fused_moe

@triton.jit
def _act(G, M, W:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+row*2*W+col,col<W,other=0).to(tl.float32)
    u=tl.load(G+row*2*W+W+col,col<W,other=0).to(tl.float32)
    a=(g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(M+row*W+col,a*u,col<W)

@triton.jit
def _sum(DOWN,W,S,Y,K:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    n=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    acc=tl.full((BLOCK,),0,tl.float32)
    for i in tl.static_range(K):
        j=n*K+tl.load(S+n*K+i)
        w=tl.load(W+j).to(tl.float32)
        val=tl.load(DOWN+j*D+col,col<D,other=0).to(tl.float32)
        prod=(w*val).to(tl.bfloat16).to(tl.float32)
        acc=(acc+prod).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+n*D+col,acc,col<D)

def experts(x,w13,w2,ids,weights):
    n,top=ids.shape
    e,wide,d=w13.shape
    count_max=n*top+e*15
    order=torch.full((count_max,),n*top,dtype=torch.int32,device=x.device)
    blocks=torch.empty((triton.cdiv(count_max,16),),dtype=torch.int32,device=x.device)
    count=torch.empty(1,dtype=torch.int32,device=x.device)
    moe_align_block_size_triton(ids.int(),e,16,order,blocks,count)
    cfg=dict(BLOCK_SIZE_M=16,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64,GROUP_SIZE_M=1)
    def gemm(a,b,out,stride):
        fused_moe(a,b,out,None,None,None,weights,ids,order,blocks,count,False,
                  stride,tl.bfloat16,False,False,False,config=cfg)
    gates=x.new_empty((n,top,wide))
    gemm(x,w13,gates,top)
    width=wide//2
    mid=x.new_empty((n*top,width))
    _act[(n*top,triton.cdiv(width,256))](gates,mid,width,256,enable_fp_fusion=False)
    down=x.new_empty((n,top,d))
    gemm(mid,w2,down,1)
    out=x.new_empty((n,d))
    _sum[(n,triton.cdiv(d,256))](down,weights.contiguous(),ids.argsort(-1).contiguous(),out,top,d,256,enable_fp_fusion=False)
    return out
