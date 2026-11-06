"""Address-only KV materialization and TorchScript dispatch with unchanged GEMMs."""
from typing import List,Tuple
import torch,triton
import triton.language as tl

@torch.jit.script
def projected(x:torch.Tensor,w:torch.Tensor,segments:List[Tuple[int,int,int]])->torch.Tensor:
    out=torch.jit.annotate(List[torch.Tensor],[])
    for expert,start,stop in segments:
        out.append(torch.nn.functional.linear(x[start:stop],w[expert]))
    return torch.cat(out)

@triton.jit
def _gather_heads(K,V,B,KO,VO,L,S,T,
                 H:tl.constexpr,D:tl.constexpr,G:tl.constexpr,P:tl.constexpr,PACK:tl.constexpr,
                 K0,K1,K2,K3,K4,V0,V1,V2,V3):
    chunk=tl.program_id(0);row=tl.program_id(1);head=tl.program_id(2)
    local=chunk*32+tl.arange(0,32)
    position=S+local
    physical=tl.load(B+row*L+position,local<T,other=0).to(tl.int64)
    offset=position%P
    d=tl.arange(0,D)
    h=head//G
    ki=physical[:,None]*K0+h*K1+(d[None,:]//PACK)*K2+offset[:,None]*K3+(d[None,:]%PACK)*K4
    vi=physical[:,None]*V0+h*V1+d[None,:]*V2+offset[:,None]*V3
    out=((row*(H*G)+head)*T+local[:,None])*D+d[None,:]
    mask=local[:,None]<T
    tl.store(KO+out,tl.load(K+ki,mask,0),mask)
    tl.store(VO+out,tl.load(V+vi,mask,0),mask)

def gather_heads(cache,blocks,length,start,n,heads):
    size=length-start
    k=torch.empty((n,heads,size,cache.width),device=cache.key.device,dtype=cache.key.dtype)
    v=torch.empty_like(k)
    _gather_heads[(triton.cdiv(size,32),n,heads)](
        cache.key,cache.value,blocks,k,v,length,start,size,cache.heads,cache.width,
        heads//cache.heads,cache.block_size,cache.pack,*cache.key.stride(),*cache.value.stride())
    return k,v
