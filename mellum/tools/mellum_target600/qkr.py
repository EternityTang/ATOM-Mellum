import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from atom.models.mellum_naive_ops import NaiveSDPAAttention

@triton.jit
def _head(X, W, C, S, OUT, SS, H:tl.constexpr, D:tl.constexpr, E:tl.constexpr):
    n=tl.program_id(0)
    head=tl.program_id(1)
    col=tl.arange(0,D)
    flip=(col+D//2)%D
    base=X+n*SS+head*D
    x=tl.load(base+col).to(tl.float32)
    f=tl.load(base+flip).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,axis=0)/D+E)
    a=((x*inv).to(tl.bfloat16).to(tl.float32)*tl.load(W+col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    b=((f*inv).to(tl.bfloat16).to(tl.float32)*tl.load(W+flip).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    l=(a*tl.load(C+n*D+col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    r=(tl.where(col<D//2,-b,b)*tl.load(S+n*D+col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT+(n*H+head)*D+col,l+r)

def attn(layer,positions,x):
    sizes=(layer.q_size,layer.kv_size,layer.kv_size)
    q,k,v=F.linear(x,layer.qkv_proj.weight,layer.qkv_proj.bias).split(sizes,dim=-1)
    c,s=layer.rope(x,positions)[layer.kind]
    n,d=len(x),layer.head_dim
    def head(a,h,norm):
        out=x.new_empty((n,h,d))
        _head[(n,h)](a,norm.weight,c,s,out,a.stride(0),h,d,norm.eps,enable_fp_fusion=False)
        return out
    q=head(q,layer.num_heads,layer.q_norm)
    k=head(k,layer.num_kv_heads,layer.k_norm)
    v=v.view(n,layer.num_kv_heads,d)
    out=NaiveSDPAAttention.forward(layer,q,k,v,positions)
    return F.linear(out.reshape(n,-1).contiguous(),layer.o_proj.weight,layer.o_proj.bias)
