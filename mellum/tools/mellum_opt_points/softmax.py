"""BF16 row softmax with FP32 register arithmetic, no full FP32 probability tensor."""
import torch,triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _softmax(X,Y,N:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,B)
    x=tl.load(X+row*N+col,col<N,other=-float('inf')).to(tl.float32)
    m=tl.max(x,0)
    e=libdevice.exp(x-m)
    denom=tl.sum(e,0)
    inv=tl.div_rn(1.,denom)
    out=e*inv
    tl.store(Y+row*N+col,out,col<N)

def softmax(x):
    assert x.is_contiguous()
    y=torch.empty_like(x)
    n=x.size(-1);block=triton.next_power_of_2(n)
    _softmax[(x.numel()//n,)](x,y,n,block,num_warps=16 if block>=8192 else 4,num_stages=1,enable_fp_fusion=False)
    return y
