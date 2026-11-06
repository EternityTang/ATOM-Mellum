import torch
import triton
import triton.language as tl

@triton.jit
def _norm(X, R, W, Y, Z, WIDTH:tl.constexpr, EPS:tl.constexpr,
          RES:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    offs=row*WIDTH+col
    x=tl.load(X+offs,col<WIDTH,other=0).to(tl.float32)
    if RES:
        r=tl.load(R+offs,col<WIDTH,other=0).to(tl.float32)
        x=(x+r).to(tl.bfloat16).to(tl.float32)
        tl.store(Z+offs,x,col<WIDTH)
    w=tl.load(W+col,col<WIDTH,other=0).to(tl.float32)
    u=x*tl.rsqrt(tl.sum(x*x,axis=0)/WIDTH+EPS)
    tl.store(Y+offs,u.to(tl.bfloat16).to(tl.float32)*w,col<WIDTH)

def norm(x,r,w,eps):
    y=torch.empty_like(x)
    z=torch.empty_like(x) if r is not None else y
    d=x.shape[-1]
    _norm[(x.numel()//d,)](x,r if r is not None else x,w,y,z,d,eps,r is not None,triton.next_power_of_2(d),enable_fp_fusion=False)
    return y if r is None else (y,z)

@triton.jit
def _rotate(X,C,S,Y,H:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0)
    dim=tl.arange(0,D)
    a=tl.load(X+row*D+dim).to(tl.float32)
    b=tl.load(X+row*D+((dim+D//2)%D)).to(tl.float32)
    b=tl.where(dim<D//2,-b,b)
    p=row//H*D+dim
    cos=tl.load(C+p).to(tl.float32)
    sin=tl.load(S+p).to(tl.float32)
    left=(a*cos).to(tl.bfloat16).to(tl.float32)
    right=(b*sin).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+row*D+dim,left+right)

def rotate(x,c,s):
    y=torch.empty_like(x)
    n,h,d=x.shape
    _rotate[(n*h,)](x,c,s,y,h,d,enable_fp_fusion=False)
    return y
