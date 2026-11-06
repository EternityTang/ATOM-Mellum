"""Dispatch-only TorchScript helpers; preserve GEMM shapes and BF16 boundaries."""
from typing import Optional,List,Tuple
import torch
from torch.nn import functional as F

@torch.jit.script
def linear_rows(x:torch.Tensor,w:torch.Tensor,b:Optional[torch.Tensor]=None)->torch.Tensor:
    out=torch.jit.annotate(List[torch.Tensor],[])
    for i in range(x.size(0)):
        out.append(F.linear(x[i:i+1],w,b))
    return torch.cat(out,0)

@torch.jit.script
def norm_rows(x:torch.Tensor,w:torch.Tensor,eps:float,residual:Optional[torch.Tensor]=None)->Tuple[torch.Tensor,torch.Tensor]:
    out=torch.jit.annotate(List[torch.Tensor],[])
    vals=torch.jit.annotate(List[torch.Tensor],[])
    for i in range(x.size(0)):
        value=x[i:i+1] if residual is None else x[i:i+1]+residual[i:i+1]
        fp32=value.float()
        normalized=fp32*torch.rsqrt(fp32.square().mean(-1,keepdim=True)+eps)
        out.append(w*normalized.to(value.dtype))
        if residual is not None:vals.append(value)
    return torch.cat(out,0),x if residual is None else torch.cat(vals,0)

@torch.jit.script
def projected_cap8(x:torch.Tensor,w:torch.Tensor,segments:List[Tuple[int,int,int]])->torch.Tensor:
    out=torch.jit.annotate(List[torch.Tensor],[])
    for expert,begin,end in segments:
        for start in range(begin,end,8):
            stop=min(start+8,end)
            out.append(F.linear(x[start:stop],w[expert]))
    return torch.cat(out,0)
