"""Batch elementwise norm work while preserving each request's mean reduction."""
import torch
def pointwise_batch(x,weight,eps,residual=None):
 values=x if residual is None else x+residual
 fp32=values.float()
 squares=fp32.square()
 means=torch.cat([squares[i:i+1].mean(-1,keepdim=True) for i in range(len(x))],0)
 normalized=fp32*torch.rsqrt(means+eps)
 output=weight*normalized.to(values.dtype)
 return output if residual is None else (output,values)
