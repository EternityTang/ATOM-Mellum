"""Address-only paged KV gather; each request keeps its original attention arithmetic."""
from types import MethodType
import torch,triton
import triton.language as tl
from atom.utils.forward_context import get_forward_context
from atom.models.mellum_naive_ops import TorchPagedKV
@triton.jit
def gather(K,V,T,P,KO,VO,TS0,TS1,K0,K1,K2,K3,K4,V0,V1,V2,V3,H:tl.constexpr,KH:tl.constexpr,D:tl.constexpr,BS:tl.constexpr,PACK:tl.constexpr):
 row=tl.program_id(1);head=tl.program_id(2);local=tl.program_id(0)*32+tl.arange(0,32)
 length=tl.load(P+row*3);shift=tl.load(P+row*3+1);base=tl.load(P+row*3+2).to(tl.int64)
 pos=shift+local;physical=tl.load(T+row*TS0+(pos//BS)*TS1,local<length,0).to(tl.int64)
 col=tl.arange(0,D);kh=head//(H//KH);off=pos%BS
 ki=physical[:,None]*K0+kh*K1+(col[None,:]//PACK)*K2+off[:,None]*K3+(col[None,:]%PACK)*K4
 vi=physical[:,None]*V0+kh*V1+col[None,:]*V2+off[:,None]*V3
 dest=base+(head*length+local[:,None])*D+col[None,:];mask=local[:,None]<length
 tl.store(KO+dest,tl.load(K+ki,mask,0),mask);tl.store(VO+dest,tl.load(V+vi,mask,0),mask)
def plan(lengths,window,heads,width,device):
 sizes=[min(x,window) if window is not None else x for x in lengths];starts=[];rows=[];total=0
 for original,size in zip(lengths,sizes):
  starts.append(total);rows.append((size,original-size,total));total+=heads*size*width
 return sizes,starts,total,torch.tensor(rows,device=device,dtype=torch.int64)
def read(cache,tables,p,heads):
 sizes,starts,total,meta=p
 k=torch.empty(total,device=cache.key.device,dtype=cache.key.dtype);v=torch.empty_like(k)
 gather[(triton.cdiv(max(sizes),32),len(sizes),heads)](cache.key,cache.value,tables,meta,k,v,*tables.stride(),*cache.key.stride(),*cache.value.stride(),heads,cache.heads,cache.width,cache.block_size,cache.pack)
 return k,v
def install(model,counts):
 from mellum_eager_precision import install as unused
 import importlib
 math_module=importlib.import_module('mellum_eager_precision.install')
 def begin(*args):
  c=get_forward_context();c.dual_kv_plans={};c.dual_lengths=None
 model.register_forward_pre_hook(begin,prepend=True)
 for layer in model.model.layers:
  a=layer.self_attn;impl=a.attn.impl;original=impl.forward_impl_server_mode
  def forward(self,q,k,v,position=None,q_scale=None,qkv=None,a=a,original=original):
   c=get_forward_context()
   if c.context.is_dummy_run or c.context.is_prefill:return original(q,k,v,position,q_scale,qkv)
   assert q_scale is None and qkv is None and c.attn_metadata.max_seqlen_q==1
   q=q.view(-1,a.num_heads,a.head_dim);k=k.view(-1,a.num_kv_heads,a.head_dim);v=v.view_as(k)
   meta=c.attn_metadata
   if c.dual_lengths is None:c.dual_lengths=meta.context_lens.tolist()
   lengths=c.dual_lengths
   cache=TorchPagedKV(c.kv_cache_data[f'layer_{a.layer_num}'],a.num_kv_heads,a.head_dim)
   cache.write(meta.slot_mapping[:len(k)],k,v)
   if a.window not in c.dual_kv_plans:c.dual_kv_plans[a.window]=plan(lengths,a.window,a.num_heads,a.head_dim,q.device)
   p=c.dual_kv_plans[a.window];keys,values=read(cache,meta.block_tables,p,a.num_heads)
   out=torch.empty_like(q)
   for i,(length,base) in enumerate(zip(p[0],p[1])):
    size=a.num_heads*length*a.head_dim
    ki=keys[base:base+size].view(1,a.num_heads,length,a.head_dim)
    vi=values[base:base+size].view_as(ki)
    mask=q.new_zeros((1,1,1,length))
    out[i:i+1]=math_module.eager_math(q[i:i+1].transpose(0,1)[None],ki,vi,mask,a.scale)[0].transpose(0,1)
   counts['kv_gathers']+=1
   return out.flatten(1)
  impl.forward_impl_server_mode=MethodType(forward,impl)
