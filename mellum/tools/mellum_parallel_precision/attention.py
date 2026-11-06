"""Batch equal effective KV lengths; preserve each request's visible keys."""
from collections import defaultdict
from types import MethodType
import torch
from atom.utils.forward_context import get_forward_context
from atom.models.mellum_naive_ops import TorchPagedKV
from mellum_eager_precision.install import eager_math
from mellum_eager_precision.repeat_ops import repeat

def install_attention(model,counts):
    for layer in model.model.layers:
        a=layer.self_attn;impl=a.attn.impl;original=impl.forward_impl_server_mode
        def forward(self,q,k,v,position=None,q_scale=None,qkv=None,a=a,original=original):
            ctx=get_forward_context()
            if ctx.context.is_dummy_run or ctx.context.is_prefill:
                return original(q,k,v,position,q_scale,qkv)
            if q_scale is not None or qkv is not None:raise ValueError('Expected ordinary BF16 Q/K/V')
            q=q.view(-1,a.num_heads,a.head_dim);k=k.view(-1,a.num_kv_heads,a.head_dim);v=v.view_as(k)
            meta=ctx.attn_metadata
            cache=TorchPagedKV(ctx.kv_cache_data[f'layer_{a.layer_num}'],a.num_kv_heads,a.head_dim)
            cache.write(meta.slot_mapping[:len(k)],k,v)
            groups=defaultdict(list)
            for i,length in enumerate(meta.context_lens.tolist()):groups[min(length,a.window) if a.window else length].append(i)
            out=torch.empty_like(q)
            for length,rows in groups.items():
                # Bound expanded K/V temporaries to approximately 256 MiB.
                cap=max(1,256*1024**2//(2*2*a.num_heads*a.head_dim*length))
                for off in range(0,len(rows),cap):
                    which=torch.tensor(rows[off:off+cap],device=q.device,dtype=torch.long)
                    positions=meta.context_lens[which].long()[:,None]-length+torch.arange(length,device=q.device)[None]
                    blocks=meta.block_tables[which[:,None],positions//cache.block_size].long()
                    offset=positions%cache.block_size
                    keys=cache.key[blocks,:,:,offset,:].reshape(len(which),length,a.num_kv_heads,a.head_dim).transpose(1,2).contiguous()
                    values=cache.value[blocks,:,:,offset].transpose(1,2).contiguous()
                    keys,values=repeat(keys,values,a.num_heads//a.num_kv_heads)
                    queries=q.index_select(0,which).unsqueeze(2)
                    mask=q.new_zeros((1,1,1,length)) if a.window else None
                    output=eager_math(queries,keys,values,mask,a.scale).reshape(len(which),a.num_heads,a.head_dim)
                    out.index_copy_(0,which,output)
                    if len(which)>1:
                        counts['batched_attention_calls']+=1
                        counts['max_attention_batch']=max(counts.get('max_attention_batch',0),len(which))
            return out.flatten(1)
        impl.forward_impl_server_mode=MethodType(forward,impl)
