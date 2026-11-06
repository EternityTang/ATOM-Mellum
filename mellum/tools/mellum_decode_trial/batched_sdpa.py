"""Share page-table addressing and retain the reference SDPA mask/dispatch rule."""
from collections import defaultdict
import torch
from torch.nn import functional as F
from atom.models.mellum_naive_ops import TorchPagedKV
from atom.utils.forward_context import get_forward_context

def build_groups(meta):
    cohorts=defaultdict(list)
    for i,length in enumerate(meta.context_lens.tolist()):
        cohorts[length].append(i)
    device=meta.block_tables.device
    groups=[]
    for length,rows in cohorts.items():
        # Limit simultaneous expanded BF16 K/V to ~256 MiB, or one sequence.
        limit=max(1,256*1024**2//(2*2*32*128*length))
        pos=torch.arange(length,device=device)
        for start in range(0,len(rows),limit):
            which=torch.tensor(rows[start:start+limit],device=device)
            blocks=meta.block_tables[which[:,None],pos[None,:]//16].long()
            groups.append((which,pos,blocks,pos%16))
    return groups

def batched(layer,q,k,v,groups,group_masked=False):
    fc=get_forward_context()
    cache=TorchPagedKV(fc.kv_cache_data[f'layer_{layer.layer_num}'],layer.num_kv_heads,layer.head_dim)
    cache.write(fc.attn_metadata.slot_mapping[:len(k)],k,v)
    out=torch.empty_like(q)
    for which,pos,blocks,offset in groups:
        length=len(pos)
        keys=cache.key[blocks,:,:,offset,:].reshape(len(which),length,layer.num_kv_heads,layer.head_dim).transpose(1,2)
        vals=cache.value[blocks,:,:,offset].transpose(1,2)
        masked=layer.window is not None and length>=layer.window
        mask=None
        if masked:
            mask=q.new_zeros(length).masked_fill_(pos<length-layer.window,float('-inf')).view(1,1,1,length)
        if masked and group_masked:
            # All query heads in a GQA group share the same keys, positions and mask.
            # Treat the 8 heads as the SDPA query axis for a single decode position.
            queries=q[which].reshape(len(which),layer.num_kv_heads,-1,layer.head_dim)
            value=F.scaled_dot_product_attention(queries,keys,vals,attn_mask=mask,
                    dropout_p=0.0,is_causal=False,scale=layer.scale)
            value=value.reshape(len(which),layer.num_heads,1,layer.head_dim)
        else:
            if masked:
                repeat=layer.num_heads//layer.num_kv_heads
                keys=keys.repeat_interleave(repeat,dim=1)
                vals=vals.repeat_interleave(repeat,dim=1)
            value=F.scaled_dot_product_attention(q[which].unsqueeze(2),keys,vals,attn_mask=mask,dropout_p=0.0,
                    is_causal=False,scale=layer.scale,enable_gqa=not masked)
        out.index_copy_(0,which,value.squeeze(2))
    return out
