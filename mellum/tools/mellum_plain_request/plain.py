"""Standalone Torch Mellum forward, one request per numerical execution.

Uses ATOM only for loaded Parameters, scheduling metadata and allocated BF16 KV
pages. No previous precision/decode adapters, fused experts or attention kernels.
The numerical order follows native Transformers 5.20.2 Mellum eager.
"""
import torch
from torch.nn import functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from atom.utils.forward_context import get_forward_context

def rms(x,weight,eps):
    f=x.float()
    return weight*(f*torch.rsqrt(f.pow(2).mean(-1,keepdim=True)+eps)).to(x.dtype)

def rotate(x,cos,sin):
    left,right=x.chunk(2,-1)
    return x*cos+torch.cat((-right,left),-1)*sin

def repeat(x,groups):
    # [token, kv-head, dim] -> [1, query-head, token, dim]
    x=x.transpose(0,1)[None]
    b,h,l,d=x.shape
    return x[:,:,None].expand(b,h,groups,l,d).reshape(b,h*groups,l,d)

def experts(x,w13,w2,ids,weights):
    out=torch.zeros_like(x)
    mask=F.one_hot(ids,num_classes=w13.shape[0]+1).permute(2,1,0)
    hit=(mask.sum((-1,-2))>0).nonzero()
    for expert in hit:
        expert=expert[0]
        slot,tokens=torch.where(mask[expert])
        gate,up=F.linear(x[tokens],w13[expert]).chunk(2,-1)
        down=F.linear(F.silu(gate)*up,w2[expert])
        weighted=down*weights[tokens,slot,None]
        out.index_add_(0,tokens,weighted.to(out.dtype))
    return out

class PlainRequestForward:
    def __init__(self,model):
        self.model=model
        self.config=model.config
        c=self.config;device=model.lm_head.weight.device
        self.inv={};self.factors={}
        for kind in sorted(set(c.layer_types)):
            p=c.rope_parameters[kind]
            with torch.device('cpu'):
                if p['rope_type']=='default':
                    inv=1.0/(p['rope_theta']**(torch.arange(0,c.head_dim,2,dtype=torch.float32)/c.head_dim))
                    factor=1.0
                else:
                    inv,factor=ROPE_INIT_FUNCTIONS[p['rope_type']](c,device=torch.device('cpu'),layer_type=kind)
            self.inv[kind]=inv.to(device=device,dtype=torch.float32)
            self.factors[kind]=factor

    def rope(self,pos,dtype):
        result={}
        for kind,inv in self.inv.items():
            freqs=pos[:,None].float()*inv
            angles=torch.cat((freqs,freqs),-1)
            result[kind]=((angles.cos()*self.factors[kind]).to(dtype)[:,None,:],
                          (angles.sin()*self.factors[kind]).to(dtype)[:,None,:])
        return result

    def attention(self,i,q,k,v,pos,seq,start,end,length):
        ctx=get_forward_context();meta=ctx.attn_metadata
        pages=ctx.kv_cache_data[f'layer_{i}']
        kp,vp=pages.k_cache,pages.v_cache
        assert kp.dtype==vp.dtype==torch.bfloat16
        bs,pack=kp.shape[3],kp.shape[4]
        h,d=self.config.num_key_value_heads,self.config.head_dim
        slots=meta.slot_mapping[start:end].long()
        # Real rows only: graph/dummy padding is excluded by the installer.
        blocks,offsets=slots//bs,slots%bs
        kp[blocks,:,:,offsets,:]=k.reshape(-1,h,d//pack,pack)
        vp[blocks,:,:,offsets]=v
        window=self.config.sliding_window if self.config.layer_types[i]=='sliding_attention' else None
        shift=max(0,length-len(q)-window+1) if window is not None else 0
        kpos=torch.arange(shift,length,device=q.device)
        if meta.block_tables is None:
            if length!=len(q):raise ValueError('Cached attention requires a block table')
            keys,values=k[shift:],v[shift:]
        else:
            physical=meta.block_tables[seq,kpos//bs].long()
            offsets=kpos%bs
            keys=kp[physical,:,:,offsets,:].reshape(-1,h,d)
            values=vp[physical,:,:,offsets]
        keys=repeat(keys,self.config.num_attention_heads//h)
        values=repeat(values,self.config.num_attention_heads//h)
        scores=torch.matmul(q.transpose(0,1)[None],keys.transpose(-2,-1))*(d**-.5)
        if window is not None or len(q)>1:
            visible=kpos[None,:]<=pos[:,None]
            if window is not None:visible &= kpos[None,:]>pos[:,None]-window
            mask=torch.zeros((len(q),len(kpos)),device=q.device,dtype=q.dtype)
            mask.masked_fill_(~visible,torch.finfo(q.dtype).min)
            scores=scores+mask[None,None]
        probs=F.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs,values)[0].transpose(0,1).contiguous().reshape(len(q),-1)

    def request(self,ids,pos,seq,start,end,length):
        c=self.config
        hidden=F.embedding(ids,self.model.model.embed_tokens.weight)
        rope=self.rope(pos,hidden.dtype)
        qsize=c.num_attention_heads*c.head_dim
        kvsize=c.num_key_value_heads*c.head_dim
        for i,layer in enumerate(self.model.model.layers):
            residual=hidden
            x=rms(hidden,layer.input_layernorm.weight,c.rms_norm_eps)
            attention=layer.self_attn
            wq,wk,wv=attention.qkv_proj.weight.split((qsize,kvsize,kvsize),0)
            bias=attention.qkv_proj.bias
            bq,bk,bv=(None,None,None) if bias is None else bias.split((qsize,kvsize,kvsize))
            q=F.linear(x,wq,bq).view(-1,c.num_attention_heads,c.head_dim)
            k=F.linear(x,wk,bk).view(-1,c.num_key_value_heads,c.head_dim)
            v=F.linear(x,wv,bv).view(-1,c.num_key_value_heads,c.head_dim)
            q=rms(q,attention.q_norm.weight,c.rms_norm_eps)
            k=rms(k,attention.k_norm.weight,c.rms_norm_eps)
            cos,sin=rope[c.layer_types[i]]
            q,k=rotate(q,cos,sin),rotate(k,cos,sin)
            out=self.attention(i,q,k,v,pos,seq,start,end,length)
            hidden=residual+F.linear(out,attention.o_proj.weight,attention.o_proj.bias)
            residual=hidden
            x=rms(hidden,layer.post_attention_layernorm.weight,c.rms_norm_eps)
            logits=F.linear(x,layer.mlp.gate.weight)
            weights,selected=F.softmax(logits,dim=-1,dtype=torch.float32).topk(c.num_experts_per_tok,dim=-1)
            if c.norm_topk_prob:weights=weights/weights.sum(-1,keepdim=True)
            weights=weights.to(logits.dtype)
            e=layer.mlp.experts
            hidden=residual+experts(x,e.w13_weight,e.w2_weight,selected,weights)
        return rms(hidden,self.model.model.norm.weight,c.rms_norm_eps)

    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None,**kwargs):
        ctx=get_forward_context()
        if ctx.context.is_dummy_run:
            return torch.zeros((len(input_ids),self.config.hidden_size),device=input_ids.device,dtype=self.model.lm_head.weight.dtype)
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError('Plain request path supports token IDs on one GPU only')
        meta=ctx.attn_metadata
        cu=meta.cu_seqlens_q.tolist();lengths=meta.context_lens.tolist()
        if cu[-1]!=len(input_ids) or len(cu)!=len(lengths)+1:
            raise ValueError('Unexpected padded/partial batch metadata')
        outputs=[]
        with torch._C.DisableTorchFunction():
            for seq,(start,end,length) in enumerate(zip(cu,cu[1:],lengths)):
                if end<=start:raise ValueError('Empty scheduled request')
                # With chunking disabled, each multi-token prefill must be whole.
                if end-start>1 and length!=end-start:
                    raise ValueError('Unexpected chunked prefill')
                outputs.append(self.request(input_ids[start:end],positions[start:end],seq,start,end,length))
        return torch.cat(outputs,0)

    def compute_logits(self,hidden):
        ctx=get_forward_context()
        cu=ctx.attn_metadata.cu_seqlens_q.tolist()
        # HF generate(logits_to_keep=1), separately for every request.
        with torch._C.DisableTorchFunction():
            return torch.cat([F.linear(hidden[end-1:end],self.model.lm_head.weight) for end in cu[1:]],0)

def install(model):
    cfg=model.atom_config;c=model.config
    if not torch.__version__.startswith('2.10.'):raise ValueError('Requires frozen Torch 2.10')
    if cfg.enable_chunked_prefill:raise ValueError('Disable prefill chunking for the single-full HF reference')
    if cfg.tensor_parallel_size!=1 or cfg.parallel_config.data_parallel_size!=1 or cfg.enable_expert_parallel:
        raise ValueError('Single GPU only')
    if not cfg.enforce_eager or cfg.compilation_config.level!=0 or cfg.speculative_config is not None or cfg.enable_tbo:
        raise ValueError('Requires eager, level 0, MTP/TBO off')
    if cfg.enable_prefix_caching or cfg.kv_cache_dtype!='bf16' or cfg.torch_dtype!=torch.bfloat16:
        raise ValueError('Requires BF16 and prefix cache off')
    if set(c.mlp_layer_types)!={'sparse'} or cfg.kv_cache_block_size!=16:
        raise ValueError('This checkpoint requires all-sparse Mellum, block16')
    if any(not getattr(l.mlp.experts.quant_method,'standard_layout',False) for l in model.model.layers):
        raise ValueError('Requires unshuffled BF16 expert weights')
    plain=PlainRequestForward(model)
    model.forward=plain.forward
    model.compute_logits=plain.compute_logits
    model.plain_request_impl=plain
    model.plain_request_metadata=dict(implementation='standalone Torch, full model per request',prefill='whole request, no padding or chunking',head='per request, last query only',experts='Torch eager loop with HF BF16 boundaries',torch=torch.__version__)
    return model.plain_request_metadata
