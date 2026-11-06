"""Layer-batched decode with isolated numerical operations and explicit ablations."""
import torch
from torch.nn import functional as F
from atom.utils.forward_context import get_forward_context
from atom.models.mellum_naive_ops import rotate
from mellum_reuse_request import install as install_reuse

MODES=('segmented','qkv_batch','moe_batch','dense_batch','bmm_dense','attention_batch','norm_isolated','rope_isolated','audit','norm_attention','norm_moe','norm_bmm','norm_qkv','norm_dense','audit_late','norm_cap8','norm_cap1')

def install(model,mode='norm_cap8'):
    if mode not in MODES:raise ValueError(mode)
    install_reuse(model,'framework_eager')
    if mode in ('norm_cap8','norm_cap1'):
        import exact
        cap=8 if mode=='norm_cap8' else 1
        def projected(x,w,segments):
            chunks=[(expert,start,min(start+cap,stop))
                for expert,begin,stop in segments for start in range(begin,stop,cap)]
            counts['expert_matrix_calls']+=len(chunks)
            counts['max_expert_matrix_rows']=max(counts['max_expert_matrix_rows'],max(stop-start for _,start,stop in chunks))
            counts['max_expert_tokens_before_chunk']=max(counts['max_expert_tokens_before_chunk'],max(stop-start for _,start,stop in segments))
            return torch.cat([F.linear(x[start:stop],w[expert]) for expert,start,stop in chunks])
        exact.projected=projected
    counts=dict(decode_batches=0,max_decode_batch=0,batch_qkv_calls=0,
                batch_moe_calls=0,segmented_moe_calls=0,batched_attention_calls=0,expert_matrix_calls=0,max_expert_matrix_rows=0,max_expert_tokens_before_chunk=0)
    def decoding():
        c=get_forward_context()
        return not c.context.is_dummy_run and not c.context.is_prefill
    def linear(x,w,b=None,role='qkv'):
        batch=(mode in ('dense_batch','norm_dense') or mode in ('qkv_batch','norm_qkv') and role=='qkv')
        if batch:
            if role=='qkv':counts['batch_qkv_calls']+=1
            return F.linear(x,w,b)
        if mode in ('bmm_dense','norm_bmm'):
            # Keep each request's GEMM M dimension equal to one while using
            # one batched call. Numerical equivalence is measured, not assumed.
            out=torch.bmm(x[:,None,:],w.T[None].expand(len(x),-1,-1)).squeeze(1)
            return out if b is None else out+b
        return torch.cat([F.linear(x[i:i+1],w,b) for i in range(len(x))],0)
    for layer in model.model.layers:
        attn=layer.self_attn;original_attention=attn.forward
        def attention(positions,hidden_states,original=original_attention,a=attn,**kwargs):
            if not decoding():return original(positions,hidden_states,**kwargs)
            w,b=a.qkv_proj.weight,a.qkv_proj.bias
            off=0;parts=[]
            for size in (a.q_size,a.kv_size,a.kv_size):
                parts.append(linear(hidden_states,w[off:off+size],None if b is None else b[off:off+size]))
                off+=size
            q,k,v=parts
            q=a.q_norm(q.view(-1,a.num_heads,a.head_dim))
            k=a.k_norm(k.view(-1,a.num_kv_heads,a.head_dim))
            q,k=rotate(q,k,*a.rope(hidden_states,positions)[a.kind])
            out=a.attn(query=q.flatten(1),key=k.flatten(1),value=v,positions=positions,**kwargs)
            out=out.reshape(len(hidden_states),-1).contiguous()
            return linear(out,a.o_proj.weight,a.o_proj.bias,'output')
        attn.forward=attention
        gate=layer.mlp.gate;original_gate=gate.forward
        def gate_forward(x,original=original_gate,g=gate):
            return linear(x,g.weight,g.bias,'gate') if decoding() else original(x)
        gate.forward=gate_forward
        method=layer.mlp.experts.quant_method;original_apply=method.apply
        def apply(layer,x,router_logits,top_k,renormalize,original=original_apply,**options):
            if not decoding():return original(layer,x,router_logits,top_k,renormalize,**options)
            if mode in ('moe_batch','norm_moe','norm_cap8','norm_cap1'):
                counts['batch_moe_calls']+=1
                return original(layer,x,router_logits,top_k,renormalize,**options)
            counts['segmented_moe_calls']+=1
            return torch.cat([original(layer,x[i:i+1],router_logits[i:i+1],top_k,renormalize,**options) for i in range(len(x))],0)
        method.apply=apply
    if mode in ('attention_batch','norm_attention'):
        from .attention import install_attention
        install_attention(model,counts)
    if mode.startswith('norm_'):
        from atom.models.mellum_rocm_precision import MellumPrecisionNorm
        for module in model.modules():
            if isinstance(module,MellumPrecisionNorm):
                original=module.forward
                def norm(x,residual=None,x_scale=None,original=original):
                    if not decoding():return original(x,residual,x_scale)
                    outs=[original(x[i:i+1],None if residual is None else residual[i:i+1],x_scale) for i in range(len(x))]
                    return torch.cat(outs) if residual is None else tuple(torch.cat([o[j] for o in outs]) for j in range(2))
                module.forward=norm
    if mode=='rope_isolated':
        rope=model.model.layers[0].self_attn.rope
        original=rope.forward
        def rope_forward(x,positions):
            if not decoding():return original(x,positions)
            result={}
            for kind in rope.kinds:
                inv=getattr(rope,kind+'_inv_freq').to(device=x.device,dtype=torch.float32)
                cos=[];sin=[]
                for i in range(len(positions)):
                    freq=positions[i:i+1,None].float()*inv
                    angles=torch.cat((freq,freq),dim=-1)
                    cos.append((angles.cos()*rope.factors[kind]).to(x.dtype)[:,None,:])
                    sin.append((angles.sin()*rope.factors[kind]).to(x.dtype)[:,None,:])
                result[kind]=(torch.cat(cos),torch.cat(sin))
            return result
        rope.forward=rope_forward
    baseline_forward=model.forward
    model.parallel_serial_forward=baseline_forward
    def forward(*args,**kwargs):
        if not decoding():return baseline_forward(*args,**kwargs)
        ctx=get_forward_context();n=len(ctx.attn_metadata.context_lens)
        if ctx.attn_metadata.max_seqlen_q!=1:raise ValueError('Only ordinary decode is supported')
        counts['decode_batches']+=1;counts['max_decode_batch']=max(counts['max_decode_batch'],n)
        previous=model.request_isolation_active
        model.request_isolation_active=True
        try:return baseline_forward(*args,**kwargs)
        finally:model.request_isolation_active=previous
    model.forward=forward
    original_logits=model.compute_logits
    def logits(hidden):
        if not decoding() or mode not in ('dense_batch','bmm_dense','norm_dense','norm_bmm'):return original_logits(hidden)
        return linear(hidden,model.lm_head.weight,model.lm_head.bias,'head')
    model.compute_logits=logits
    model.parallel_metadata=dict(mode=mode,torch=torch.__version__,counts=counts,
         prefill='per-request full prefill',decode='one shared backbone forward for all requests',
         projections='batched ablation' if mode in ('qkv_batch','dense_batch','bmm_dense','norm_qkv','norm_dense','norm_bmm') else 'per-request F.linear',
         experts='cross-request expert grouping' if mode in ('moe_batch','norm_moe','norm_cap8','norm_cap1') else 'per-request expert arithmetic',
         backbone=model.reuse_stage_metadata)
    return model.parallel_metadata
