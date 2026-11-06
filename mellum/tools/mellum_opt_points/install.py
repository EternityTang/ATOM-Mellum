"""Three independently switchable candidates on the frozen norm_cap8 baseline."""
import os,importlib
import torch
from torch.nn import functional as F
from atom.utils.forward_context import get_forward_context
from atom.models.mellum_naive_ops import rotate
from mellum_parallel_precision import install as install_base
from .ops import linear_rows,norm_rows,projected_cap8

def install(model,attention='off',moe='off',dispatch='off'):
    assert attention in ('off','bounded')
    assert moe in ('off','script')
    assert dispatch in ('off','script')
    install_base(model,'norm_cap8')
    counts=dict(attention_tiled=0,expert_script_calls=0,linear_script_calls=0,norm_script_calls=0)
    def decoding():
        c=get_forward_context()
        return not c.context.is_dummy_run and not c.context.is_prefill
    if attention=='bounded':
        eager=importlib.import_module('mellum_eager_precision.install')
        original=eager.eager_math
        def bounded(q,k,v,mask,scale):
            # Keep both matmuls, masks, scaling and softmax row length intact.
            # Only split independent heads at softmax to bound FP32 temporaries.
            if q.size(-2)==1 or q.size(-3)*q.size(-2)*k.size(-2)*q.element_size()<=1024**3:
                return original(q,k,v,mask,scale)
            scores=torch.matmul(q,k.transpose(-2,-1))*scale
            if mask is not None:scores=scores+mask
            probs=torch.empty_like(scores)
            for h in range(0,scores.size(-3),4):
                probs[...,h:h+4,:,:]=F.softmax(scores[...,h:h+4,:,:],dim=-1,dtype=torch.float32).to(q.dtype)
            counts['attention_tiled']+=1
            return torch.matmul(probs,v)
        eager.eager_math=bounded
    if moe=='script':
        import exact
        def projected(x,w,segments):
            counts['expert_script_calls']+=1
            return projected_cap8(x,w,segments)
        exact.projected=projected
    if dispatch=='script':
        def linear(x,w,b=None):
            counts['linear_script_calls']+=1
            return linear_rows(x,w,b)
        for layer in model.model.layers:
            a=layer.self_attn;old=a.forward
            def attention_forward(positions,hidden_states,original=old,a=a,**kwargs):
                if not decoding():return original(positions,hidden_states,**kwargs)
                w,b=a.qkv_proj.weight,a.qkv_proj.bias
                parts=[];off=0
                for size in (a.q_size,a.kv_size,a.kv_size):
                    parts.append(linear(hidden_states,w[off:off+size],None if b is None else b[off:off+size]))
                    off+=size
                q,k,v=parts
                q=a.q_norm(q.view(-1,a.num_heads,a.head_dim))
                k=a.k_norm(k.view(-1,a.num_kv_heads,a.head_dim))
                q,k=rotate(q,k,*a.rope(hidden_states,positions)[a.kind])
                out=a.attn(query=q.flatten(1),key=k.flatten(1),value=v,positions=positions,**kwargs)
                out=out.reshape(len(hidden_states),-1).contiguous()
                return linear(out,a.o_proj.weight,a.o_proj.bias)
            a.forward=attention_forward
            g=layer.mlp.gate;old_gate=g.forward
            def gate(x,original=old_gate,g=g):
                return linear(x,g.weight,g.bias) if decoding() else original(x)
            g.forward=gate
        from atom.models.mellum_rocm_precision import MellumPrecisionNorm
        for module in model.modules():
            if isinstance(module,MellumPrecisionNorm):
                old_norm=module.forward
                def norm(x,residual=None,x_scale=None,m=module,original=old_norm):
                    if not decoding():return original(x,residual,x_scale)
                    if x_scale is not None:raise ValueError('Quantized norm unsupported')
                    counts['norm_script_calls']+=1
                    out,values=norm_rows(x,m.weight,float(m.eps),residual)
                    return out if residual is None else (out,values)
                module.forward=norm
        old_logits=model.compute_logits
        def logits(hidden):
            if not decoding():return old_logits(hidden)
            assert get_forward_context().attn_metadata.max_seqlen_q==1
            return linear(hidden,model.lm_head.weight,model.lm_head.bias)
        model.compute_logits=logits
    model.opt_points_metadata=dict(attention=attention,moe=moe,dispatch=dispatch,counts=counts,baseline=model.parallel_metadata)
    return model.opt_points_metadata
