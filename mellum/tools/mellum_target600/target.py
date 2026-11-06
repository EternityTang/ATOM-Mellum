"""Use the actual decode shape to choose sparse-workload or grouped operators."""
import grouped as fused_experts
import qkr
import adapter,step_cache,batched_sdpa
import exact,gathered,kernels
from atom.utils.forward_context import get_forward_context
from atom.models import mellum_rocm_precision as precision
MODES=dict(step_cache.MODES,uniform=('paged_grouped',True,'native'),
           post=('mixed_grouped',True,'native'),packed=('mixed_grouped',True,'native'),adaptive=('mixed_grouped',True,'native'),sparse=('mixed_grouped',True,'native'))
def install(model,select):
    step_cache.MODES.update(MODES)
    grouped=adapter.grouped_experts
    batched=batched_sdpa.batched
    base_norm=precision.MellumPrecisionNorm.forward
    base_rotate=precision.rotate
    expert_count=model.config.num_experts
    top_k=model.config.num_experts_per_tok
    def start(_model,args):
        ctx=get_forward_context()
        active=select()
        decode=not ctx.context.is_prefill and not ctx.context.is_dummy_run and ctx.attn_metadata.max_seqlen_q==1
        ctx.target_sparse=decode and (active=='sparse' or (active in ('adaptive','post','packed') and args[0].numel()*top_k<expert_count))
        ctx.target_fuse=decode and active in ('uniform','adaptive','post','packed') and not ctx.target_sparse
        ctx.target_post=ctx.target_fuse and active in ('post','packed')
        ctx.target_qkr=ctx.target_fuse and active=='packed'
    model.register_forward_pre_hook(start)
    def choose():
        active=select()
        if active in ('adaptive','post','packed'):
            return 'sparse' if get_forward_context().target_sparse else 'uniform'
        return active
    def experts(*args):
        if get_forward_context().target_sparse:return exact.experts(*args)
        if get_forward_context().target_post:return fused_experts.experts(*args)
        return grouped(*args)
    def attend(*args,**kw):
        if get_forward_context().target_sparse:return gathered.batched(*args,**kw)
        return batched(*args,**kw)
    def norm(layer,x,residual=None,x_scale=None):
        if get_forward_context().target_fuse and x_scale is None and x.is_contiguous() and (residual is None or residual.is_contiguous()):
            return kernels.norm(x,residual,layer.weight,layer.eps)
        return base_norm(layer,x,residual,x_scale)
    def rotate(q,k,c,s):
        if get_forward_context().target_fuse and all(x.is_contiguous() for x in (q,k,c,s)):
            return kernels.rotate(q,c,s),kernels.rotate(k,c,s)
        return base_rotate(q,k,c,s)
    base_attention=precision.MellumPrecisionAttention.forward
    def attention(layer,positions,hidden_states,**kwargs):
        if get_forward_context().target_qkr:return qkr.attn(layer,positions,hidden_states)
        return base_attention(layer,positions,hidden_states,**kwargs)
    precision.MellumPrecisionAttention.forward=attention
    precision.MellumPrecisionNorm.forward=norm
    precision.rotate=rotate
    adapter.grouped_experts=experts
    batched_sdpa.batched=attend
    step_cache.install(model,choose)
