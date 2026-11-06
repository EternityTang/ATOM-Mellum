"""Reuse ATOM's weights/backbone while retaining native HF eager arithmetic."""
import torch
from torch.nn import functional as F
import adapter,batched_sdpa,step_cache,exact
from atom.models import mellum_naive_ops as ops
from atom.models import mellum_rocm_precision as precision
from atom.utils.forward_context import get_forward_context
from . import safe_ops,repeat_ops

def eager_math(q,k,v,mask,scale):
    scores=torch.matmul(q,k.transpose(-2,-1))*scale
    if mask is not None:scores=scores+mask
    probs=F.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs,v)

def eager_one(q,k,v,qpos,kpos,window,scale):
    # Match HF DynamicSlidingWindowLayer: W-1 cached keys + the current
    # contiguous query chunk. Masking older keys is semantically equivalent,
    # but a different BF16 GEMM/softmax shape can change greedy tokens.
    shift=max(0,len(k)-len(q)-window+1) if window is not None else 0
    if shift:
        k,v,kpos=k[shift:],v[shift:],kpos[shift:]
    groups=q.shape[1]//k.shape[1]
    def repeat(x):
        x=x.transpose(0,1)[None]
        return x[:,:,None].expand(1,k.shape[1],groups,len(k),k.shape[-1]).reshape(1,q.shape[1],len(k),k.shape[-1])
    ctx=get_forward_context()
    key=(qpos.data_ptr(),len(qpos),len(kpos),window)
    mask=ctx.opt_masks.get(key)
    if mask is None:
        mask=torch.zeros((len(qpos),len(kpos)),dtype=q.dtype,device=q.device)
        mask.masked_fill_(~ops.visibility(qpos,kpos,window),torch.finfo(q.dtype).min)
        ctx.opt_masks[key]=mask
    out=eager_math(q.transpose(0,1)[None],repeat(k),repeat(v),mask[None,None],scale)
    return out[0].transpose(0,1)

def eager_batched(layer,q,k,v,groups,group_masked=False):
    ctx=get_forward_context()
    cache=ops.TorchPagedKV(ctx.kv_cache_data[f'layer_{layer.layer_num}'],layer.num_kv_heads,layer.head_dim)
    cache.write(ctx.attn_metadata.slot_mapping[:len(k)],k,v)
    out=torch.empty_like(q)
    for which,pos,blocks,offset in groups:
        length=len(pos);n=len(which);h=layer.num_kv_heads;d=layer.head_dim
        shift=max(0,length-layer.window) if layer.window is not None else 0
        keys,values=safe_ops.gather_heads(cache,blocks,length,shift,n,h)
        length-=shift
        keys,values=repeat_ops.repeat(keys,values,layer.num_heads//h)
        queries=q.contiguous() if len(groups)==1 else q[which]
        queries=queries.unsqueeze(2)
        mask=q.new_zeros((1,1,1,length)) if layer.window is not None else None
        result=eager_math(queries,keys,values,mask,layer.scale).reshape(n,layer.num_heads,d)
        if len(groups)==1:return result
        out.index_copy_(0,which,result)
    return out

def install(model):
    if not isinstance(model,precision.MellumRocmPrecisionForCausalLM) or model.precision_profile!='correctness':
        raise ValueError('Requires the Mellum correctness backbone')
    cfg=model.atom_config
    if torch.cuda.device_count()!=1 or cfg.speculative_config is not None or cfg.compilation_config.level!=0 or not cfg.enforce_eager:
        raise ValueError('Requires single GPU, eager, level 0, MTP disabled')
    if cfg.kv_cache_dtype!='bf16' or cfg.torch_dtype!=torch.bfloat16 or cfg.enable_prefix_caching:
        raise ValueError('Requires BF16 model/KV and prefix cache disabled')
    if not torch.__version__.startswith('2.10.'):
        raise ValueError('This entry is validated for the frozen Torch 2.10 runtime')
    rope=model.model.layers[0].self_attn.rope
    def rope_forward(x,positions):
        ctx=get_forward_context()
        if ctx.opt_rope_cache is not None:return ctx.opt_rope_cache
        result={}
        for kind in rope.kinds:
            inv=getattr(rope,kind+'_inv_freq').to(device=x.device,dtype=torch.float32)
            freqs=positions[:,None].float()*inv
            angles=torch.cat((freqs,freqs),dim=-1)
            result[kind]=((angles.cos()*rope.factors[kind]).to(x.dtype)[:,None,:],
                          (angles.sin()*rope.factors[kind]).to(x.dtype)[:,None,:])
        ctx.opt_rope_cache=result
        return result
    rope.forward=rope_forward
    # These existing binding names route to our explicit eager operations;
    # no SDPA is executed in either prefill or decode.
    ops.naive_sdpa=eager_one
    batched_sdpa.batched=eager_batched
    adapter.grouped_experts=exact.experts
    step_cache.MODES['hf_eager_precision']=('mixed_grouped',True,'native')
    step_cache.install(model,lambda:'hf_eager_precision')
    original_projected=exact.projected
    def projected(x,w,segments):
        # Preserve each expert's exact GEMM shape and row ordering.
        return safe_ops.projected(x,w,segments) if len(x)>=128 else original_projected(x,w,segments)
    exact.projected=projected
    def begin(*args):
        ctx=get_forward_context();ctx.opt_rope_cache=None;ctx.opt_masks={}
    model.register_forward_pre_hook(begin,prepend=True)
    original_forward=model.forward
    def forward(*args,**kwargs):
        # All factories in this supported forward path use explicit device
        # arguments or inherit their input tensor's device. Remove only the
        # per-operation default-device Python interception during this scope.
        with torch._C.DisableTorchFunction():return original_forward(*args,**kwargs)
    model.forward=forward
    model.eager_precision_metadata=dict(
        actual_attention='native HF eager operation boundaries',
        sliding_decode='window-only KV',
        torch=torch.__version__,expert_projection='per-expert F.linear; unchanged shapes',
        compiled_dispatch='TorchScript loop for >=128 routed rows',
        norm='original Torch',rope='original Torch rotation with shared FP32 frequencies',
        device_dispatch='scoped DisableTorchFunction; explicit device factories',
        kv_materialization='bitwise gather + bitwise contiguous head replication')
    return model.eager_precision_metadata
