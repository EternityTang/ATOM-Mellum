"""Restore ATOM components under per-request numerical isolation.

The original Mellum/Qwen3-MoE backbone remains the numerical forward.
The compatibility attention backend preserves HF eager operation boundaries;
the native-SWA stages explicitly exercise the existing ATOM accelerated backend.
"""
import copy
from types import MethodType,SimpleNamespace
import torch
from torch.nn import functional as F
from atom.utils.forward_context import get_forward_context,AttentionMetaData,AttnState
from atom.models import mellum_naive_ops as ops
from mellum_eager_precision import install as install_eager

STAGES=('backbone','framework_eager','native_swa','native_swa_prefill','native_swa_decode','native_norm')

def install(model,stage='framework_eager'):
    if stage not in STAGES:raise ValueError(stage)
    cfg=model.atom_config
    if cfg.enable_chunked_prefill:raise ValueError('Per-request full-prefill reference requires chunking off')
    if cfg.enable_tbo:raise ValueError('Request isolation requires TBO off')
    before={n:id(p) for n,p in model.named_parameters()}
    install_eager(model)
    counts={'backbone':0,'decoder_layers':0,'moe_blocks':0,
            'framework_attention':0,'eager_swa':0,'eager_full':0,
            'native_swa_prefill':0,'native_swa_decode':0,'outer_batches':0,'max_outer_n':0}
    def count(name):
        def hook(m,a,o):
            if not get_forward_context().context.is_dummy_run:counts[name]+=1
        return hook
    model.model.register_forward_hook(count('backbone'))
    for layer in model.model.layers:
        layer.register_forward_hook(count('decoder_layers'))
        layer.mlp.register_forward_hook(count('moe_blocks'))

    if stage!='backbone':
        for layer in model.model.layers:
            attn=layer.self_attn;impl=attn.attn.impl
            expected=attn.window if attn.window is not None else -1
            if impl.sliding_window!=expected:raise ValueError('Per-layer SWA binding mismatch')
            if impl.rotary_emb is not None or impl.q_norm is not None or impl.k_norm is not None:
                raise ValueError('RoPE/QK norm must be applied exactly once before the attention component')
            attn.settings=dict(attn.settings,attention='native')
            original_native=impl.forward_impl_server_mode
            proxy=SimpleNamespace(num_heads=impl.num_heads,num_kv_heads=impl.num_kv_heads,
                                  head_dim=impl.head_dim,layer_num=impl.layer_num,
                                  window=attn.window,scale=impl.scale)
            def backend(self,q,k,v,position=None,q_scale=None,qkv=None,
                        original_native=original_native,proxy=proxy):
                ctx=get_forward_context()
                if ctx.context.is_dummy_run:return torch.zeros_like(q)
                counts['framework_attention']+=1
                prefill=ctx.context.is_prefill
                native=(proxy.window is not None and
                        (stage=='native_swa' or
                         (stage=='native_swa_prefill' and prefill) or
                         (stage=='native_swa_decode' and not prefill)))
                if native:
                    counts['native_swa_prefill' if prefill else 'native_swa_decode']+=1
                    return original_native(q,k,v,position,q_scale,qkv)
                if q_scale is not None or qkv is not None:raise ValueError('Unexpected quantized/fused attention input')
                counts['eager_swa' if proxy.window is not None else 'eager_full']+=1
                query=q.view(-1,proxy.num_heads,proxy.head_dim)
                key=k.view(-1,proxy.num_kv_heads,proxy.head_dim)
                value=v.view(-1,proxy.num_kv_heads,proxy.head_dim)
                if stage.startswith('native_swa') and not prefill:
                    # The legacy eager adapter initializes decode groups at layer 0.
                    # Native SWA can bypass that layer, so build from this request's
                    # metadata explicitly rather than reuse missing/stale groups.
                    from batched_sdpa import build_groups
                    from mellum_eager_precision.install import eager_batched
                    out=eager_batched(proxy,query,key,value,build_groups(ctx.attn_metadata))
                else:
                    out=ops.NaiveSDPAAttention.forward(proxy,query,key,value,position)
                return out.reshape(len(q),-1)
            impl.forward_impl_server_mode=MethodType(backend,impl)

    if stage=='native_norm':
        from atom.models.mellum_rocm_precision import MellumPrecisionNorm
        from atom.model_ops.layernorm import RMSNorm
        for parent in list(model.modules()):
            for name,child in list(parent.named_children()):
                if isinstance(child,MellumPrecisionNorm):
                    norm=RMSNorm(child.weight.numel(),eps=child.eps)
                    norm.weight=child.weight
                    setattr(parent,name,norm)

    # Re-enter the same Module so its existing per-forward hooks initialize
    # RoPE/cache/mask state for each request, then run the inherited backbone.
    original_forward=model.forward
    model.request_isolation_active=False
    def forward(input_ids,positions,intermediate_tensors=None,inputs_embeds=None,**kwargs):
        ctx=get_forward_context()
        if model.request_isolation_active or ctx.context.is_dummy_run:
            return original_forward(input_ids,positions,intermediate_tensors,inputs_embeds,**kwargs)
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError('Only one-GPU token-ID inference is supported')
        meta=ctx.attn_metadata;outer_context=ctx.context
        cu=meta.cu_seqlens_q.tolist();lengths=meta.context_lens.tolist()
        if len(cu)!=len(lengths)+1 or cu[-1]!=len(input_ids):raise ValueError('Unexpected padded metadata')
        counts['outer_batches']+=1;counts['max_outer_n']=max(counts['max_outer_n'],len(lengths))
        saved=dict(ctx.__dict__);outputs=[]
        try:
            model.request_isolation_active=True
            for seq,(start,end,length) in enumerate(zip(cu,cu[1:],lengths)):
                qlen=end-start
                if qlen<=0 or (outer_context.is_prefill and qlen!=length):
                    raise ValueError('Empty/chunked/prefix-cached request is unsupported')
                qcu=meta.cu_seqlens_q.new_tensor([0,qlen])
                kcu=meta.cu_seqlens_q.new_tensor([0,length])
                single=AttentionMetaData(
                    cu_seqlens_q=qcu,cu_seqlens_k=kcu,
                    max_seqlen_q=qlen,min_seqlen_q=qlen,max_seqlen_k=length,
                    slot_mapping=meta.slot_mapping[start:end],
                    context_lens=meta.context_lens[seq:seq+1],
                    block_tables=None if meta.block_tables is None else meta.block_tables[seq:seq+1],
                    dropout_p=0.0,has_cached=False,total_kv=length,
                    state=AttnState.PREFILL_NATIVE if outer_context.is_prefill else AttnState.DECODE)
                sub=copy.copy(outer_context)
                sub.positions=positions[start:end];sub.input_ids=input_ids[start:end]
                sub.batch_size=1;sub.graph_bs=qlen
                ctx.context=sub;ctx.attn_metadata=single
                outputs.append(model(input_ids[start:end],positions[start:end],**kwargs))
        finally:
            model.request_isolation_active=False
            ctx.__dict__.clear();ctx.__dict__.update(saved)
        return torch.cat(outputs,0)
    model.forward=forward
    def compute_logits(hidden):
        ctx=get_forward_context();cu=ctx.attn_metadata.cu_seqlens_q.tolist()
        with torch._C.DisableTorchFunction():
            return torch.cat([F.linear(hidden[end-1:end],model.lm_head.weight,model.lm_head.bias) for end in cu[1:]],0)
    model.compute_logits=compute_logits
    after={n:id(p) for n,p in model.named_parameters()}
    if before!=after:raise RuntimeError('Component reconnection changed checkpoint Parameters')
    model.reuse_stage_metadata=dict(stage=stage,torch=torch.__version__,
        backbone_forward=model.model.forward.__qualname__,
        decoder_forward=model.model.layers[0].forward.__qualname__,
        moe_forward=model.model.layers[0].mlp.forward.__qualname__,
        attention_components=[dict(layer=i,kind=model.config.layer_types[i],
            component=type(l.self_attn.attn).__name__,window=l.self_attn.attn.impl.sliding_window)
            for i,l in enumerate(model.model.layers)],
        parameter_identity_preserved=True,counts=counts)
    return model.reuse_stage_metadata
