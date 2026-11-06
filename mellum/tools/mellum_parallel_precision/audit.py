import json
import torch
from atom.utils.forward_context import get_forward_context

def install(model,path,step=1):
    ordinary=model.forward;state=dict(phase=None,done=False,calls=0);record={};diffs=[]
    def tensors(x,prefix=''):
        if isinstance(x,torch.Tensor):return {prefix:x.detach().cpu()}
        if isinstance(x,(tuple,list)):
            return {k:v for i,item in enumerate(x) for k,v in tensors(item,prefix+'/'+str(i)).items()}
        return {}
    def hook(name):
        def call(m,args,out):
            if not state['phase']:return
            vals={**tensors(args,'input'),**tensors(out,'output')}
            for key,value in vals.items():
                label=name+'/'+key
                if state['phase']=='serial':record.setdefault(label,[]).append(value)
                else:
                    ref=torch.cat(record[label])
                    assert value.shape==ref.shape,(label,value.shape,ref.shape)
                    delta=(value.float()-ref.float()).abs()
                    diffs.append(dict(name=label,equal=torch.equal(value,ref),max_abs=delta.max().item(),nonzero=int(torch.count_nonzero(delta)),numel=value.numel()))
        return call
    for i,layer in enumerate(model.model.layers):
        for name,module in [('input_norm',layer.input_layernorm),('qnorm',layer.self_attn.q_norm),('knorm',layer.self_attn.k_norm),('attention',layer.self_attn),('post_norm',layer.post_attention_layernorm),('gate',layer.mlp.gate),('moe',layer.mlp),('layer',layer)]:
            module.register_forward_hook(hook(f'{i:02d}/{name}'))
    model.model.norm.register_forward_hook(hook('final_norm'))
    def forward(*args,**kwargs):
        ctx=get_forward_context()
        if state['phase'] or state['done'] or ctx.context.is_dummy_run or ctx.context.is_prefill or len(ctx.attn_metadata.context_lens)!=10:
            return ordinary(*args,**kwargs)
        state['calls']+=1
        if state['calls']!=step:return ordinary(*args,**kwargs)
        state['phase']='serial'
        with torch.no_grad():ref=model.parallel_serial_forward(*args,**kwargs)
        state['phase']='batch'
        with torch.no_grad():out=ordinary(*args,**kwargs)
        state['phase']=None;state['done']=True
        path.write_text(json.dumps(dict(differences=diffs,hidden_equal=torch.equal(ref,out)),indent=2))
        # Only a one-step diagnostic; return the serial hidden result.
        return ref
    model.forward=forward
