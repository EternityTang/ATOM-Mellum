"""Validate actual ATOM loading, native HF weights, layer outputs and paged KV.

Small engineering checks only; never resumes the paused public200 job.
Run on the reference GPU with HIP_VISIBLE_DEVICES=1 and PYTHONPATH=/workspace/ATOM.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from atom.config import Config, KVCacheTensor, set_current_atom_config
from atom.model_loader.loader import load_model
from atom.models.mellum import MellumForCausalLM
from atom.models.mellum_ops import MellumKVCacheView, causal_visibility, masked_mellum_sdpa
from atom.utils.forward_context import (
    AttentionMetaData, Context, set_forward_context, set_kv_cache_data,
)

W = Path('/workspace')
MODEL = '/data/models/Mellum2.1-12B-A2.5B-Thinking'
OUT = W / 'results/native_adapter_components.json'


def difference(left, right):
    left, right = left.float(), right.float()
    return dict(equal=torch.equal(left, right), max_abs=(left-right).abs().max().item(),
                relative_l2=((left-right).norm()/right.norm().clamp_min(1e-12)).item())


def allocate_cache(config, max_length, sequences=1):
    block_size = 16
    per_seq = math.ceil(max_length / block_size)
    count = per_seq * sequences
    # Reversed physical pages catch an accidental assumption that KV is contiguous.
    table = torch.arange(count-1, -1, -1, device='cuda', dtype=torch.int32).view(sequences, per_seq)
    h, d = config.num_key_value_heads, config.head_dim
    caches = {}
    for i in range(config.num_hidden_layers):
        caches[f'layer_{i}'] = KVCacheTensor(
            i, torch.zeros(count, h, d//8, block_size, 8, device='cuda', dtype=torch.bfloat16),
            torch.zeros(count, h, d, block_size, device='cuda', dtype=torch.bfloat16))
    return caches, table, block_size


def adapter_forward(model, config, chunks, history, caches, table, block_size, uncached=False):
    lengths = [past+len(chunk) for past, chunk in zip(history, chunks)]
    positions = torch.cat([torch.arange(past, length, device='cuda') for past, length in zip(history, lengths)])
    slots = torch.cat([
        table[i, torch.arange(past, length, device='cuda')//block_size].long()*block_size
        + torch.arange(past, length, device='cuda') % block_size
        for i, (past, length) in enumerate(zip(history, lengths))])
    cu = [0]
    for chunk in chunks:
        cu.append(cu[-1]+len(chunk))
    metadata = AttentionMetaData(
        cu_seqlens_q=torch.tensor(cu, device='cuda', dtype=torch.int32),
        context_lens=torch.tensor(lengths, device='cuda', dtype=torch.int32),
        slot_mapping=slots, block_tables=None if uncached else table,
        max_seqlen_q=max(map(len,chunks)), max_seqlen_k=max(lengths))
    set_kv_cache_data(caches)
    set_forward_context(metadata, config, Context(positions, is_prefill=True, batch_size=len(chunks)))
    ids = torch.tensor(sum(chunks, []), device='cuda')
    return model(ids, positions)


def reference_forward(model, ids):
    batch = torch.tensor([ids], device='cuda')
    pos = torch.arange(len(ids), device='cuda')
    masks = {kind: torch.where(causal_visibility(pos,pos,model.config.sliding_window if kind=='sliding_attention' else None),0.,float('-inf')).to(torch.bfloat16)[None,None]
             for kind in set(model.config.layer_types)}
    return model.model(input_ids=batch, attention_mask=masks, use_cache=False).last_hidden_state[0]


def check_cache_and_window():
    results = []
    generator = torch.Generator(device='cuda').manual_seed(17)
    for length in [1023,1024,1025,8191,8192,8193,16384]:
        block_size = 16
        blocks = math.ceil(length/block_size)
        table = torch.arange(blocks-1,-1,-1,device='cuda')
        key = torch.randn(length,4,128,device='cuda',dtype=torch.bfloat16,generator=generator)
        value = torch.randn_like(key)
        cache = KVCacheTensor(0,torch.zeros(blocks,4,16,16,8,device='cuda',dtype=torch.bfloat16),torch.zeros(blocks,4,128,16,device='cuda',dtype=torch.bfloat16))
        view = MellumKVCacheView(cache,4,128)
        p = torch.arange(length,device='cuda')
        slots = table[p//block_size]*block_size+p%block_size
        view.write(slots,key,value)
        got_k, got_v, got_p = view.read(table,length)
        assert torch.equal(got_k,key) and torch.equal(got_v,value) and torch.equal(got_p,p)
        q = torch.randn(1,32,128,device='cuda',dtype=torch.bfloat16,generator=generator)
        for window in [None,1024]:
            output = masked_mellum_sdpa(q,got_k,got_v,p[-1:],p,window,128**-.5)
            # Independent single-query FP32 attention oracle, with explicit slicing.
            begin = 0 if window is None else max(0,length-window)
            k = key[begin:].repeat_interleave(8,dim=1).transpose(0,1).float()
            v = value[begin:].repeat_interleave(8,dim=1).transpose(0,1).float()
            scores = (q[0].float()[:,None,:]*k).sum(-1)*128**-.5
            expected = (scores.softmax(-1)[...,None]*v).sum(1)[None]
            error = (output.float()-expected).abs().max().item()
            assert error < .01,(length,window,error)
            if window is not None:
                visible = causal_visibility(p[-1:],p,window)
                assert visible.sum().item()==min(length,1024)
                assert visible[0,max(0,length-1024)].item()
                if length>1024:assert not visible[0,length-1025].item()
            results.append(dict(length=length,window=window,max_abs_error_vs_fp32=error))
    return results


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    started = time.time()
    config = Config(model=MODEL,enforce_eager=True,enable_prefix_caching=False,
                    max_model_len=32768,kv_cache_dtype='bf16')
    set_current_atom_config(config)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    adapter = MellumForCausalLM(config).eval()
    torch.set_default_device('cpu')
    loaded = load_model(adapter,MODEL,config.hf_config)
    expected = set(dict(adapter.named_parameters()))
    assert loaded==expected,dict(missing=sorted(expected-loaded),extra=sorted(loaded-expected))
    reference = AutoModelForCausalLM.from_pretrained(
        MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',experts_implementation='eager',
        device_map={'':'cuda:0'},local_files_only=True).eval()
    actual_params = dict(adapter.named_parameters())
    reference_params = dict(reference.named_parameters())
    assert actual_params.keys()==reference_params.keys()
    mismatched = [name for name in actual_params if not torch.equal(actual_params[name],reference_params[name])]
    assert not mismatched,mismatched
    index=json.loads((Path(MODEL)/'model.safetensors.index.json').read_text())['weight_map']
    mappings=adapter.get_expert_mapping()
    for source in index:
        target=source
        for destination,fragment,_,_ in mappings:
            if fragment in source:
                target=source.replace(fragment,destination)
                break
        assert target in actual_params,(source,target)
    cache_checks=check_cache_and_window()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=tokenizer.encode('def add(a, b):\n    return a + b\n\n# Explain the result.\n')
    layer_adapter,layer_reference={},{}
    hooks=[]
    for i,(a,b) in enumerate(zip(adapter.model.layers,reference.model.layers)):
        hooks.append(a.register_forward_hook(lambda m,args,out,i=i:layer_adapter.update({i:out.detach().clone()})))
        hooks.append(b.register_forward_hook(lambda m,args,out,i=i:layer_reference.update({i:out.detach().reshape(-1,out.shape[-1]).clone()})))
    caches,table,block_size=allocate_cache(adapter.config,len(ids)+8)
    actual=adapter_forward(adapter,config,[ids],[0],caches,table,block_size,uncached=True)
    wanted=reference_forward(reference,ids)
    layers={str(i):difference(layer_adapter[i],layer_reference[i]) for i in layer_adapter}
    for hook in hooks:hook.remove()
    full=difference(actual,wanted)
    assert full['equal'],dict(full=full,layers=layers)
    split=len(ids)//2
    caches,table,block_size=allocate_cache(adapter.config,len(ids)+8)
    adapter_forward(adapter,config,[ids[:split]],[0],caches,table,block_size,uncached=True)
    cached=adapter_forward(adapter,config,[ids[split:]],[split],caches,table,block_size)
    cache_forward=difference(cached,wanted[split:])
    cached_logits=adapter.lm_head(cached[-1:])
    full_logits=reference.lm_head(wanted[-1:])
    assert cached_logits.argmax(-1).item()==full_logits.argmax(-1).item(),cache_forward
    # Distinct requests share an interleaved physical page pool.
    other=tokenizer.encode('The capital of France is')
    caches,table,block_size=allocate_cache(adapter.config,max(len(ids),len(other))+8,2)
    mixed=adapter_forward(adapter,config,[ids,other],[0,0],caches,table,block_size)
    other_wanted=reference_forward(reference,other)
    mixed_errors=[difference(mixed[:len(ids)],wanted),difference(mixed[len(ids):],other_wanted)]
    assert all(item['equal'] for item in mixed_errors),mixed_errors
    rotary=adapter.model.rotary_emb
    frequencies={name:value.clone() for name,value in rotary.native.named_buffers()}
    rotary.to(dtype=torch.bfloat16)
    assert all(value.dtype==torch.float32 and torch.equal(value,frequencies[name]) for name,value in rotary.native.named_buffers())
    result=dict(status='passed',transformers_version=transformers.__version__,torch_version=torch.__version__,
                reference_attention=reference.config._attn_implementation,
                reference_experts=reference.config._experts_implementation,
                checkpoint_tensors=len(index),loaded_runtime_parameters=len(loaded),
                all_parameters_bitwise_equal_to_native_hf=True,
                layer_differences=layers,full_forward=full,chunked_forward=cache_forward,
                chunked_last_token_argmax_equal=True,mixed_request_differences=mixed_errors,
                kv_window_checks=cache_checks,rope_buffers_preserve_fp32=True,
                elapsed_s=time.time()-started,
                scope='engineering component checks, not full public200/HumanEval acceptance')
    OUT.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
