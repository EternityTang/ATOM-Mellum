import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/reference')
from mellum_reference import ReferenceMellum, AutoTokenizer

w = Path('/workspace')
path = '/data/models/Mellum2.1-12B-A2.5B-Thinking'
tok = AutoTokenizer.from_pretrained(path)
row = [json.loads(line) for line in (w / 'datasets/correctness_200.jsonl').read_text().splitlines()][2]
prompt = tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
model = ReferenceMellum(path)
packets = {}
wanted = ['input_layernorm', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_norm', 'self_attn.k_norm', 'self_attn', 'self_attn.o_proj', 'post_attention_layernorm', 'mlp.gate', 'mlp', '']
names = {'layers.0' + ('.' + name if name else '') for name in wanted}


def hook(name):
    def save(module, args, kwargs, output):
        x = args[0] if args else kwargs['hidden_states']
        y = output[0] if isinstance(output, tuple) else output
        packet = {'input': x.detach().cpu(), 'output': y.detach().cpu()}
        if hasattr(module, 'weight'):
            packet['weight'] = module.weight.detach().cpu()
        packets['model.' + name] = packet
    return save


for name, module in model.named_modules():
    if name in names:
        module.register_forward_hook(hook(name), with_kwargs=True)
model.step(tok(prompt, return_tensors='pt').input_ids.cuda())
for attempt in range(90):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2).close()
        break
    except Exception:
        time.sleep(1)
request = urllib.request.Request('http://127.0.0.1:8000/v1/completions', data=json.dumps({'model': path, 'prompt': prompt, 'max_tokens': 1, 'temperature': 0, 'ignore_eos': True}).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=180) as response:
    print(response.read().decode(), flush=True)
results = []
for name, reference in packets.items():
    actual = torch.load(w / 'results/atom_layer_dump' / (name + '.pt'), weights_only=True)
    result = {'name': name}
    for field in ['input', 'output', 'weight']:
        if field not in reference or field not in actual:
            continue
        a, b = reference[field].flatten().float(), actual[field].flatten().float()
        assert len(a) == len(b), (name, field, a.shape, b.shape)
        result[field] = {'equal': torch.equal(a, b), 'different_fraction': (a != b).float().mean().item(), 'max_abs_error': (a - b).abs().max().item()}
    results.append(result)
    print(json.dumps(result), flush=True)
(w / 'results/first_layer_comparison.json').write_text(json.dumps(results, indent=2))
