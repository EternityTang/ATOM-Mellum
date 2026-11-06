import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, '/workspace/ATOM')
from aiter import rmsnorm2d_fwd, rmsnorm2d_fwd_with_add

torch.manual_seed(123)
torch.set_default_dtype(torch.bfloat16)
model = Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
results = []
for name, dim in [('model.layers.0.self_attn.q_norm.weight', 128), ('model.layers.0.input_layernorm.weight', 2304)]:
    with safe_open(model / index[name], framework='pt', device='cpu') as f:
        weight = f.get_tensor(name).cuda()
    for residual_flag in [False, True]:
        x = torch.randn((257, dim), device='cuda')
        residual = torch.randn_like(x) if residual_flag else None
        value = x + residual if residual_flag else x
        expected = value.float()
        expected = expected * torch.rsqrt(expected.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = weight * expected.to(value.dtype)
        if residual_flag:
            actual, residual_output = torch.empty_like(x), torch.empty_like(x)
            rmsnorm2d_fwd_with_add(actual, x, residual, residual_output, weight, 1e-6)
        else:
            actual = rmsnorm2d_fwd(x, weight, 1e-6)
        results.append({'name': name, 'residual': residual_flag, 'max_abs_error': (actual.float() - expected.float()).abs().max().item(), 'different_fraction': (actual != expected).float().mean().item()})
Path('/workspace/results/norm_probe.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
