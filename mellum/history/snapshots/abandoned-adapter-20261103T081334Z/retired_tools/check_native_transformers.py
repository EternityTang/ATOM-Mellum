"""Smoke-test installed native Transformers Mellum, without a local model backport."""
import importlib.util
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
path = '/data/models/Mellum2.1-12B-A2.5B-Thinking'
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
kwargs = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
if importlib.util.find_spec('accelerate'):
    kwargs['device_map'] = {'': 'cuda:0'}
model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
if 'device_map' not in kwargs:
    model = model.cuda()
model.eval()
assert model.__class__.__module__ == 'transformers.models.mellum.modeling_mellum'
results = []
for prompt in ['def add(a, b):\n    return', tokenizer.apply_chat_template([{'role': 'user', 'content': 'Write a Python function that adds two numbers.'}], tokenize=False, add_generation_prompt=True)]:
    batch = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    ids = output[0, batch.input_ids.shape[1]:].tolist()
    assert len(ids) > 0
    results.append({'prompt': prompt, 'input_ids': batch.input_ids[0].tolist(), 'output_ids': ids, 'text': tokenizer.decode(ids)})
result = {'transformers_version': transformers.__version__, 'torch_version': torch.__version__, 'model_class': model.__class__.__name__, 'model_module': model.__class__.__module__, 'dtype': str(model.dtype), 'device': str(model.device), 'results': results}
Path('/workspace/results/native_transformers_smoke.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
