"""Validate the reference, then run the frozen 200-prompt token comparison."""
import ast
import concurrent.futures
import inspect
import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/reference')
import mellum_reference as ref
from eval_reference_batch import generate

W = Path('/workspace')
MODEL = '/data/models/Mellum2.1-12B-A2.5B-Thinking'


def save(name, data):
    p = W / 'results' / name
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(p)


def validate(model, tokenizer):
    upstream = ast.parse((W / 'reference/modeling_mellum.py').read_text())
    checked = []
    for node in upstream.body:
        name = getattr(node, 'name', '')
        if name in vars(ref) and (name.startswith('Mellum') or name in ['rotate_half', 'repeat_kv', 'eager_attention_forward', 'apply_rotary_pos_emb']):
            if name in ['MellumModel', 'MellumPreTrainedModel', 'MellumForCausalLM']:
                continue
            node.decorator_list = []
            actual = ast.parse(inspect.getsource(getattr(ref, name))).body[0]
            actual.decorator_list = []
            assert ast.dump(node) == ast.dump(actual), name
            checked.append(name)
    assert len(checked) >= 10
    cache_checks = []
    source = tokenizer.encode('def identity(x):\n    return x\n\n' * 150)
    for length, split in [(73, 31), (1025, 1000), (1033, 1020)]:
        ids = torch.tensor([source[:length]], device='cuda')
        full, _ = model.step(ids)
        _, cache = model.step(ids[:, :split])
        cached, _ = model.step(ids[:, split:], cache)
        item = {'length': length, 'split': split, 'max_abs_logit_error': (full.float() - cached.float()).abs().max().item(), 'same_argmax': full.argmax(-1).item() == cached.argmax(-1).item()}
        assert item['same_argmax'] and torch.allclose(full.float(), cached.float(), atol=0.5, rtol=0.03), item
        cache_checks.append(item)
    smoke = json.loads((W / 'results/smoke_reference.json').read_text())
    batched = [generate(model, [r['input_ids']], 16)[0] for r in smoke]
    save('reference_batch_diagnostic.json', {'batch': batched, 'single': [r['reference_ids'] for r in smoke], 'cache_checks': cache_checks})
    assert all(a == b['reference_ids'] for a, b in zip(batched, smoke)), 'Padded batch driver differs from single reference'
    result = {'status': 'passed_single_sequence_only', 'upstream_components_ast_equal': checked, 'cache_consistency': cache_checks, 'generation_driver_vs_single_smoke': True, 'padded_batch_driver': 'not used: smoke output differed; diagnostic retained', 'caveat': 'Local HF source backport; validation does not establish independent official certification.'}
    save('reference_validation.json', result)
    print(json.dumps(result), flush=True)


def atom_request(row):
    request = urllib.request.Request('http://127.0.0.1:8000/v1/completions', data=json.dumps({'model': MODEL, 'prompt': row['prompt'], 'max_tokens': 64, 'temperature': 0, 'ignore_eos': True}).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)['token_ids']


def main():
    torch.set_num_threads(8)
    tokenizer = ref.AutoTokenizer.from_pretrained(MODEL)
    model = ref.ReferenceMellum(MODEL)
    validate(model, tokenizer)
    rows = [json.loads(line) for line in (W / 'datasets/correctness_200.jsonl').read_text().splitlines()]
    for row in rows:
        row['prompt'] = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
        row['input_ids'] = tokenizer.encode(row['prompt'])
    results = []
    for start in range(0, len(rows), 1):
        batch = rows[start:start + 1]
        begin = time.time()
        reference = [model.greedy(torch.tensor([r['input_ids']], device='cuda'), 64) for r in batch]
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            actual = list(pool.map(atom_request, batch))
        for row, a, b in zip(batch, reference, actual):
            assert len(a) == len(b) == 64
            results.append({'id': row['id'], 'input_tokens': len(row['input_ids']), 'reference_ids': a, 'atom_ids': b, 'matching_tokens': sum(x == y for x, y in zip(a, b)), 'exact_match': a == b})
        summary = {'count': len(results), 'expected_count': 200, 'matching_tokens': sum(r['matching_tokens'] for r in results), 'total_tokens': len(results) * 64, 'token_match_rate': sum(r['matching_tokens'] for r in results) / (len(results) * 64), 'exact_sequence_match_rate': sum(r['exact_match'] for r in results) / len(results), 'batch_elapsed_s': time.time() - begin}
        save('public200_correctness.json', {'summary': summary, 'results': results})
        print(json.dumps(summary), flush=True)
    assert len(results) == 200 and summary['token_match_rate'] >= .90


if __name__ == '__main__':
    main()
