"""Paired, same-setting HumanEval. No score is borrowed from HumanEval+."""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

W = Path('/workspace')
MODEL = '/data/models/Mellum2.1-12B-A2.5B-Thinking'
WORKER = '''import json,sys,resource,builtins
globals()['__builtins__']=dict(vars(builtins))
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(8,8))
data=json.load(sys.stdin)
exec(data['harness'],globals())
sys.modules['socket']=None
result=[]
unsafe_execute(data['problem'],data['completion'],5.0,result)
print(json.dumps(result))
'''


def tasks():
    return [json.loads(line) for line in Path('/data/datasets/HumanEval.jsonl').read_text().splitlines()]


def run_test(problem, completion):
    payload = {'harness': (W / 'reference/human_eval/execution.py').read_text(), 'problem': problem, 'completion': completion}
    with tempfile.TemporaryDirectory(prefix='paired-humaneval-') as directory:
        os.chown(directory, 65534, 65534)
        try:
            proc = subprocess.run([sys.executable, '-I', '-c', WORKER], input=json.dumps(payload), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory, user=65534, group=65534, extra_groups=(), timeout=12)
            if proc.returncode:
                return {'passed': False, 'result': f'worker_error_{proc.returncode}: {proc.stderr[-1000:]}'}
            result = json.loads(proc.stdout)
            return {'passed': result == ['passed'], 'result': result[0] if result else 'empty_result'}
        except subprocess.TimeoutExpired:
            return {'passed': False, 'result': 'worker_timed_out'}


def extract(text, entry):
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    elif '<think>' in text:
        return None, 'unfinished_thinking'
    for marker in ['<|im_end|>', '<|endoftext|>']:
        text = text.replace(marker, '')
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, flags=re.DOTALL)
    code = '\n\n'.join(blocks).strip() if blocks else text.strip()
    try:
        tree = ast.parse(code)
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry for node in tree.body):
            return None, 'entry_function_missing'
    except SyntaxError as exc:
        return None, f'syntax_error: {exc}'
    return code, None


def prompt(tokenizer, problem):
    message = 'Complete the following Python function. Return the complete function implementation.\n\n' + problem['prompt']
    return tokenizer.apply_chat_template([{'role': 'user', 'content': message}], tokenize=False, add_generation_prompt=True)


def generate_samples(backend, max_tokens):
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = None
    if backend == 'reference':
        sys.path.insert(0, '/workspace/reference')
        from mellum_reference import ReferenceMellum
        from eval_reference_batch import generate
        torch.set_num_threads(8)
        model = ReferenceMellum(MODEL)
    output = W / f'results/humaneval_{backend}_samples.jsonl'
    done = {json.loads(line)['task_id'] for line in output.read_text().splitlines()} if output.exists() else set()
    for i, problem in enumerate(tasks()):
        if problem['task_id'] in done:
            continue
        text = prompt(tokenizer, problem)
        begin = time.time()
        if backend == 'reference':
            ids = generate(model, [tokenizer.encode(text)], max_tokens, eos_id=tokenizer.eos_token_id)[0]
            result_text = tokenizer.decode(ids)
        else:
            request = urllib.request.Request('http://127.0.0.1:8000/v1/completions', data=json.dumps({'model': MODEL, 'prompt': text, 'max_tokens': max_tokens, 'temperature': 0, 'ignore_eos': False}).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=3600) as response:
                response = json.load(response)
            ids, result_text = response['token_ids'], response['choices'][0]['text']
        row = {'task_id': problem['task_id'], 'backend': backend, 'max_tokens': max_tokens, 'thinking': True, 'greedy': True, 'prompt': text, 'token_ids': ids, 'text': result_text, 'truncated': len(ids) == max_tokens and ids[-1] != tokenizer.eos_token_id, 'elapsed_s': time.time() - begin}
        with output.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'backend': backend, 'completed': i + 1, 'task_id': row['task_id'], 'output_tokens': len(ids), 'truncated': row['truncated'], 'elapsed_s': row['elapsed_s']}), flush=True)


def selftest():
    results = []
    for problem in tasks():
        result = run_test(problem, problem['canonical_solution'])
        results.append(dict(task_id=problem['task_id'], **result))
    (W / 'results/humaneval_harness_selftest.json').write_text(json.dumps(results, indent=2))
    assert len(results) == 164 and all(r['passed'] for r in results), {'failed': sum(not r['passed'] for r in results), 'examples': [r for r in results if not r['passed']][:3]}
    print('Official harness: all 164 canonical solutions passed', flush=True)


def evaluate():
    problems = {r['task_id']: r for r in tasks()}
    summary = {'settings': {'dataset': 'original HumanEval, 164 tasks', 'thinking': True, 'temperature': 0, 'max_tokens': 4096, 'execution_timeout_s': 5, 'reference': 'local source-attributed HF Mellum backport, single sequence', 'comparison': 'percentage points; not HumanEval+ model-card replication'}}
    all_results = {}
    for backend in ['reference', 'atom']:
        samples = [json.loads(line) for line in (W / f'results/humaneval_{backend}_samples.jsonl').read_text().splitlines()]
        assert len(samples) == 164 and len({r['task_id'] for r in samples}) == 164
        results = []
        for sample in samples:
            problem = problems[sample['task_id']]
            code, error = extract(sample['text'], problem['entry_point'])
            if error:
                result = {'passed': False, 'result': error}
            else:
                # Retain imports from the original prompt, then execute the full returned implementation.
                imports = [ast.unparse(node) for node in ast.parse(problem['prompt']).body if isinstance(node, (ast.Import, ast.ImportFrom))]
                local_problem = dict(problem, prompt='\n'.join(imports) + '\n')
                result = run_test(local_problem, code)
            results.append({'task_id': sample['task_id'], 'truncated': sample['truncated'], **result})
        summary[backend] = {'count': 164, 'passed': sum(r['passed'] for r in results), 'pass_at_1': sum(r['passed'] for r in results) / 164, 'truncated': sum(r['truncated'] for r in results), 'failures': {reason: sum(r['result'] == reason for r in results) for reason in sorted({r['result'] for r in results if not r['passed']})}}
        all_results[backend] = results
    difference = abs(summary['reference']['pass_at_1'] - summary['atom']['pass_at_1']) * 100
    summary['difference_percentage_points'] = difference
    summary['bf16_difference_within_2pp'] = difference <= 2
    (W / 'results/humaneval_paired.json').write_text(json.dumps({'summary': summary, 'results': all_results}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    assert summary['bf16_difference_within_2pp']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['selftest', 'reference', 'atom', 'evaluate'])
    args = parser.parse_args()
    if args.mode == 'selftest':
        selftest()
    elif args.mode == 'evaluate':
        evaluate()
    else:
        generate_samples(args.mode, 4096)
