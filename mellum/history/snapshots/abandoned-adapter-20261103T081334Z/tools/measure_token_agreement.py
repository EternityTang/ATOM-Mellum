"""Measure the frozen adapter, without correctness thresholds or engine changes."""
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/reference')
from mellum_reference import ReferenceMellum, AutoTokenizer

W = Path('/workspace')
MODEL = '/data/models/Mellum2.1-12B-A2.5B-Thinking'
OUT = W / 'results/current_token_agreement.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atom_request(row):
    body = {'model': MODEL, 'prompt': row['prompt'], 'max_tokens': 64, 'temperature': 0.0, 'ignore_eos': True}
    req = urllib.request.Request('http://127.0.0.1:8000/v1/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)['token_ids']


def main():
    torch.set_num_threads(8)
    sources = [W / 'ATOM/atom/models/mellum.py', W / 'ATOM/atom/models/mellum_ops.py', W / 'reference/mellum_reference.py', W / 'run0.sh', W / 'datasets/correctness_200.jsonl']
    frozen = {str(p): sha(p) for p in sources}
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(line) for line in sources[-1].read_text().splitlines()]
    assert len(rows) == 200
    for row in rows:
        row['prompt'] = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
        row['input_ids'] = tokenizer.encode(row['prompt'])
    urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=10).close()
    model = ReferenceMellum(MODEL)
    results = []
    started = time.time()
    metadata = {'source_sha256': frozen, 'reference': 'local backport of HF Mellum; installed Transformers remains 5.2.0', 'selection': 'fixed 164 HumanEval + 36 MBPP prompts; no sample changes', 'temperature': 0, 'ignore_eos': True, 'output_tokens_per_prompt': 64, 'atom_concurrency': 1, 'reference_concurrency': 1, 'timing': 'two GPUs run concurrently; timings are NOT performance benchmark results'}
    print(json.dumps({'status': 'started', **metadata}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = [executor.submit(atom_request, row) for row in rows]
        for row, future in zip(rows, pending):
            before = time.time()
            reference = model.greedy(torch.tensor([row['input_ids']], device='cuda'), 64)
            actual = future.result()
            assert len(reference) == len(actual) == 64, 'Incomplete request; do not silently score missing tokens'
            matches = [a == b for a, b in zip(reference, actual)]
            prefix = next((i for i, equal in enumerate(matches) if not equal), 64)
            results.append({'id': row['id'], 'input_tokens': len(row['input_ids']), 'input_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(), 'input_ids': row['input_ids'], 'reference_ids': reference, 'atom_ids': actual, 'matching_tokens': sum(matches), 'exact_match': all(matches), 'matching_prefix_tokens': prefix})
            summary = {'status': 'complete' if len(results) == 200 else 'running', 'completed': len(results), 'expected': 200, 'matching_tokens': sum(r['matching_tokens'] for r in results), 'total_tokens': len(results) * 64, 'token_match_rate': sum(r['matching_tokens'] for r in results) / (len(results) * 64), 'exact_sequences': sum(r['exact_match'] for r in results), 'exact_sequence_match_rate': sum(r['exact_match'] for r in results) / len(results), 'elapsed_s': time.time() - started, 'latest_pair_s': time.time() - before}
            summary['by_source'] = {}
            for group in ['HumanEval', 'MBPP']:
                subset = [r for r in results if r['id'].startswith(group + '/')]
                if subset:
                    summary['by_source'][group] = {'count': len(subset), 'token_match_rate': sum(r['matching_tokens'] for r in subset) / (64 * len(subset))}
            assert all(sha(p) == frozen[str(p)] for p in sources), 'Frozen source changed during measurement'
            temp = OUT.with_suffix('.tmp')
            temp.write_text(json.dumps({'metadata': metadata, 'summary': summary, 'results': results}, ensure_ascii=False, indent=2))
            temp.replace(OUT)
            print(json.dumps(summary), flush=True)
    report = '\n'.join(['# 当前适配版本与 HF 参考的 token 一致率', '', f'- 完成：{len(results)}/200，每条 greedy 生成 64 token。', f'- token 一致率：{summary["token_match_rate"]:.4%}（{summary["matching_tokens"]}/{summary["total_tokens"]}）。', f'- 整条输出完全一致：{summary["exact_sequences"]}/200（{summary["exact_sequence_match_rate"]:.2%}）。', '- 参考为固定的 HF Mellum 源码回移实现，不是冻结包原生支持。', '- 本次只测量当前版本，不按 90% 门槛修改代码，不做性能测试，也不提交 Git。', '- 完整原始 token、输入和源码哈希保存在 current_token_agreement.json。', ''])
    (W / 'results/current_token_agreement.md').write_text(report)


if __name__ == '__main__':
    main()
