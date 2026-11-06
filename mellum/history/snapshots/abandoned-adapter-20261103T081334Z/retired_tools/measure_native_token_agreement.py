"""Measure the unchanged ATOM adapter against installed, native HF Mellum."""
import concurrent.futures
import hashlib
import importlib.util
import json
import time
import traceback
import urllib.request
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

W = Path('/workspace')
MODEL = Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
OUT = W / 'results/native_token_agreement.json'
SCRIPT = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(document):
    temp = OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(document, ensure_ascii=False, indent=2))
    temp.replace(OUT)


def atom_request(row):
    body = dict(model=str(MODEL), prompt=row['prompt'], max_tokens=64,
                temperature=0.0, ignore_eos=True)
    req = urllib.request.Request(
        'http://127.0.0.1:8000/v1/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)['token_ids']


def main():
    assert not OUT.exists(), f'Refusing to overwrite existing evidence: {OUT}'
    torch.set_num_threads(8)
    torch.manual_seed(0)
    assert transformers.__version__ == '5.10.1', transformers.__version__
    import transformers.models.mellum.modeling_mellum as native_module
    sources = [W / 'ATOM/atom/models/mellum.py', W / 'ATOM/atom/models/mellum_ops.py',
               W / 'run0.sh', W / 'datasets/correctness_200.jsonl', SCRIPT,
               Path(native_module.__file__), MODEL / 'config.json',
               MODEL / 'tokenizer_config.json', MODEL / 'model.safetensors.index.json']
    frozen = {str(p): sha(p) for p in sources}
    old = json.loads((W / 'results/current_token_agreement.json').read_text())
    assert old['summary']['completed'] == 200
    previous = {r['id']: r for r in old['results']}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = [json.loads(line) for line in (W / 'datasets/correctness_200.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 200
    for row in rows:
        row['prompt'] = tokenizer.apply_chat_template(
            row['messages'], tokenize=False, add_generation_prompt=True)
        row['input_ids'] = tokenizer.encode(row['prompt'])
        assert row['input_ids'] == previous[row['id']]['input_ids'], row['id']
        assert sha_bytes(row['prompt'].encode()) == previous[row['id']]['input_sha256']
    urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=10).close()
    kwargs = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    if importlib.util.find_spec('accelerate'):
        kwargs['device_map'] = {'': 'cuda:0'}
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    if 'device_map' not in kwargs:
        model = model.cuda()
    model.eval()
    assert model.__class__.__module__ == 'transformers.models.mellum.modeling_mellum'
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=64, use_cache=True,
        eos_token_id=None, forced_eos_token_id=None,
        pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
        repetition_penalty=1.0, no_repeat_ngram_size=0)
    metadata = dict(
        source_sha256=frozen, reference='installed native Transformers MellumForCausalLM; no local backport',
        transformers_version=transformers.__version__, torch_version=torch.__version__,
        model_class=model.__class__.__name__, model_module=model.__class__.__module__,
        dtype=str(model.dtype), attention_implementation=model.config._attn_implementation,
        generation_config=generation.to_dict(), selection='same fixed 164 HumanEval + 36 MBPP prompts',
        prompt_inputs_identical_to_previous_run=True, temperature=0, ignore_eos=True,
        output_tokens_per_prompt=64, atom_concurrency=1, reference_concurrency=1,
        timing='separate GPUs concurrently; NOT a performance benchmark',
        previous_result_sha256=sha(W / 'results/current_token_agreement.json'))
    results = []
    started = time.time()
    summary = dict(status='running', completed=0, expected=200)
    save(dict(metadata=metadata, summary=summary, results=results))
    print(json.dumps(dict(status='started', **metadata)), flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for row in rows:
                before = time.time()
                future = executor.submit(atom_request, row)
                ids = torch.tensor([row['input_ids']], device=model.device)
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        generation_config=generation)
                reference = output[0, ids.shape[1]:].tolist()
                actual = future.result()
                assert len(reference) == len(actual) == 64, 'Incomplete request'
                matches = [a == b for a, b in zip(reference, actual)]
                prior = previous[row['id']]
                results.append(dict(
                    id=row['id'], input_tokens=len(row['input_ids']),
                    input_sha256=sha_bytes(row['prompt'].encode()), input_ids=row['input_ids'],
                    reference_ids=reference, atom_ids=actual,
                    matching_tokens=sum(matches), exact_match=all(matches),
                    matching_prefix_tokens=next((i for i, eq in enumerate(matches) if not eq), 64),
                    native_vs_old_reference_matching_tokens=sum(a == b for a, b in zip(reference, prior['reference_ids'])),
                    atom_vs_previous_matching_tokens=sum(a == b for a, b in zip(actual, prior['atom_ids']))))
                total = 64 * len(results)
                summary = dict(
                    status='complete' if len(results) == 200 else 'running', completed=len(results),
                    expected=200, matching_tokens=sum(r['matching_tokens'] for r in results),
                    total_tokens=total, exact_sequences=sum(r['exact_match'] for r in results),
                    elapsed_s=time.time() - started, latest_pair_s=time.time() - before,
                    native_vs_old_reference_matching_tokens=sum(r['native_vs_old_reference_matching_tokens'] for r in results),
                    atom_vs_previous_matching_tokens=sum(r['atom_vs_previous_matching_tokens'] for r in results))
                summary['token_match_rate'] = summary['matching_tokens'] / total
                summary['exact_sequence_match_rate'] = summary['exact_sequences'] / len(results)
                summary['by_source'] = {}
                for group in ['HumanEval', 'MBPP']:
                    subset = [r for r in results if r['id'].startswith(group + '/')]
                    if subset:
                        summary['by_source'][group] = dict(
                            count=len(subset), token_match_rate=sum(r['matching_tokens'] for r in subset) / (64 * len(subset)))
                assert all(sha(p) == frozen[str(p)] for p in sources), 'Frozen source changed'
                save(dict(metadata=metadata, summary=summary, results=results))
                print(json.dumps(summary), flush=True)
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        save(dict(metadata=metadata, summary=summary, results=results))
        raise
    report = '\n'.join([
        '# 原生 Transformers 5.10.1 与当前 ATOM 的 token 一致率', '',
        f'- 完成：{len(results)}/200，每条 greedy 生成 64 token，忽略 EOS。',
        f'- token 一致率：{summary["token_match_rate"]:.4%}（{summary["matching_tokens"]}/{summary["total_tokens"]}）。',
        f'- 整条 64 token 完全一致：{summary["exact_sequences"]}/200（{summary["exact_sequence_match_rate"]:.2%}）。',
        '- 参考：安装包原生 MellumForCausalLM，bf16、SDPA、单请求，使用官方 generate。',
        '- 与旧测量使用相同的 200 条 prompt，输入 token 和渲染文本哈希均逐条验证。',
        '- ATOM 代码未修改，全程冻结文件哈希；此结果不是 HumanEval pass@1 或性能成绩。',
        '- 完整 token 和源码哈希见 native_token_agreement.json。', ''])
    (W / 'results/native_token_agreement.md').write_text(report)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    main()
