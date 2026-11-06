import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, '/workspace/tools')
from evaluate_correctness import MODEL, atom_request
from transformers import AutoTokenizer

w = Path('/workspace')
tok = AutoTokenizer.from_pretrained(MODEL)
rows = {r['id']: r for r in [json.loads(line) for line in (w / 'datasets/correctness_200.jsonl').read_text().splitlines()]}
previous = json.loads((w / 'results/public200_before_norm_fix.json').read_text())['results']
for attempt in range(90):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2).close()
        break
    except Exception:
        time.sleep(1)
results = []
for old in previous[:8]:
    row = rows[old['id']]
    row['prompt'] = tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
    actual = atom_request(row)
    item = {'id': old['id'], 'before_matching_tokens': old['matching_tokens'], 'after_matching_tokens': sum(a == b for a, b in zip(old['reference_ids'], actual)), 'reference_ids': old['reference_ids'], 'atom_ids': actual}
    results.append(item)
    (w / 'results/norm_regression.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in item.items() if not k.endswith('_ids')}), flush=True)
