import os,sys,json,subprocess
from pathlib import Path
import psutil
w=Path('/workspace')
try: psutil.Process(int((w/'logs/reference.pid').read_text())).wait(timeout=180)
except psutil.NoSuchProcess: pass
smoke=json.loads((w/'results/smoke_reference.json').read_text())
assert all(x['exact_ids_match'] for x in smoke), 'Smoke IDs differ or missing; stop before long validation'
env=dict(os.environ,HIP_VISIBLE_DEVICES='1')
with open(w/'logs/boundary_correctness.log','w') as f:
    subprocess.run([sys.executable,str(w/'tools/check_boundaries.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
# Reference has exited; diagnostic load check runs in isolation.
with open(w/'logs/benchmark_smoke.log','w') as f:
    subprocess.run([sys.executable,str(w/'tools/benchmark_local.py'),'A','--limit','8','--output-tokens','16','--output',str(w/'results/benchmark_smoke.json')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Boundary validation and non-official benchmark smoke completed',flush=True)
