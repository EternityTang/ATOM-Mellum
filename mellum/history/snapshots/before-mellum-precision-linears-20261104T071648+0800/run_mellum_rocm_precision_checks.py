"""GPU0-only validation, immutable measurement inputs, guaranteed service restore."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import urllib.request

W = Path('/workspace')
STAMP = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%dT%H%M%S%z')
OUT = W / 'results' / ('mellum_rocm_precision_' + STAMP)
OUT.mkdir()
STATE = dict(status='preparing',directory=str(OUT),gpu=0,driver_pid=os.getpid())


def save():
    (OUT/'control.json').write_text(json.dumps(STATE,indent=2)+'\n')


def frozen():
    files = [p for p in (W/'ATOM/atom').rglob('*.py')]
    files += [W/'run0.sh', W/'datasets/correctness_200.jsonl',
        W/'results/atom_reuse_mtp_off_eager_200.json',
        W/'results/mellum_naive_baseline/quick10_selection.json',
        W/'tools/measure_mellum_rocm_precision.py',
        Path('/app/aiter-test/aiter/ops/rmsnorm.py'),
        Path('/app/aiter-test/aiter/rotary_embedding.py'),
        Path('/app/aiter-test/csrc/include/opus/opus.hpp')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def archive(label):
    p=Path('/root/.cache/atom')
    if p.exists():p.rename(p.with_name('atom.backup-rocm-precision-'+STAMP+'-'+label))
    p.mkdir(parents=True,exist_ok=True)


def generate(port):
    request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(dict(
        model='/data/models/Mellum2.1-12B-A2.5B-Thinking',prompt='def add(a, b):\n',
        max_tokens=8,temperature=0,ignore_eos=True)).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=90) as response:result=json.load(response)
    assert result['usage']['completion_tokens']==8
    return result


def main():
    snapshot=frozen()
    (OUT/'source_sha256.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    STATE['head']=subprocess.check_output(['git','-C',str(W/'ATOM'),'rev-parse','HEAD'],text=True).strip()
    save()
    stopped=False
    worker=None
    try:
        pid=int((W/'logs/run0.pid').read_text())
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        assert b'atom.entrypoints.openai_server' in cmd and b'8000' in cmd and os.getpgid(pid)==pid
        os.killpg(pid,signal.SIGTERM)
        stopped=True
        time.sleep(4)
        try:os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass
        archive('worker')
        env=dict(os.environ,HIP_VISIBLE_DEVICES='0',PYTHONPATH='/workspace/ATOM',AITER_LOG_LEVEL='WARNING',
            ATOM_MELLUM_NAIVE_BASELINE='0',ATOM_MELLUM_TORCH_EXPERTS='1',ATOM_MELLUM_ROCM_PRECISION='1',
            ATOM_MELLUM_ROCM_PRECISION_PROFILE='correctness',ATOM_MELLUM_ROUTE_WEIGHT_BF16='1',ATOM_MELLUM_RNE_NORM='0',
            ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION='0',ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION='0',
            MELLUM_ROCM_OUT=str(OUT),MELLUM_ROCM_AUDIT='1',MELLUM_VARIANT_OUT=str(OUT),MELLUM_VARIANT_ROLE='precision')
        with (OUT/'worker.log').open('w') as log:
            worker=subprocess.Popen(['/opt/venv/bin/python',str(W/'tools/measure_mellum_rocm_precision.py')],
                cwd=W/'ATOM',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        STATE.update(status='running',worker_pid=worker.pid)
        save()
        assert worker.wait(timeout=2400)==0,'Worker failed; inspect log and debug guide before retry'
        assert json.loads((OUT/'validation.json').read_text())['status']=='complete'
        STATE['measurement_status']='complete'
    except BaseException:
        STATE.update(measurement_status='failed',error=traceback.format_exc())
    finally:
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid,signal.SIGTERM)
        try:
            if stopped:
                archive('restore')
                env=dict(os.environ,PATH='/opt/venv/bin:'+os.environ.get('PATH',''),STEP0_GPU='0',SERVER_PORT='8000',
                    ATOM_MELLUM_NAIVE_BASELINE='0',ATOM_MELLUM_TORCH_EXPERTS='0',ATOM_MELLUM_ROCM_PRECISION='0',
                    ATOM_MELLUM_ROUTE_WEIGHT_BF16='0',ATOM_MELLUM_RNE_NORM='0')
                with (W/'logs/run0.log').open('a') as log:
                    server=subprocess.Popen(['bash',str(W/'run0.sh')],cwd=W/'ATOM',env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (W/'logs/run0.pid').write_text(str(server.pid)+'\n')
                STATE.update(status='restoring',server_restart_pid=server.pid)
                save()
                ready=False
                for _ in range(100):
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=2):pass
                        ready=True
                        break
                    except Exception:
                        if server.poll() is not None:break
                        time.sleep(2)
                assert ready
            STATE['services']={str(port):generate(port) for port in [8000,8001]}
            STATE['gpu_memory']=subprocess.check_output(['rocm-smi','--showmeminfo','vram'],text=True)
            STATE['source_unchanged']=frozen()==snapshot
            assert STATE['source_unchanged']
            final_head=subprocess.check_output(['git','-C',str(W/'ATOM'),'rev-parse','HEAD'],text=True).strip()
            assert final_head==STATE['head']
            old=W/'results/mellum_torch_experts_route_bf16_200_20261103T215005+0800/status.json'
            STATE['bf16_full200_status']=json.loads(old.read_text())['status']
            assert STATE['bf16_full200_status']=='stopped_by_user'
        except BaseException:
            STATE['restore_error']=traceback.format_exc()
        STATE['status']='complete' if STATE.get('measurement_status')=='complete' and not STATE.get('restore_error') else 'failed'
        save()


if __name__=='__main__':
    main()
