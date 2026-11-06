"""Serial, checkpointed validation workflow. Never makes git commits."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

W = Path('/workspace')
STATUS = W / 'results/step0_suite_status.json'
state = {'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'status': 'running', 'completed': [], 'git_commit': 'requires user approval; no commits performed'}


def status(phase, **kwargs):
    state.update(phase=phase, updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATUS)
    print(json.dumps(state), flush=True)


def run(phase, command, gpu=None, allow_failure=False):
    status(phase)
    env = dict(os.environ)
    if gpu is not None:
        env['HIP_VISIBLE_DEVICES'] = str(gpu)
    with (W / f'logs/suite_{phase}.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = proc.pid
        status(phase)
        code = proc.wait()
    if code and not allow_failure:
        raise RuntimeError(f'{phase} failed with exit code {code}; see logs/suite_{phase}.log')
    if code:
        state.setdefault('failed_benchmarks', {})[phase] = code
    state['completed'].append(phase)
    state.pop('child_pid', None)
    status(phase)


def summarize():
    lines = ['\n## Ordered validation run (automatically recorded; no git commit)\n']
    for filename in ['reference_validation.json', 'public200_correctness.json', 'humaneval_paired.json', 'benchmark_A.json', 'benchmark_B.json', 'benchmark_C.json']:
        path = W / 'results' / filename
        if path.exists():
            item = json.loads(path.read_text())
            lines.append(f'### {filename}\n\n```json\n{json.dumps(item.get("summary", item), ensure_ascii=False, indent=2)}\n```\n')
    (W / 'results/step0_suite_summary.md').write_text('\n'.join(lines))


def main():
    try:
        status('waiting_for_regression')
        pid = int((W / 'logs/norm_regression.pid').read_text())
        try:
            psutil.Process(pid).wait(timeout=1200)
        except psutil.NoSuchProcess:
            pass
        results = json.loads((W / 'results/norm_regression.json').read_text())
        assert len(results) == 8, 'Regression suite did not finish'
        rate = sum(r['after_matching_tokens'] for r in results) / (64 * len(results))
        assert rate >= .9, f'Regression matching rate still {rate:.3%}; fix before proceeding'
        run('correctness200', [sys.executable, 'tools/evaluate_correctness.py'], 1)
        run('boundaries', [sys.executable, 'tools/check_boundaries.py'], 1)
        selftest = json.loads((W / 'results/humaneval_harness_selftest.json').read_text())
        assert len(selftest) == 164 and all(r['passed'] for r in selftest)
        run('humaneval_reference', [sys.executable, 'tools/evaluate_humaneval.py', 'reference'], 1)
        run('humaneval_atom', [sys.executable, 'tools/evaluate_humaneval.py', 'atom'], 0)
        run('humaneval_compare', [sys.executable, 'tools/evaluate_humaneval.py', 'evaluate'])
        # Both reference inference and test execution have exited before timing.
        run('clean_server_restart', [sys.executable, 'tools/restart_step0.py'])
        import urllib.request
        for attempt in range(180):
            try:
                urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2).close()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError('Server readiness timeout')
        for scenario in ['A', 'B', 'C']:
            run('benchmark_' + scenario, [sys.executable, 'tools/benchmark_local.py', scenario, '--output', f'/workspace/results/benchmark_{scenario}.json'], allow_failure=True)
        summarize()
        status('complete', status='complete_with_benchmark_failures' if state.get('failed_benchmarks') else 'complete')
    except Exception as exc:
        summarize()
        status(state.get('phase', 'unknown'), status='failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
