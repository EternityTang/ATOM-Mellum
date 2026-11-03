from pathlib import Path
import shutil
w=Path('/workspace'); dest=w/'ATOM/submission';dest.mkdir(exist_ok=True)
for name in ['run0.sh','report.md','WORKLOG.md','README.step0.txt']:
    shutil.copy2(w/name,dest/name)
for name in ['tools','reference']:
    shutil.copytree(w/name,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
(dest/'evidence').mkdir(exist_ok=True)
for p in (w/'results').glob('*.json'):
    shutil.copy2(p,dest/'evidence'/p.name)
print('Submission sources and current evidence snapshotted into code repository')
