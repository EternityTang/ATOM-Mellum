from pathlib import Path
import psutil,subprocess,time
w=Path('/workspace')
for name in ['run0','reference']:
    pidfile=w/f'logs/{name}.pid'
    if not pidfile.exists(): continue
    try:
        p=psutil.Process(int(pidfile.read_text()))
        procs=p.children(recursive=True)+[p]
        for proc in reversed(procs):
            try: proc.terminate()
            except psutil.NoSuchProcess: pass
        _,alive=psutil.wait_procs(procs,timeout=3)
        for proc in alive:
            try: proc.kill()
            except psutil.NoSuchProcess: pass
    except psutil.NoSuchProcess: pass
log=w/'logs/run0.log'
if log.exists(): log.rename(w/f'logs/run0-{time.time_ns()}.log')
f=open(log,'w')
p=subprocess.Popen([str(w/'run0.sh')],cwd=w,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(w/'logs/run0.pid').write_text(str(p.pid))
print('Started',p.pid)
