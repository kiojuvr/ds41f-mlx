"""Single official CLI conversation. Reuses Phase 3 bootstrap and supervisor boundaries.
No generation POSTs/tool dispatch; only health, owned processes and budget monitoring.
"""
import datetime, hashlib, json, os, shutil, signal, socket, subprocess, time, urllib.request
from pathlib import Path
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx')
OUT=Path(__file__).resolve().parent
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp'])
CLI=Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def health():
    with urllib.request.urlopen('http://127.0.0.1:18083/health',timeout=3) as r: return json.load(r)
def retire(p):
    if p is None or p.poll() is not None: return
    os.killpg(p.pid,signal.SIGTERM)
    try: p.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL); p.wait()
def main():
    runtime=harness=None; receipt={'status':'STARTING','conversations':0,'automatic_retry':False}
    env=dict(os.environ,DSH_HOME=str(TEMP/'home'),DSH_AGENTS_HOME=str(TEMP/'agents'),PHASE3_WORKSPACE=str((TEMP/'workspace').resolve()),PHASE3_EVIDENCE=str(OUT),PHASE3_RUNTIME_WORKSPACE=str(TEMP/'runtime'),DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',DS41F_LOCAL_API_KEY='local-only-placeholder',PYTHONDONTWRITEBYTECODE='1')
    try:
        with socket.socket() as s: s.bind(('127.0.0.1',18083))
        # Configuration preflight without model generation.
        c=subprocess.run(['node',str(CLI),'--profile','headless','--dump-config'],cwd=TEMP/'workspace',env=env,capture_output=True,text=True,timeout=60)
        (OUT/'composed-config.yml').write_text(c.stdout); (OUT/'config-stderr.log').write_text(c.stderr)
        assert c.returncode==0, c.stderr
        with (OUT/'runtime.log').open('w') as log:
            runtime_env={k:v for k,v in env.items() if k!='DS41F_LOCAL_API_KEY'}
            runtime=subprocess.Popen(['/tmp/ds41f-f0-runtime/bin/python',str(OUT/'runtime.py')],cwd=TEMP/'runtime',env=runtime_env,stdout=log,stderr=log,start_new_session=True)
        receipt['runtime_pid']=runtime.pid; save('execution.json',receipt)
        start=time.monotonic()
        while True:
            assert runtime.poll() is None,'Runtime exited before ready'
            try:
                ready=health()
                if ready.get('model_ready') and ready.get('status')=='ready': break
            except Exception: pass
            if time.monotonic()-start>1200: raise TimeoutError('startup budget')
            time.sleep(2)
        receipt.update(status='RUNNING',runtime_health_before=ready,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),conversations=1)
        env['NODE_OPTIONS']='--import='+str(OUT/'observer.mjs')
        command=['node',str(CLI),'--profile','headless','--json']
        start=time.monotonic()
        with (OUT/'harness-events.jsonl').open('w') as stdout, (OUT/'harness-stderr.log').open('w') as stderr, (TEMP/'prompt.txt').open() as prompt:
            harness=subprocess.Popen(command,cwd=TEMP/'workspace',env=env,stdin=prompt,stdout=stdout,stderr=stderr,start_new_session=True)
        receipt.update(harness_pid=harness.pid,command=command); save('execution.json',receipt)
        while harness.poll() is None:
            if time.monotonic()-start>900: raise TimeoutError('total task budget')
            rows=[]
            if (OUT/'wire.jsonl').exists():
                for line in (OUT/'wire.jsonl').read_text().splitlines():
                    try: rows.append(json.loads(line))
                    except json.JSONDecodeError: pass
            req=[r for r in rows if r.get('kind')=='request' and r.get('path')=='/v1/messages']
            if any(r.get('kind')=='budget_stop' for r in rows): raise RuntimeError('request budget fence reached')
            if req:
                last=req[-1]
                settled=any(r.get('kind')=='native_message_stop' and r['request']==last['request'] for r in rows)
                elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(last['at'])).total_seconds()
                if not settled and elapsed>180: raise TimeoutError('generation wall budget')
            time.sleep(.2)
        receipt.update(status='EXITED',exit_code=harness.returncode,elapsed_s=time.monotonic()-start)
    except BaseException as e:
        receipt.update(status='STOPPED',error=repr(e))
    finally:
        retire(harness); retire(runtime)
        receipt['harness_exit']=harness.returncode if harness else None
        receipt['runtime_exit']=runtime.returncode if runtime else None
        save('execution.json',receipt)
        copies=[]
        for r in json.loads((OUT/'images.json').read_text()):
            p=Path(r['copy']); copies.append(dict(label=r['label'],sha256=hashlib.sha256(p.read_bytes()).hexdigest(),unchanged=hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256']))
        with socket.socket() as s: free=s.connect_ex(('127.0.0.1',18083))!=0
        save('final-state.json',dict(runtime_exit=receipt['runtime_exit'],harness_exit=receipt['harness_exit'],port_18083_free=free,images=copies,temporary_root=str(TEMP),removed=True))
        shutil.rmtree(TEMP)
if __name__=='__main__': main()
