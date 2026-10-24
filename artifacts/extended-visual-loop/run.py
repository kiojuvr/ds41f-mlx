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
    runtime=harness=renderer=None; receipt={'status':'STARTING','conversations':0,'automatic_retry':False}
    env=dict(os.environ,DSH_HOME=str(TEMP/'home'),DSH_AGENTS_HOME=str(TEMP/'agents'),PHASE3_MAX_RENDERS='12',PHASE3_WORKSPACE=str((TEMP/'workspace').resolve()),PHASE3_POLICY=str(TEMP/'policy.json'),PHASE3_EVIDENCE=str(OUT),PHASE3_RUNTIME_WORKSPACE=str(TEMP/'runtime'),DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',DS41F_LOCAL_API_KEY='local-only-placeholder',PYTHONDONTWRITEBYTECODE='1')
    total_start=time.monotonic()
    try:
        assert json.loads((OUT/'pre-execution.json').read_text())['status']=='PASS'
        for name, value in json.loads((OUT/'identities.json').read_text())['packages'].items():
            assert value == json.loads((ROOT/'artifacts/visual-feedback-loop/phase-3-evidence/visual-loop/identities.json').read_text())['packages'][name]
        with socket.socket() as s: s.bind(('127.0.0.1',18083))
        # Configuration preflight without model generation.
        c=subprocess.run(['node',str(CLI),'--profile','headless','--dump-config'],cwd=TEMP/'workspace',env=env,capture_output=True,text=True,timeout=60)
        (OUT/'composed-config.yml').write_text(c.stdout); (OUT/'config-stderr.log').write_text(c.stderr)
        assert c.returncode==0, c.stderr
        with (OUT/'runtime.log').open('w') as log:
            runtime_env={k:v for k,v in env.items() if k!='DS41F_LOCAL_API_KEY'}
            runtime=subprocess.Popen(['/private/tmp/ds41f-visual-capacity-test/bin/python',str(OUT/'runtime.py')],cwd=TEMP/'runtime',env=runtime_env,stdout=log,stderr=log,start_new_session=True)
        receipt['runtime_pid']=runtime.pid; save('execution.json',receipt)
        start=time.monotonic()
        while True:
            assert runtime.poll() is None,'Runtime exited before ready'
            assert not (OUT/'SAFETY_STOP.json').exists(), 'startup safety stop'
            try:
                ready=health()
                if ready.get('model_ready') and ready.get('status')=='ready': break
            except Exception: pass
            if time.monotonic()-start>1200: raise TimeoutError('startup budget')
            time.sleep(2)
        assert not (OUT/'SAFETY_STOP.json').exists(), 'startup safety stop'
        receipt.update(status='RUNNING',runtime_health_before=ready,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),conversations=1)
        render_evidence=OUT/'renders'
        render_evidence.mkdir()
        with (OUT/'renderer-supervisor.log').open('w') as log:
            renderer=subprocess.Popen(['/tmp/ds41f-phase2-renderer/bin/python',str(ROOT/'tools/phase3_render_handoff.py'),'supervise','--workspace',str(TEMP/'workspace'),'--evidence',str(render_evidence),'--stop',str(TEMP/'render-stop'),'--timeout','5400'],env=env,stdout=log,stderr=log,start_new_session=True)
        receipt['renderer_supervisor_pid']=renderer.pid
        env['NODE_OPTIONS']='--import='+str(OUT/'observer.mjs')
        command=['node',str(CLI),'--profile','headless','--json']
        start=time.monotonic()
        with (OUT/'harness-events.jsonl').open('w') as stdout, (OUT/'harness-stderr.log').open('w') as stderr, (TEMP/'prompt.txt').open() as prompt:
            harness=subprocess.Popen(command,cwd=TEMP/'workspace',env=env,stdin=prompt,stdout=stdout,stderr=stderr,start_new_session=True)
        receipt.update(harness_pid=harness.pid,command=command); save('execution.json',receipt)
        while harness.poll() is None:
            if time.monotonic()-total_start>5400: raise TimeoutError('total task budget')
            if (OUT/'SAFETY_STOP.json').exists(): raise RuntimeError('resource safety stop; see receipt')
            if runtime.poll() is not None: raise RuntimeError('Runtime exited')
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
                if not settled and elapsed>600: raise TimeoutError('generation wall budget')
            traces=sorted(OUT.glob('trace-*.raw.json.gz'))
            import gzip
            parsed=[json.loads(gzip.decompress(p.read_bytes())) for p in traces]
            if sum(t['response']['usage']['output_tokens'] for t in parsed)>18000: raise RuntimeError('aggregate output cap')
            for t in parsed:
                assert t['prompt_replay']==t['full_cache_repack']==0, 'forbidden replay/repack'
                assert t['cache_published'] and t['authority_connection']['disposition']=='settled', 'missing settlement'
            memory=subprocess.run(['ps','-o','pid=,rss=','-p',str(runtime.pid)+','+str(harness.pid)],capture_output=True,text=True)
            with (OUT/'memory.jsonl').open('a') as f: f.write(json.dumps(dict(at=time.time(),ps_rss_kib=memory.stdout))+'\n')
            time.sleep(.2)
        receipt.update(status='EXITED',exit_code=harness.returncode,elapsed_s=time.monotonic()-start)
    except BaseException as e:
        receipt.update(status='STOPPED',error=repr(e))
    finally:
        retire(harness)
        (TEMP/'render-stop').touch()
        if renderer:
            try: renderer.wait(timeout=30)
            except subprocess.TimeoutExpired: retire(renderer)
        # Graceful server exit waits for serving, then synchronously flushes.
        retire(runtime)
        receipt['harness_exit']=harness.returncode if harness else None
        receipt['runtime_exit']=runtime.returncode if runtime else None
        save('execution.json',receipt)
        copies=[]
        archive=OUT/'workspace-final'
        shutil.copytree(TEMP/'workspace',archive)
        for p in archive.rglob('*'):
            if p.is_file():
                original=TEMP/'workspace'/p.relative_to(archive)
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                assert digest==hashlib.sha256(original.read_bytes()).hexdigest()
                copies.append(dict(file=str(p.relative_to(archive)),sha256=digest))
        with socket.socket() as s: free=s.connect_ex(('127.0.0.1',18083))!=0
        save('final-state.json',dict(runtime_exit=receipt['runtime_exit'],harness_exit=receipt['harness_exit'],port_18083_free=free,workspace_files=copies,renderer_exit=renderer.returncode if renderer else None,temporary_root=str(TEMP),removed=True))
        shutil.rmtree(TEMP)
if __name__=='__main__': main()
