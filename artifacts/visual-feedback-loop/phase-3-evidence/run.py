"""One official CLI conversation; external deadlines/observation, no LLM/tool dispatch."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path('/Volumes/SDXC-512/ds41f-mlx')
OUT = ROOT/'artifacts/visual-feedback-loop/phase-3-evidence'
TEMP = Path('/tmp/ds41f-phase3-harness.xQVspE')
CLI = Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'

def health():
    with urllib.request.urlopen('http://127.0.0.1:18083/health', timeout=3) as r:
        return json.load(r)

def main():
    ready = health()
    assert ready['model_ready'] and ready['status'] == 'ready', ready
    env = dict(os.environ, DSH_HOME=str(TEMP/'home'), DSH_AGENTS_HOME=str(TEMP/'agents'),
               PHASE3_WORKSPACE=str((TEMP/'workspace').resolve()), PHASE3_EVIDENCE=str(OUT),
               PHASE3_POLICY=str(TEMP/'policy.json'), DS41F_LOCAL_API_KEY='local-only-placeholder',
               NODE_OPTIONS='--import='+str(OUT/'observer.mjs'), PYTHONDONTWRITEBYTECODE='1')
    command = ['node', str(CLI), '--profile', 'headless', '--json']
    start = time.monotonic()
    receipt = dict(status='RUNNING', started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=command,
                   workspace=str(TEMP/'workspace'), runtime_health_before=ready, automatic_retry=False)
    def save():
        (OUT/'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    reason = None
    with (OUT/'harness-events.jsonl').open('w') as stdout, (OUT/'harness-stderr.log').open('w') as stderr:
        with (TEMP/'prompt.txt').open() as prompt:
            p = subprocess.Popen(command, cwd=TEMP/'workspace', env=env, stdin=prompt, stdout=stdout, stderr=stderr, start_new_session=True)
        receipt['harness_pid'] = p.pid
        save()
        try:
            while p.poll() is None:
                if time.monotonic()-start > 5400:
                    raise TimeoutError('total task budget')
                wire = OUT/'wire.jsonl'
                if wire.exists():
                    rows = []
                    for line in wire.read_text().splitlines():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # append in progress, not a protocol retry
                    requests = [r for r in rows if r.get('kind')=='request' and r.get('path')=='/v1/messages']
                    if len(requests) > 24:
                        raise RuntimeError('generation request budget exceeded; no restart')
                    previous_images = []
                    for r in requests:
                        images = [i['sha256'] for i in r.get('images', [])]
                        if images[:len(previous_images)] != previous_images:
                            raise RuntimeError('historical image projection changed')
                        previous_images = images
                        if r['body_bytes'] > 16777216:
                            raise RuntimeError('whole body budget exceeded')
                    if requests:
                        latest = requests[-1]
                        settled = any(r.get('kind')=='native_message_stop' and r['request']==latest['request'] for r in rows)
                        elapsed = (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(latest['at'])).total_seconds()
                        if not settled and elapsed > 600:
                            raise TimeoutError('generation wall deadline / observer settlement uncertainty')
                time.sleep(.5)
        except BaseException as e:
            reason = repr(e)
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        receipt.update(status='EXITED' if p.returncode==0 and not reason else 'STOPPED', exit_code=p.returncode, stop_reason=reason, elapsed_s=time.monotonic()-start)
        save()

if __name__ == '__main__':
    main()
