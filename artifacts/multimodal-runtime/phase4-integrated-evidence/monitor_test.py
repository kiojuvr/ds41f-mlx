"""Focused actual-process regression, no weights or Harness Session."""
import json,os,subprocess,sys,time
from pathlib import Path
import monitor
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
try:
 time.sleep(.2)
 child=monitor.provenance(p.pid)
 assert child['ownership']=='task' and child['chain'][0]['pgid']==p.pid
 # Same PID/argv is insufficient: wrong root birth and unrelated root refuse.
 reused=monitor.provenance(p.pid,os.getpid(),monitor.ROOT_BIRTH+1)
 unknown=monitor.provenance(os.getppid())
 assert reused['ownership']=='unknown' and unknown['ownership']=='unknown'
 screen=monitor.process_screen()
 assert any(r['pid']==p.pid and r['ownership']=='task' for r in screen)
 sample=monitor.sample()
 assert sample['available']>64*monitor.G and sample['pressure']==1
 Path(__file__).with_name('monitor-tests.json').write_text(json.dumps(dict(pass_=True,child=child,pid_reuse=reused,unrelated=unknown,sample=sample),indent=2)+'\n')
 print('PASS: actual separate-process-group child, PID birth mismatch, unknown external ancestry and full host sampler')
finally:
 p.terminate();p.wait(timeout=10)
