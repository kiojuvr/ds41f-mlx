"""Normal CLI dump with Phase4 guard, WITHOUT control adapter/tool substitutions.
No followup, Agent or provider request. Does not authorize native execution.
"""
import hashlib,json,os,subprocess,tempfile
from pathlib import Path
from profile import make_profile,CLI,OUT
with tempfile.TemporaryDirectory(prefix='phase4-guard-compose-only-') as tmp:
 t=Path(tmp);w=make_profile(t,controlled=False)
 env=dict(os.environ,DSH_HOME=str(t/'home'),DSH_AGENTS_HOME=str(t/'agents'),PHASE4_WORKSPACE=str(w),DS41F_LOCAL_API_KEY='compose-only');env.pop('NODE_OPTIONS',None)
 c=subprocess.run(['node',str(CLI),'--profile','headless','--dump-config'],env=env,cwd=w,capture_output=True,text=True,timeout=60)
 assert c.returncode==0,c.stderr
 assert 'phase4-control' not in c.stdout and 'phase4-budget' in c.stdout
 script="const fs=require('fs');const y=require(process.env.HOME+'/.local/share/deepseek-harness/node_modules/yaml');const t=y.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(t.filter(r=>['system-prompt','llm-deepseek','compaction-basic','tool-fs','phase4-budget','sandbox-policy','session-log-deepseek','llm-retry','permission','agent-loop'].includes(r.id))));"
 j=subprocess.run(['node','-e',script],input=c.stdout,text=True,capture_output=True,check=True)
 (OUT/'native-configuration.json').write_text(json.dumps(dict(kind='NORMAL_CLI_GUARD_COMPOSITION_WITHOUT_CONTROLS',exit=c.returncode,composed_sha256=hashlib.sha256(c.stdout.encode()).hexdigest(),rows=json.loads(j.stdout),weights_loaded=0,agent_started=False),indent=2)+'\n')
 (OUT/'acceptance-guard.patch.yml').write_text('- insert:\n    - id: phase4-budget\n      name: '+str(OUT/'guard.mjs')+'\n')
 print('PASS: official normal composition with guard only, no control adapter, no Agent')
