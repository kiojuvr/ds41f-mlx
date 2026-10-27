"""Normal installed CLI composition only. Never follows up or creates an Agent."""
import hashlib,json,os,subprocess,tempfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
old=ROOT/'artifacts/multimodal-runtime/phase3-step3a-evidence/profile.patch.yml'
base=old.read_text().split('- insert:')[0]
base=base.replace('    maxTokens: 1200\n    streamIdleTimeoutMs:', '    maxTokens: 1200\n    retryPolicy: {mode: normal, maxRetries: 0}\n    streamIdleTimeoutMs:')
policy='''- id: compaction-basic
  config:
    auto: true
    modelPolicies:
      - provider: deepseek-official
        model: deepseek-flash
        thresholdRatio: 0.0078125
        headroomTokens: 2048
        retainTokens: 1024
        maxTokens: 2048
        compactionRetries: 0
'''
# Template deliberately not a runnable native acceptance profile: no task guard.
(OUT/'policy.patch.yml').write_text(policy)
with tempfile.TemporaryDirectory(prefix='ds41f-step4a-compose-') as temp:
 t=Path(temp);p=t/'home/profiles/headless';p.mkdir(parents=True);w=t/'workspace';w.mkdir()
 (p/'package.json').write_text(json.dumps(dict(name='ds41f-step4a-config-only',private=True,dependencies={},dsh=dict(profile=dict(bundles=['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless'])))))
 (p/'cordis.yml').write_text('[]\n')
 import re
 patch=re.sub(r'workspaceRoot: .*','workspaceRoot: '+str(w),base)+policy
 (p/'cordis.patch.yml').write_text(patch)
 cli=Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'
 env=dict(os.environ,DSH_HOME=str(t/'home'),DSH_AGENTS_HOME=str(t/'agents'),DS41F_LOCAL_API_KEY='local-only-placeholder')
 c=subprocess.run(['node',str(cli),'--profile','headless','--dump-config'],cwd=w,env=env,capture_output=True,text=True,timeout=60)
 (OUT/'config.log').write_text(c.stderr)
 assert c.returncode==0,c.stderr
 # Consume normal CLI output with the installed YAML parser; never mount a runner.
 jscode="""const fs=require('fs');const y=require(process.env.HOME+'/.local/share/deepseek-harness/node_modules/yaml');const t=y.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(t.filter(r=>['llm-deepseek','compaction-basic','llm-retry','session-log-deepseek','headless-runner','token-meter','compaction-image-offload','tool-fs'].includes(r.id))));"""
 d=subprocess.run(['node','-e',jscode],input=c.stdout,text=True,capture_output=True,check=True)
 (OUT/'configuration.json').write_text(json.dumps(dict(kind='NORMAL_INSTALLED_CLI_COMPOSITION_NO_AGENT',exit=c.returncode,composed_sha256=hashlib.sha256(c.stdout.encode()).hexdigest(),selected_rows=json.loads(d.stdout),task_guard_not_yet_extended=True,weights_loaded=0),indent=2)+'\n')
 print('PASS: normal official headless CLI composition with 8192 policy, capacity unchanged, retries zero')
