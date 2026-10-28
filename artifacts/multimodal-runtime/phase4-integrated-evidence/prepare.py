"""Task-local composition using unchanged qualified profile and guards; no inference."""
import hashlib,importlib.util,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx');OUT=Path(__file__).resolve().parent
R1=OUT.parent/'phase4-step4ar1-evidence';A=OUT.parent/'phase4-step4a-evidence'
spec=importlib.util.spec_from_file_location('qualified_profile',R1/'profile.py');profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
t=Path(tempfile.mkdtemp(prefix='ds41f-phase4-integrated-session-'));w=profile.make_profile(t,controlled=False)
(t/'agents').mkdir();(t/'runtime').mkdir()
for p in w.iterdir():p.chmod(0o444)
w.chmod(0o555)
p=t/'home/profiles/headless/cordis.patch.yml'
p.write_text(p.read_text()+'    - id: phase4-native-passive\n      name: '+str(OUT/'passive.mjs')+'\n')
(OUT/'location.json').write_text(json.dumps(dict(temp=str(t),installation='/tmp/ds41f-phase4-integrated-runtime',build='/tmp/ds41f-phase4-integrated-build'),indent=2)+'\n')
manifest=json.loads((A/'images.json').read_text())
for row in manifest:
 data=(w/('image-'+row['label']+'.png')).read_bytes()
 assert hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes']
(row if False else manifest) # No visual manifest enters workspace or task.
(OUT/'sources.json').write_text(json.dumps([dict(label=r['label'],sha256=r['sha256'],bytes=r['bytes'],copy=str(w/('image-'+r['label']+'.png'))) for r in manifest],indent=2)+'\n')
env=dict(os.environ,DSH_HOME=str(t/'home'),DSH_AGENTS_HOME=str(t/'agents'),PHASE4_WORKSPACE=str(w),PHASE4_EVIDENCE=str(OUT),DS41F_LOCAL_API_KEY='local-only-placeholder')
env.pop('NODE_OPTIONS',None)
c=subprocess.run(['node',str(profile.CLI),'--profile','headless','--dump-config'],cwd=w,env=env,capture_output=True,text=True,timeout=60)
(OUT/'composed-config.yml').write_text(c.stdout);(OUT/'composition.log').write_text(c.stderr)
assert c.returncode==0,c.stderr
assert 'phase4-control' not in c.stdout and 'phase4-native-passive' in c.stdout and str(R1/'guard.mjs') in c.stdout
assert 'maxImagesPerRequest: 3' in c.stdout and 'thresholdRatio: 0.0078125' in c.stdout
(OUT/'composition.json').write_text(json.dumps(dict(exit=c.returncode,composed_sha256=hashlib.sha256(c.stdout.encode()).hexdigest(),native=True,controlled=False,weights_loaded=0,workspace=str(w),guard=str(R1/'guard.mjs'),observer=str(OUT.parent/'phase4-step4br1-evidence/observer.mjs')),indent=2)+'\n')
paths=[A/n for n in ['budgets.json','images.json','tasks.json','policy.patch.yml']]+[R1/n for n in ['readiness.json','acceptance-guard.patch.yml','guard.mjs','safety.py']]
paths += [OUT.parent/'phase4-step4br1-evidence'/n for n in ['observer.mjs','terminal.mjs']]
(OUT/'frozen-inputs.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
print(t)
