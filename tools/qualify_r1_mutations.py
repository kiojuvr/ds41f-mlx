"""M42 rejection proof: run the ordinary verifier on source-mutated candidates."""
import hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'artifacts/m42';O.mkdir(exist_ok=True)
mutations=[('stateful-stop-admitted','serving/request_policy.py','    if user_stop_present(body):','    if False:'),('quiescence-work-admitted','runtime/mtp_lifecycle.py','if self.new_verify_cycles or self.new_proposals or self.history_replay or self.full_cache_repack:','if False:'),('noncanonical-fence-admitted','mtp_profile.py',"rb'[1-9][0-9]{0,19}'","rb'[0-9][0-9]{0,19}'")]
rows=[]
for name,path,old,new in mutations:
    with tempfile.TemporaryDirectory(prefix='ds41f-r1-candidate-') as tmp:
        root=Path(tmp)
        shutil.copytree(R/'ds41f_mlx',root/'ds41f_mlx',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(R/'reference',root/'reference',ignore=shutil.ignore_patterns('__pycache__'))
        subprocess.run(['git','init','-q',str(root)],check=True)
        subprocess.run(['git','-C',str(root),'-c','user.name=R1','-c','user.email=r1@local','commit','--allow-empty','-qm','candidate'],check=True)
        source=root/'ds41f_mlx'/path;s=source.read_text();assert s.count(old)==1
        source.write_text(s.replace(old,new))
        out=O/(name+'.json');cmd=[sys.executable,'-m','ds41f_mlx.reference','--output',str(out)]
        proc=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=180)
        (O/(name+'.log')).write_text(proc.stdout+proc.stderr)
        receipt=json.loads(out.read_text())
        assert proc.returncode==1 and receipt['status']=='FAIL'
        failed=next(g for g in receipt['gates'] if g['returncode'])
        log=Path(failed['log']).read_text()
        assert 'AssertionError' in log or 'Failed: DID NOT RAISE' in log
        rows.append(dict(name=name,source=path,mutation={'before':old,'after':new},decision='REJECTED_SEMANTIC_VIOLATION',failed_gate=failed['name'],receipt_sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
(O/'mutations.json').write_text(json.dumps(dict(schema='ds41f.m42.mutations.v1',status='PASS',cases=rows),indent=2)+'\n')
