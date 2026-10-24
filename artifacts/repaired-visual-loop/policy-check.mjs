// Model-free checks of the actual guard and installed official sandbox.
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';
import {writeFileSync, realpathSync, symlinkSync, unlinkSync} from 'node:fs';
import {apply} from '../../tools/phase3_harness_guard.mjs';
const out = process.env.PHASE3_EVIDENCE, workspace = realpathSync(process.env.PHASE3_WORKSPACE);
let guard;
apply({tools:{guard: f => guard=f}});
const cases = [
 ['read source','read',{file_path:'index.html'},true],
 ['edit outside','edit',{file_path:'../outside.html',old_string:'a',new_string:'b'},false],
 ['read outside','read',{file_path:'/etc/passwd'},false],
 ['pwd','bash',{command:'pwd'},true],
 ['listing','bash',{command:'ls index.html'},true],
 ['size','bash',{command:'wc -lc index.html'},true],
 ['source range','bash',{command:"sed -n '1,20p' index.html"},true],
 ['huge range','bash',{command:"sed -n '1,9999p' index.html"},false],
 ...['rm index.html','cat /etc/passwd','curl https://example.com','sudo true','pwd &','pwd; rm index.html','sed -n \'1,2p\' index.html > /tmp/probe','$(pwd)','sleep 999'].map(command=>[command,'bash',{command},false]),
 ['wrong cwd','bash',{command:'pwd',workdir:'/tmp'},false],
 ['background','bash',{command:'pwd',run_in_background:true},false],
 ['escalate','bash',{command:'pwd',sandbox_permissions:'require_escalated'},false],
 ['timeout','bash',{command:'pwd',timeoutMs:26000},false],
 ['renderer','bash',{command:'/tmp/ds41f-phase2-renderer/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render_handoff.py request'},true],
];
const checks=cases.map(([label,name,args,expected])=>{
 const reason=guard({name,arguments:args,callId:'preflight-'+label});
 if (!reason !== expected) throw Error(label+': '+reason);
 return {label,allowed:!reason,reason};
});
symlinkSync('/etc/passwd',workspace+'/render-00.png');
try { if(!guard({name:'read_image',arguments:{file_path:'render-00.png'},callId:'symlink'})) throw Error('symlink admitted'); }
finally { unlinkSync(workspace+'/render-00.png'); }
const require=createRequire(process.env.HOME+'/.local/share/deepseek-harness/package.json');
const {Context}=await import(require.resolve('@deepseek-ai/cordis'));
const {LocalSandboxProvider}=await import(require.resolve('@deepseek-ai/dsh-sandbox-local'));
const provider=new LocalSandboxProvider(new Context(),{runnerCommand:[],runnerFailureSignatures:[],probeTimeoutMs:5000});
const policy={mode:'workspace-write',workspaceRoot:workspace};
const runs=[];
for(const [label,cmd,expected] of [
 ['inspection',"sed -n '1,20p' index.html",true],
 ['outside-write',`/tmp/ds41f-phase2-renderer/bin/python -c 'from pathlib import Path; Path("${out}/denial-probe.txt").write_text("must not write")'`,false],
]) {
 const confined=await provider.confine(['bash','-c',cmd],policy);
 const r=spawnSync(confined.argv[0],confined.argv.slice(1),{cwd:workspace,env:process.env,encoding:'utf8',timeout:25000});
 runs.push({label,confined,status:r.status,stdout:r.stdout,stderr:r.stderr});
 if ((r.status===0)!==expected) throw Error(label+' sandbox failure');
 if(!expected && !/Operation not permitted/.test(r.stderr)) throw Error('wrong denial');
}
provider.revokeAclGrants();
writeFileSync(out+'/policy-check.json',JSON.stringify({status:'PASS',checks,runs},null,2)+'\n');
