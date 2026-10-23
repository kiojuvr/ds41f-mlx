// Offline guard-only checks; no tool execution, renderer, or model calls.
import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,rmSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {apply} from '/Volumes/SDXC-512/ds41f-mlx/tools/phase3_harness_guard.mjs';
const temp=mkdtempSync(join(tmpdir(),'phase3-guard-test-'));
const command='/tmp/ds41f-phase2-renderer/bin/python /Volumes/SDXC-512/ds41f-mlx/tools/phase3_render.py';
process.env.PHASE3_WORKSPACE=temp;
process.env.PHASE3_POLICY=join(temp,'policy.json');
writeFileSync(process.env.PHASE3_POLICY,JSON.stringify({command,tools:30,deadline_ms:Date.now()+60000,audit:join(temp,'audit.jsonl')}));
let guard;
apply({tools:{guard(fn){guard=fn;}}});
let count=0;
function check(name,args,denied){const r=guard({name,arguments:args,callId:'offline-'+(++count)});assert.equal(typeof r==='string',denied);}
try {
 check('write',{file_path:'index.html',content:'x'},false);
 check('edit',{file_path:'index.html'},false);
 check('read',{file_path:'index.html'},false);
 for(const name of ['read','write','edit','read_image']) check(name,{file_path:'phase3-stage/artifact-00.html',content:'x'},true);
 for(const name of ['read','write','edit']) check(name,{file_path:'../outside.html',content:'x'},true);
 check('write',{file_path:'index.html',content:'x'.repeat(65537)},true);
 check('bash',{command},false);
 check('bash',{command:command+'; id'},true);
 check('bash',{command:'ls'},true);
 check('bash',{command,workdir:'/'},true);
 check('bash',{command,run_in_background:true},true);
 check('bash',{command,timeoutMs:26000},true);
 check('bash',{command,sandbox_permissions:'danger-full-access'},true);
 check('read_image',{file_path:'render-99.png'},true);
 check('read_image',{file_path:'/etc/passwd'},true);
 check('unknown',{},true);
 console.log(JSON.stringify({checks:count,passed:true,no_effects_executed:true,private_stage_denied:true}));
} finally {rmSync(temp,{recursive:true,force:true});}
