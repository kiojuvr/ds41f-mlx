// Actual installed Cordis middleware registration. Transaction arrays below are
// predicate inputs ONLY, never appended to an official Session or native proof.
import assert from 'node:assert/strict';
import {writeFileSync,readFileSync} from 'node:fs';
import {Gate,install,budgets,checkReceipt} from './guard.mjs';
const root=process.env.HOME+'/.local/share/deepseek-harness/node_modules/@deepseek-ai/';
const load=n=>import(root+n+'/lib/index.js');
const {Context}=await load('cordis');
const plugins=await Promise.all(['dsh-llm','dsh-session','dsh-session-projection','dsh-system-prompt','dsh-tools','dsh-agent','dsh-token-meter'].map(load));
const checks=[];const check=(name,fn)=>{fn();checks.push(name);};
const ordered=()=>[...'ABCDEFGHI','A'].map((label,i)=>({name:'read_image',callId:'id-'+i,arguments:{file_path:'image-'+label+'.png'}}));
const done=[{type:'compaction/start'},{type:'compaction/summary',seq:2},{surfaceOp:{op:'replace'},sourceEventSeqs:[2]},{type:'compaction/end',data:{}}];
check('A through I and new A distinct IDs; exactly ten',()=>{const g=new Gate();for(const [i,e] of ordered().entries()){if(i===4)g.offload();g.effect(e,'/fixture',()=>{});g.result(e.callId,false);}assert.equal(g.effects.length,10);assert.notEqual(g.effects[0].id,g.effects[9].id);assert.throws(()=>g.effect(ordered()[0],'/fixture',()=>{}));});
for(const [name,edit] of [['out of order',e=>e.arguments.file_path='image-B.png'],['unauthorized tool',e=>e.name='bash'],['unauthorized path',e=>e.arguments.file_path='/elsewhere/image-A.png'],['mutation/escalation',e=>e.arguments.sandbox_permissions='require_escalated']])check(name,()=>{const g=new Gate(),e=ordered()[0];edit(e);assert.throws(()=>g.effect(e,'/fixture',()=>{}));});
check('duplicate identity and uncompleted effect blocked',()=>{let g=new Gate();g.effect(ordered()[0],'/fixture',()=>{});assert.throws(()=>g.effect(ordered()[1],'/fixture',()=>{}));g=new Gate();g.effect(ordered()[0],'/fixture',()=>{});g.result('id-0',false);const e=ordered()[1];e.callId='id-0';assert.throws(()=>g.effect(e,'/fixture',()=>{}));});
check('all nineteen ordinary plus summary; allocation and HTTP distinct',()=>{const g=new Gate();for(let i=0;i<19;i++){g.attempt('ordinary',1200);g.httpStart('ordinary',1000,1200);}g.attempt('compaction',2048);g.httpStart('compaction',1000,2048);assert.equal(g.allocation,24848);assert.equal(g.http,20);assert.throws(()=>g.attempt('ordinary',1200));});
check('twenty-seven attempts: seven pre-HTTP refunds only',()=>{const g=new Gate();for(let i=0;i<7;i++){g.attempt('ordinary',1200);g.budgetRefusal({code:'IMAGE_OFFLOAD_REQUIRED',offloadImages:1});g.offload();}for(let i=0;i<19;i++)g.attempt('ordinary',1200);g.attempt('compaction',2048);assert.equal(g.attempts,27);assert.equal(g.ordinary,19);assert.equal(g.allocation,24848);assert.throws(()=>g.attempt('ordinary',1200));});
check('no refund of actual HTTP/incomplete attempt',()=>{const g=new Gate();g.attempt('ordinary',1200);g.httpStart('ordinary',100,1200);assert.throws(()=>g.budgetRefusal({code:'IMAGE_OFFLOAD_REQUIRED',offloadImages:1}));assert.equal(g.ordinary,1);assert.equal(g.http,1);});
check('second summary, wrong cap, eighth offload, body overflow',()=>{for(const action of [g=>{g.attempt('compaction',2048);g.attempt('compaction',2048);},g=>g.attempt('ordinary',2048),g=>{for(let i=0;i<8;i++)g.offload();},g=>g.httpStart('ordinary',16777217,1200)])assert.throws(()=>action(new Gate()));});
check('prepared-plus-output and image envelope before execution',()=>{new Gate().prepared(30720,2048,3);assert.throws(()=>new Gate().prepared(30721,2048,3));assert.throws(()=>new Gate().prepared(1000,1200,4));});
check('uncertain protocol/effect is sticky across further submission',()=>{for(const cause of ['incomplete SSE','native cancellation','uncertain external effect','resource refusal','failed persistence']){const g=new Gate();g.uncertainty(cause);assert.throws(()=>g.attempt('ordinary',1200));assert.throws(()=>g.effect(ordered()[0],'/fixture',()=>{}));}});
check('one call per response, no batch bypass of visual inspection turns',()=>{new Gate().assistant({content:[{type:'tool-call'}]});assert.throws(()=>new Gate().assistant({content:[{type:'tool-call'},{type:'tool-call'}]}));});
check('no extra reads during review/round1; missing offload blocks next read',()=>{assert.throws(()=>new Gate().effect(ordered()[0],'/fixture',()=>{},'review'));const g=new Gate({effects:ordered().slice(0,4).map(e=>({id:e.callId}))});assert.throws(()=>g.effect(ordered()[4],'/fixture',()=>{},'round2'));});
check('Supervisor freshness/storage/persistence receipts fail closed',()=>{const now=Date.now(),ok={at_ms:now,stop:null,session_bytes:1000,attachment_bytes:31800};checkReceipt(new Gate(),ok,'',now);for(const s of [null,{}, {...ok,at_ms:now-10001},{...ok,stop:'physical-footprint stop'},{...ok,session_bytes:67108865},{...ok,attachment_bytes:8388609}])assert.throws(()=>checkReceipt(new Gate(),s,'',now));assert.throws(()=>checkReceipt(new Gate(),ok,'round3',now));checkReceipt(new Gate(),{...ok,persistence_ok:true},'round3',now);});
const ctx=new Context();
try{
 for(const p of plugins)await ctx.plugin(p.default);
 let pressure=100,events=[],round='round1';const gate=new Gate(),callbacks=[];
 const original=ctx.tools.guard.bind(ctx.tools);ctx.tools.guard=fn=>{callbacks.push(fn);return original(fn);};
 ctx.tokenMeter.measure=()=>({totalTokens:pressure});
 install(ctx,{gate,workspace:'/fixture',round:()=>round});
 let listenerCalls=0,after=null;
 const engine=ctx.on('agent/pre-step',(_,next)=>{listenerCalls++;if(after)events=after;return next();});
 const run=()=>ctx.waterfall('agent/pre-step',{agent:{session:{snapshotEvents:()=>events}}},()=>Promise.resolve({kind:'enter'}));
 assert.equal((await run()).kind,'enter');checks.push('real Cordis first legitimate pre-step');
 assert.equal(callbacks[0](ordered()[0]),undefined);gate.result('id-0',false);assert.equal(gate.pending,null);checks.push('actual tools.guard registration; result predicate (official result event path in full-profile test)');
 // Failure after the official listener, before downstream ordinary admission.
 after=[{type:'compaction/start'},{type:'compaction/end',data:{error:'incomplete summary'}}];pressure=8200;gate.effects=ordered().slice(0,6).map((e,i)=>({label:'ABCDEF'[i],id:e.callId}));gate.offloads=3;
 assert.equal((await run()).kind,'reject');assert.equal(listenerCalls,2);assert.equal((await run()).kind,'reject');assert.equal(listenerCalls,2);checks.push('prepend before AND after listener; failed attempt cannot continue or reenter');
 engine();
}finally{await ctx.fiber.dispose();}
// Each independent case installs the same supported event registration. No fake
// successful compaction events are written to a Session.
for(const [label,ev,p,r,allow] of [
 ['initial automatic compaction eligible',[],8192,false,true],
 ['complete replacement permits continuation',done,3900,false,true],
 ['incomplete transaction',done.filter(x=>x.type!=='compaction/summary'),1000,false,false],
 ['unclosed transaction',[{type:'compaction/start'}],1000,false,false],
 ['second compaction',done.concat({type:'compaction/start'}),1000,false,false],
 ['residual high pressure',done,8192,false,false],
 ['Round3 high entry',done,4609,true,false],
 ['Round3 exact entry',done,4608,true,true]
]){
 const c=new Context();try{
  for(const pl of plugins)await c.plugin(pl.default);
  c.tokenMeter.measure=()=>({totalTokens:p});const g=new Gate({effects:[...'ABCDEF'].map((label,i)=>({label,id:'x'+i})),offloads:3});let calls=0;
  install(c,{gate:g,workspace:'/fixture',round:()=>r?'round3':''});
  c.on('agent/pre-step',(_,next)=>{calls++;return next();});
  const res=await c.waterfall('agent/pre-step',{agent:{session:{snapshotEvents:()=>ev}}},()=>Promise.resolve({kind:'enter'}));
  assert.equal(res.kind,allow?'enter':'reject');assert.equal(calls,allow?1:0);checks.push('real event path: '+label);
 }finally{await c.fiber.dispose();}
}
for(const code of ['CONTEXT_WINDOW_EXCEEDED','INVALID_REQUEST','TRANSPORT']){
 const c=new Context();try{
  for(const pl of plugins)await c.plugin(pl.default);const g=new Gate();install(c,{gate:g,workspace:'/fixture'});let calls=0;
  c.on('agent/request-error',(_,next)=>{calls++;return next();});
  const result=await c.waterfall('agent/request-error',{failure:{code}},()=>Promise.resolve({kind:'retry'}));
  assert.equal(result,undefined);assert.equal(calls,0);assert(g.stopped);checks.push('request-error prepended non-retry fence: '+code);
 }finally{await c.fiber.dispose();}
}
// The full-profile CLI test additionally drives the ACTUAL BasicCompaction and
// official image-offload handlers, rather than the predicate test listener above.
writeFileSync(new URL('gate-a-tests.json',import.meta.url),JSON.stringify({kind:'MODEL_FREE_INSTALLED_CORDIS_TESTS',pass:true,checks,weights_loaded:0,successful_transaction_is_predicate_input_only:true,shutdown:'all Cordis contexts disposed, no Runtime ownership'},null,2)+'\n');
console.log('PASS: '+checks.length+' focused guard checks; installed event registration and middleware ordering');
