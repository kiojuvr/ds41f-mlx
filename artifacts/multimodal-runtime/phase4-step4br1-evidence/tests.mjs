// Bounded model-free fixtures; real retained events, installed adapter/Agent/guard.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {Gate,install,manifest} from '../phase4-step4ar1-evidence/guard.mjs';
import {adapterIdentity} from './terminal.mjs';
const prior=new URL('../phase4-step4b-evidence/',import.meta.url);
const wire=readFileSync(new URL('wire.jsonl',prior),'utf8').trim().split('\n').map(JSON.parse);
const genuine=wire.filter(x=>x.kind==='sse_event').map(x=>x.event);
const encode=es=>es.map(e=>'event: '+e.type+'\ndata: '+JSON.stringify(e)+'\n\n').join('');
const text=[genuine[0],{type:'content_block_start',index:0,content_block:{type:'text',text:''}},{type:'content_block_delta',index:0,delta:{type:'text_delta',text:'Model-free ordinary text.'}},{type:'content_block_stop',index:0},{type:'message_delta',delta:{stop_reason:'end_turn'},usage:{output_tokens:5}},{type:'message_stop'}];
const home=process.env.HOME+'/.local/share/deepseek-harness/node_modules/@deepseek-ai/';
const load=n=>import(home+n+'/lib/index.js');
const {DeepSeekAdapter}=await load('dsh-llm-deepseek');
const full=JSON.parse(readFileSync(new URL('../phase4-step4ar1-evidence/gate-b-headroom.json',import.meta.url),'utf8'));
const temp=mkdtempSync(tmpdir()+'/phase4br1-');
process.env.PHASE4_EVIDENCE=temp;process.env.PHASE4_INVOCATION='fixture';
let transport,submissions=0,lastResponse,lastArgs;
globalThis.fetch=async(...args)=>{submissions++;lastArgs=args;lastResponse=transport(...args);return lastResponse;};
const {observe,observerIdle}=await import('./observer.mjs');
const observedFetch=globalThis.fetch;
const results=[],classifications=[];const unhandled=[];process.on('unhandledRejection',e=>unhandled.push(String(e)));
function gateSetup(){const gate=new Gate();gate.attempt('ordinary',1200);globalThis.phase4Gate={uncertain:r=>gate.uncertainty(r),httpStart:(...a)=>gate.httpStart(...a)};return gate;}
async function check(name,fn){await fn();results.push({name,pass:true});console.log('PASS',name);}
function fixture(es,{ending='eof',raw}={}) {
 let controller,cancelled=false;
 const response=new Response(new ReadableStream({start(c){controller=c;},cancel(){cancelled=true;}}),{headers:{'content-type':'text/event-stream'}});
 const task=async()=>{controller.enqueue(new TextEncoder().encode(raw??encode(es)));await new Promise(r=>setTimeout(r,10));if(cancelled)return;if(ending==='eof')controller.close();else controller.error(ending==='abort'?new DOMException('fixture cleanup','AbortError'):new Error(ending));};
 return {response,task};
}
async function run(es,options={}) {
 const gate=gateSetup(),rows=[];const {response,task}=fixture(es,options);
 const p=observe(response,'00:1',x=>rows.push(structuredClone(x)));await Promise.all([p,task()]);
 classifications.push({events:es.map(e=>e.type),fixture:options,terminal:rows.find(x=>x.kind==='native_message_stop'),outcomes:rows.filter(x=>['observer_eof','observer_read_error'].includes(x.kind)),gate});
 return {gate,rows};
}
const accepted=r=>r.rows.some(x=>x.kind==='native_message_stop'&&x.terminal_validated);
const benign=r=>r.rows.some(x=>x.classification==='POST_TERMINAL_CLONE_ABORT');
function refused(r){assert(r.gate.stopped);assert.throws(()=>r.gate.effect({name:'read_image',callId:'negative',arguments:{file_path:'image-A.png'}},'/fixture',()=>{}),/PHASE4_STOP/);assert.throws(()=>r.gate.httpStart('ordinary',10,1200),/PHASE4_STOP/);assert.equal(r.gate.ordinary,1);assert.equal(r.gate.allocation,1200);assert(!benign(r));}
try {
 await check('T1 retained exact terminal then clone AbortError',async()=>{const r=await run(genuine,{ending:'abort'});assert(accepted(r)&&benign(r));assert.equal(r.gate.stopped,null);assert.deepEqual(r.rows.filter(x=>x.kind==='sse_event').map(x=>x.event),genuine);assert.deepEqual(r.rows.find(x=>x.kind==='native_message_stop').usage,wire.find(x=>x.kind==='native_message_stop').usage);const error=wire.find(x=>x.kind==='observer_read_error');assert.equal(error.error_class,'AbortError');writeFileSync(new URL('replay-00-1.json',import.meta.url),JSON.stringify({kind:'OFFLINE_COUNTERFACTUAL_ONLY',pass:true,event_sha256:createHash('sha256').update(JSON.stringify(genuine)).digest('hex'),event_count:genuine.length,original_error:error,terminal:r.rows.find(x=>x.kind==='native_message_stop'),classification:'POST_TERMINAL_CLONE_ABORT',gate:r.gate,historical_disposition:'BLOCKED; denied tool remains failed; no C8 PASS'},null,2)+'\n');});
 await check('T2 complete normal EOF',async()=>{const r=await run(genuine);assert(accepted(r));assert.equal(r.gate.stopped,null);assert(r.rows.some(x=>x.classification==='VALIDATED_TERMINAL_EOF'));});
 await check('T3 ordinary end_turn cleanup',async()=>{const r=await run(text,{ending:'abort'});assert(accepted(r)&&benign(r));assert.equal(r.gate.stopped,null);});
 await check('T5 abort before terminal',async()=>refused(await run(genuine.slice(0,-1),{ending:'abort'})));
 await check('T6 EOF before terminal',async()=>refused(await run(genuine.slice(0,-1))));
 await check('T7 missing stop reason',async()=>refused(await run(genuine.filter(e=>e.type!=='message_delta'),{ending:'abort'})));
 await check('T8 unclosed block',async()=>refused(await run(genuine.filter(e=>!(e.type==='content_block_stop'&&e.index===1)),{ending:'abort'})));
 await check('T9 malformed completed tool JSON',async()=>{const es=structuredClone(genuine);es.find(e=>e.delta?.partial_json==='}').delta.partial_json='';const r=await run(es,{ending:'abort'});refused(r);assert(!accepted(r));});
 await check('T10 malformed SSE JSON before terminal',async()=>refused(await run([],{raw:'data: {bad}\n\n'+encode(genuine),ending:'abort'})));
 await check('T11 duplicated terminal sequence',async()=>refused(await run([...genuine,{type:'message_stop'}],{ending:'abort'})));
 await check('T12 preterminal timeout and transport failure',async()=>{for(const ending of ['TimeoutError','socket reset'])refused(await run(genuine.slice(0,3),{ending}));});
 await check('T13 unexpected exception after terminal and logging failure',async()=>{refused(await run(genuine,{ending:'unexpected observer exception'}));const gate=gateSetup();const {response,task}=fixture(genuine,{ending:'abort'});await Promise.all([observe(response,'logging-fixture',x=>{if(x.kind==='native_message_stop')throw Error('injected record failure');}),task()]);assert(gate.stopped);});
 await check('T14 existing sticky stop never cleared',async()=>{const gate=gateSetup();gate.uncertainty('earlier genuine failure');const {response,task}=fixture(genuine,{ending:'abort'});const rows=[];await Promise.all([observe(response,'sticky',x=>rows.push(x)),task()]);assert.equal(gate.stopped,'earlier genuine failure');assert(rows.some(x=>x.classification==='POST_TERMINAL_CLONE_ABORT'));});
 await check('extra invalid terminal ordering / stop reason / duplicate tool identity',async()=>{
  refused(await run([{type:'message_stop'}],{ending:'abort'}));
  const bad=structuredClone(genuine);bad.find(e=>e.type==='message_delta').delta.stop_reason='invalid';refused(await run(bad,{ending:'abort'}));
  const tool=genuine.find(e=>e.type==='content_block_start'&&e.index===1);refused(await run([...genuine.slice(0,-2),{...tool,index:2},{type:'content_block_stop',index:2},...genuine.slice(-2)],{ending:'abort'}));
 });
 await check('extra malformed JSON after terminal',async()=>refused(await run([],{raw:encode(genuine)+'data: {bad}\n\n',ending:'abort'})));
 await check('extra truncated tool JSON under max_tokens',async()=>{const es=structuredClone(genuine);es.find(e=>e.type==='message_delta').delta.stop_reason='max_tokens';es.find(e=>e.delta?.partial_json==='}').delta.partial_json='';refused(await run(es,{ending:'abort'}));});
 await check('extra duplicated index / wrong delta / event identity / tail',async()=>{for(const es of [[...genuine.slice(0,2),genuine[1],...genuine.slice(2)],genuine.map(e=>e.type==='content_block_delta'&&e.index===1?{...e,index:99}:e)])refused(await run(es));refused(await run([],{raw:encode(genuine).replace('event: message_stop','event: other')}));refused(await run([],{raw:encode(genuine).trimEnd()}));});
 // Actual installed adapter lifecycle over a controlled transport. No subclass,
 // request replacement, protocol translator fixture, network or Runtime.
 async function adapterCase(es,{agent=false}={}) {
  const gate=gateSetup(),order=[],observerUncertainties=[];globalThis.phase4Gate.uncertain=r=>{observerUncertainties.push(r);gate.uncertainty(r);};let aborted=false,returnedSame=false,transportCalls=0,terminalBeforeCleanup=false;
  const before=submissions;
  transport=(input,init)=>{
   transportCalls++;assert.equal(input,'http://127.0.0.1:18083/v1/messages');
   let c;const response=new Response(new ReadableStream({start(x){c=x;}}),{headers:{'content-type':'text/event-stream'}});
   init.signal.addEventListener('abort',()=>{aborted=true;const journal=readFileSync(temp+'/wire.jsonl','utf8').trim().split('\n').map(JSON.parse);const id=journal.filter(x=>x.kind==='request').at(-1).request;terminalBeforeCleanup=journal.some(x=>x.request===id&&x.terminal_validated===true);order.push('adapter_cleanup_abort');c.error(new DOMException('adapter consumer cleanup','AbortError'));},{once:true});
   // Pauses are bounded transport scheduling, not observer/Agent effect delays.
   (async()=>{for(const e of es){c.enqueue(new TextEncoder().encode(encode([e])));await new Promise(r=>setTimeout(r,1));}})().catch(e=>unhandled.push(String(e)));
   return response;
  };
  // Interpose only a TEST assertion around the already-installed observer wrapper.
  globalThis.fetch=async(...args)=>{const r=await observedFetch(...args);assert.equal(lastArgs[0],args[0]);assert.equal(lastArgs[1],args[1]);assert.equal(r,lastResponse);returnedSame=true;return r;};
  const adapter=new DeepSeekAdapter({options:()=>({...full.effective.adapter,baseURL:'http://127.0.0.1:18083',streamIdleTimeoutMs:1000}),prepareExtensions:async()=>({fields:{},accept:async()=>{}}),resolveAuth:async()=>({headers:{}}),resolveUserId:()=> 'model-free-observer-test',resolveFiles:()=>({})});
  let events;
  if(agent){
   const {Context}=await load('cordis');const {SessionId}=await load('dsh-session');const {createUserMessage}=await load('dsh-llm');const {defineContentToolFixture}=await load('dsh-tools');const ctx=new Context();let reads=0;
   try{
    for(const n of ['dsh-llm','dsh-session','dsh-session-projection','dsh-system-prompt','dsh-tools','dsh-agent','dsh-token-meter'])await ctx.plugin((await load(n)).default);
    ctx.llm.registerAdapter(['deepseek-official'],adapter);
    ctx.tools.register(defineContentToolFixture({name:'read_image',description:full.effective.actual_tool.description,parameters:{file_path:{type:'string',required:true}},async execute(){assert.equal(gate.stopped,null);reads++;order.push('no_effect_tool');return [{type:'text',text:'Model-free no-effect boundary fixture. No image acquired.'}];}}));
    // gateSetup reserved one only for unit fixtures; registration owns attempts here.
    gate.ordinary=gate.attempts=gate.allocation=0;
    install(ctx,{gate,workspace:'/fixture',round:()=> 'round1',authenticate(label){assert.equal(label,'A');const r=manifest.find(x=>x.label===label);assert.equal(createHash('sha256').update(readFileSync(new URL('../../../'+r.source,import.meta.url))).digest('hex'),r.sha256);}});
    ctx.on('agent/pre-step',(_,next)=>reads?{kind:'reject'}:next(),{prepend:true});
    await ctx.plugin((await load('dsh-agent-loop')).default,{agents:[],maxParallelToolCalls:1});
    const a=await ctx.agentLoop.create(SessionId('MODEL_FREE_OBSERVER_BOUNDARY'),{provider:'deepseek-official',model:'deepseek-flash'});
    a.followup(createUserMessage({content:[{type:'text',text:'Model-free no-effect observer boundary.'}],source:{kind:'user'}}));await a.whenIdle();events=a.session.snapshotEvents();
    assert.equal(reads,1);assert.equal(gate.effects.length,1);assert.equal(gate.pending,null);assert.equal(events.filter(e=>e.type==='tool/call').length,1);assert.equal(events.find(e=>e.type==='tool/result').data.message.isError,false);
   }finally{await ctx.fiber.dispose();}
  }else{
   const chunks=[];for await(const c of adapter.stream({provider:'deepseek-official',model:'deepseek-flash',maxTokens:1200,messages:[{role:'user',content:[{type:'text',text:'model-free fixture'}]}]})){chunks.push(c);if(c.type==='finish'){for(let i=0;i<100;i++){const journal=readFileSync(temp+'/wire.jsonl','utf8').trim().split('\n').map(JSON.parse);const id=journal.filter(x=>x.kind==='request').at(-1).request;if(journal.some(x=>x.request===id&&x.terminal_validated))break;await new Promise(r=>setTimeout(r,1));}}}
   order.push('official_adapter_completed');assert.equal(chunks.filter(c=>c.type==='finish').length,1);assert.equal(chunks.filter(c=>c.type==='block-end'&&c.block.type==='tool-call').length,es===genuine?1:0);
  }
  await observerIdle();await new Promise(r=>setTimeout(r,10));
  const rows=readFileSync(temp+'/wire.jsonl','utf8').trim().split('\n').map(JSON.parse);const request=rows.filter(x=>x.kind==='request').at(-1).request;const own=rows.filter(x=>x.request===request);
  assert(own.some(x=>x.kind==='native_message_stop'&&x.terminal_validated));assert(own.some(x=>x.classification==='POST_TERMINAL_CLONE_ABORT'));assert(aborted&&returnedSame);assert.deepEqual(observerUncertainties,[]);assert.equal(gate.stopped,agent?'non-complete official turn: no resubmission':null);if(!agent)assert(terminalBeforeCleanup);assert.equal(submissions-before,1);assert.equal(transportCalls,1);assert.equal(gate.http,1);assert.equal(gate.ordinary,1);assert.equal(gate.allocation,1200);
  if(agent)assert(order.indexOf('adapter_cleanup_abort')<order.indexOf('no_effect_tool'));
  return {order,request,gate,observerUncertainties,terminalBeforeCleanup,test_stop:agent?'intentional pre-step rejection AFTER successful no-effect tool; guard correctly makes this sticky':null,original_response_identity:true,original_fetch_argument_identity:true,submissions:1,actual_external_effects:0,vision:false};
 }
 await check('installed adapter tool response terminal/finally',async()=>{const r=await adapterCase(genuine);writeFileSync(new URL('adapter-lifecycle.json',import.meta.url),JSON.stringify(r,null,2)+'\n');});
 await check('installed adapter ordinary text terminal/finally',async()=>{await adapterCase(text);});
 await check('T4 installed Agent registered guard authenticated A no-effect race',async()=>{const r=await adapterCase(genuine,{agent:true});writeFileSync(new URL('guard-race.json',import.meta.url),JSON.stringify(r,null,2)+'\n');});
 await check('installed idle timeout before semantic terminal; no resubmission',async()=>{
  const gate=gateSetup(),before=submissions;
  transport=(_input,init)=>new Response(new ReadableStream({start(c){c.enqueue(new TextEncoder().encode(encode(genuine.slice(0,1))));init.signal.addEventListener('abort',()=>c.error(new DOMException('watchdog abort','AbortError')),{once:true});}}),{headers:{'content-type':'text/event-stream'}});
  const adapter=new DeepSeekAdapter({options:()=>({...full.effective.adapter,baseURL:'http://127.0.0.1:18083',streamIdleTimeoutMs:30}),prepareExtensions:async()=>({fields:{},accept:async()=>{}}),resolveAuth:async()=>({headers:{}}),resolveUserId:()=> 'model-free-timeout',resolveFiles:()=>({})});
  await assert.rejects(async()=>{for await(const c of adapter.stream({provider:'deepseek-official',model:'deepseek-flash',maxTokens:1200,messages:[{role:'user',content:[{type:'text',text:'fixture'}]}]})){}},e=>e.code==='TIMEOUT');
  await observerIdle();assert(gate.stopped);assert.equal(submissions-before,1);assert.equal(gate.http,1);assert.equal(gate.ordinary,1);assert.equal(gate.allocation,1200);
 });
 await check('malformed clone cancellation preserves original bytes and EOF',async()=>{
  const gate=gateSetup(),raw='data: {bad}\\n\\n'+encode(genuine);const {response,task}=fixture([],{raw});const clone=response.clone();
  const original=response.text();await Promise.all([observe(clone,'malformed-clone',()=>{}),task()]);assert.equal(await original,raw);assert(gate.stopped);
 });
 await check('observer tasks retired; no unhandled promises',async()=>{await observerIdle();await new Promise(r=>setTimeout(r,20));assert.deepEqual(unhandled,[]);});
 writeFileSync(new URL('fixture-wire.jsonl',import.meta.url),readFileSync(temp+'/wire.jsonl'));
 writeFileSync(new URL('classifications.json',import.meta.url),JSON.stringify(classifications,null,2)+'\n');
 writeFileSync(new URL('results.json',import.meta.url),JSON.stringify({pass:true,node:process.version,adapterIdentity,tests:results,count:results.length,model_loads:0,runtime_requests:0,controlled_adapter_submissions:submissions,unhandled},null,2)+'\n');
}finally{rmSync(temp,{recursive:true,force:true});}
