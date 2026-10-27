// In-memory guard qualification only. Ten no-effect tool DTOs from controlled
// responses; not a live/native nine-image session, not Vision or C8 evidence.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {Gate,install,manifest} from './guard.mjs';
const home=process.env.HOME+'/.local/share/deepseek-harness/node_modules/@deepseek-ai/';
const load=n=>import(home+n+'/lib/index.js');
const {Context}=await load('cordis');
const {LlmAdapter,createUserMessage,LlmError,requiredImageOffload,IMAGE_OFFLOAD_REQUIRED_CODE}=await load('dsh-llm');
const {SessionId}=await load('dsh-session');
const {defineContentToolFixture}=await load('dsh-tools');
const {DeepSeekAdapter,resolveAdapterOptions}=await load('dsh-llm-deepseek');
const full=JSON.parse(readFileSync(new URL('gate-b-headroom.json',import.meta.url),'utf8'));
const real=new DeepSeekAdapter({options:()=>full.effective.adapter,resolveFiles:()=>({})});
const ctx=new Context(),gate=new Gate();let reads=0;
class Controlled extends LlmAdapter{
 async resolveModel(p,m){return real.resolveModel(p,m);}
 imageRequestPricing(p,m){return real.imageRequestPricing(p,m);}
 async *stream(r){
  const n=requiredImageOffload(r.messages,{maxImages:3,countQuantum:1},b=>b.attachment.bytes);
  if(n)throw new LlmError('controlled count budget',IMAGE_OFFLOAD_REQUIRED_CODE,{offloadImages:n});
  if(reads<10){yield {type:'block-start',index:0,blockType:'tool-call'};yield {type:'block-end',index:0,block:{type:'tool-call',id:'guard-control-'+reads,name:'read_image',arguments:JSON.stringify({file_path:'image-'+('ABCDEFGHI'+ 'A')[reads]+'.png'})}};yield {type:'finish',reason:{kind:'tool-calls'}};}
  else {yield {type:'block-start',index:0,blockType:'text'};yield {type:'block-end',index:0,block:{type:'text',text:'Model-free guard check finished.'}};yield {type:'finish',reason:{kind:'stop'}};}
 }
}
try{
 for(const n of ['dsh-llm','dsh-session','dsh-session-projection','dsh-system-prompt','dsh-tools','dsh-agent','dsh-token-meter'])await ctx.plugin((await load(n)).default);
 ctx.llm.registerAdapter(['deepseek-official'],new Controlled());
 const d=full.effective.actual_tool;
 ctx.tools.register(defineContentToolFixture({name:d.name,description:d.description,parameters:{file_path:{type:'string',required:true,description:d.parameters.properties.file_path.description}},async execute(args){const label=('ABCDEFGHI'+'A')[reads++],r=manifest.find(x=>x.label===label);return [{type:'text',text:'No-effect guard control.'},{type:'image',attachment:{attachmentId:'sha256:'+r.sha256,mediaType:r.media_type,bytes:r.bytes,width:r.width,height:r.height,name:'image-'+label+'.png'}}];}}));
 install(ctx,{gate,workspace:'/fixture',authenticate(label){const r=manifest.find(x=>x.label===label),path=new URL('../../../'+r.source,import.meta.url);assert.equal(createHash('sha256').update(readFileSync(path)).digest('hex'),r.sha256);}});
 await ctx.plugin(await load('dsh-compaction-image-offload'));
 await ctx.plugin((await load('dsh-agent-loop')).default,{agents:[],maxParallelToolCalls:1});
 const agent=await ctx.agentLoop.create(SessionId('MODEL_FREE_GUARD_EFFECTS'),{provider:'deepseek-official',model:'deepseek-flash'});
 agent.followup(createUserMessage({content:[{type:'text',text:'Model-free ordered guard exercise.'}],source:{kind:'user'}}));await agent.whenIdle();
 assert.equal(gate.effects.length,10);assert.equal(gate.pending,null);assert.equal(gate.offloads,7);assert.equal(gate.ordinary,11);assert.equal(gate.attempts,18);assert.equal(gate.stopped,null);
 const events=agent.session.snapshotEvents();assert.equal(events.filter(e=>e.type==='tool/result').length,10);assert(!events.some(e=>e.type.startsWith('compaction/')));
 assert.notEqual(gate.effects[0].id,gate.effects[9].id);
 writeFileSync(new URL('gate-a-effects.json',import.meta.url),JSON.stringify({kind:'MODEL_FREE_OFFICIAL_AGENT_NO_EFFECT_TOOL_EXECUTION',pass:true,weights_loaded:0,effects:gate.effects,gate,official_offloads:events.filter(e=>e.type==='image/offload').map(e=>({seq:e.seq,data:e.data})),compaction_events:0,vision_execution:false,actual_external_read_effects:0},null,2)+'\n');
 console.log('PASS: actual Agent/tool/event path, A-I plus new A, ten controlled results, seven official offloads, eighteen adapter attempts/eleven ordinary reservations');
}finally{await ctx.fiber.dispose();}
