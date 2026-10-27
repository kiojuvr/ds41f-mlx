// Controlled model-free responses only; no HTTP/native inference or persisted session.
// Reuses Step 3B-R1 official service mounting. No events are injected.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
const home=process.env.HOME+'/.local/share/deepseek-harness';
const require=createRequire(home+'/package.json');
const load=n=>import(home+'/node_modules/@deepseek-ai/'+n+'/lib/index.js');
const {Context}=await load('cordis');
const {default:LLM,LlmAdapter,createUserMessage,requiredImageOffload,LlmError,IMAGE_OFFLOAD_REQUIRED_CODE}=await load('dsh-llm');
const {default:Store,SessionId,Session}=await load('dsh-session');
const {default:Projection}=await load('dsh-session-projection');
const {default:System}=await load('dsh-system-prompt');
const {default:Tools,defineContentToolFixture}=await load('dsh-tools');
const {default:Agents}=await load('dsh-agent');
const {default:Loop}=await load('dsh-agent-loop');
const {default:Meter}=await load('dsh-token-meter');
const {default:Basic}=await load('dsh-compaction-basic');
const Offload=await load('dsh-compaction-image-offload');
const {DeepSeekAdapter,resolveAdapterOptions}=await load('dsh-llm-deepseek');
const {deepFreeze}=await load('dsh-util-values');
const out=new URL('.',import.meta.url);
const tree=require('yaml').parse(readFileSync(new URL('../phase3-step3br3-evidence/composed-config.yml',out),'utf8'));
const raw={...tree.find(x=>x.id==='llm-deepseek').config,maxImagesPerRequest:3,imageOffloadCountQuantum:1,retryPolicy:{mode:'normal',maxRetries:0}};
const options=resolveAdapterOptions(raw),real=new DeepSeekAdapter({options:()=>options,resolveFiles:()=>({})});
const info=await real.resolveModel('deepseek-official','deepseek-flash');
const src=readFileSync(home+'/node_modules/@deepseek-ai/dsh-compaction-basic/lib/index.js','utf8');
const policy=new Function('deepFreeze',src.slice(src.indexOf('const DEFAULT_THRESHOLD_RATIO'),src.indexOf('//#endregion'))+';return {resolveConfig,resolveTargetPolicy,resolveCompactSpec};')(deepFreeze);
const manifest=JSON.parse(readFileSync(new URL('images.json',out),'utf8'));
const tasks=JSON.parse(readFileSync(new URL('tasks.json',out),'utf8'));
const records=[];
for(const threshold of [4096,8192]){
 const ctx=new Context(),hooks=[],attempts=[],requests=[];
 let agent,reads=0,goal=0,stopped=false;
 const config={auto:true,modelPolicies:[{provider:'deepseek-official',model:'deepseek-flash',thresholdRatio:threshold/1048576,headroomTokens:2048,retainTokens:1024,maxTokens:2048,compactionRetries:0}]};
 const resolved=policy.resolveConfig(config),targetPolicy=policy.resolveTargetPolicy(resolved,{provider:'deepseek-official',model:'deepseek-flash'});
 const spec=policy.resolveCompactSpec(targetPolicy,info.context.contextWindow,1200);
 class Control extends LlmAdapter{
  async resolveModel(p,m){return real.resolveModel(p,m);}
  imageRequestPricing(p,m){return real.imageRequestPricing(p,m);}
  async *stream(r){
   if(r.purpose==='compaction'){
    const measurement=ctx.tokenMeter.measure(agent.session);
    const selected=agent.session.surface.nodes.filter(seq=>r.messages.slice(0,-1).some(m=>JSON.stringify(m)===JSON.stringify(agent.session.deriveEventMessage(agent.session.eventAt(seq))))&&agent.session.eventAt(seq).type!=='system/message');
    attempts.push({reads,selected,retained:agent.session.surface.nodes.filter(s=>!selected.includes(s)),selected_tokens:measurement.nodes.filter(n=>selected.includes(n.seq)).reduce((a,n)=>a+n.tokens,0),measurement,maxTokens:r.maxTokens,message_count:r.messages.length});
    stopped=true;throw Error('MODEL_FREE_STOP_BEFORE_SUMMARY');
   }
   const count=requiredImageOffload(r.messages,{maxImages:3,countQuantum:1,maxBytes:134217728,byteQuantum:67108864},b=>b.attachment.bytes);
   if(count)throw new LlmError('Controlled adapter applies official count budget',IMAGE_OFFLOAD_REQUIRED_CODE,{offloadImages:count});
   requests.push({reads,images:r.messages.flatMap(m=>m.content.filter(b=>b.type==='image'&&!b.offloaded)).length});
   yield {type:'block-start',index:0,blockType:reads<goal?'tool-call':'text'};
   // Density varies across real outputs: this bounded control is not a prediction.
   const text='Controlled descriptive review of observed evidence, uncertainty and comparison. '.repeat(12);
   if(reads<goal){yield {type:'block-end',index:0,block:{type:'tool-call',id:'control-'+reads,name:'read_image',arguments:'{}'}};yield {type:'finish',reason:{kind:'tool-calls'}};}
   else {yield {type:'block-end',index:0,block:{type:'text',text}};yield {type:'finish',reason:{kind:'stop'}};}
  }
 }
 try{
  for(const p of [LLM,Store,Projection])await ctx.plugin(p);
  await ctx.plugin(System,{includeHarnessIdentity:false,includeRuntimeContext:false});
  await ctx.plugin(Tools);await ctx.plugin(Agents);await ctx.plugin(Meter);
  ctx.llm.registerAdapter(['deepseek-official'],new Control());
  ctx.tools.register(defineContentToolFixture({name:'read_image',description:'MODEL FREE no-effect attachment control, not official tool execution.',parameters:{},async execute(){const x=manifest[reads++];return [{type:'text',text:'Controlled result metadata; not a genuine image read.'},{type:'image',attachment:{attachmentId:'sha256:'+x.sha256,mediaType:x.media_type,bytes:x.bytes,width:x.width,height:x.height,name:'image-'+x.label+'.png'}}];}}));
  await ctx.plugin(Offload);await ctx.plugin(Basic,config);
  ctx.on('agent/pre-step',({agent:a},next)=>{hooks.push({reads,pressure:ctx.tokenMeter.measure(a.session).totalTokens,header:!!a.session.requestHeader(),offloads:a.session.snapshotEvents().filter(e=>e.type==='image/offload').length});return stopped?Promise.resolve({kind:'reject',reason:'model-free summary boundary reached'}):next();},{prepend:true});
  await ctx.plugin(Loop,{agents:[]});
  agent=await ctx.agentLoop.create(SessionId('model-free-phase4-'+threshold),{provider:'deepseek-official',model:'deepseek-flash'});
  const follow=async t=>{agent.followup(createUserMessage({content:[{type:'text',text:t}],source:{kind:'user'}}));await agent.whenIdle();};
  goal=3;await follow(tasks.round1);
  const round1=ctx.tokenMeter.measure(agent.session).totalTokens;
  if(!stopped){goal=6;await follow(tasks.round2);}
  const round2=ctx.tokenMeter.measure(agent.session).totalTokens;
  // At most four genuinely user-input-shaped review sections; controlled replies.
  for(const text of tasks.pressure_followups){if(stopped)break;await follow(text);}
  if(!stopped)await follow(tasks.trigger);
  const events=agent.session.snapshotEvents(),offloads=events.filter(e=>e.type==='image/offload');
  const snapshot=Session.create(SessionId('snapshot-control'),agent.session.snapshotEvents(),undefined,undefined,ctx.sessions.messageProjections);
  assert(snapshot.requestHeader());assert(attempts.length===1);assert(!events.some(e=>e.type==='compaction/summary'||e.surfaceOp?.op==='replace'));
  const groups=events.filter(e=>e.type==='assistant/message').flatMap(e=>e.data.message.content.filter(b=>b.type==='tool-call').map(b=>[e.seq,events.find(r=>r.type==='tool/result'&&r.data.message.toolCallId===b.id)?.seq]));
  assert(groups.every(g=>attempts[0].selected.includes(g[0])===attempts[0].selected.includes(g[1])));
  records.push({threshold,config,spec,round1,round2,hooks,attempts,offloads,requests,balanced_groups:groups,header_survives_snapshot:true,summary_success:false,replacement:false,failed_transaction_events:events.filter(e=>e.type.startsWith('compaction/')),budget_refusal_events:events.filter(e=>e.type==='assistant/attempt'&&JSON.stringify(e).includes('IMAGE_OFFLOAD_REQUIRED'))});
 }finally{await ctx.fiber.dispose();}
}
writeFileSync(new URL('ordering.json',out),JSON.stringify({kind:'CONTROLLED_RESPONSES_NOT_NATIVE_SESSION_EVIDENCE',weights_loaded:0,options,info,records},null,2)+'\n');
console.log(JSON.stringify(records.map(r=>({threshold:r.threshold,round1:r.round1,round2:r.round2,summary_after_reads:r.attempts[0].reads,offloads:r.offloads.length,selected_tokens:r.attempts[0].selected_tokens,pressures:r.hooks.map(h=>h.pressure)})),null,2));
