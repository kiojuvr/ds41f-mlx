// Loaded ONLY by model-free test CLI. Controls model output and no-effect tool
// execution; all config/prompt/schema/Agent/Session/meter/lifecycle owners remain
// the installed official profile. No successful summary is generated.
import {appendFileSync,readFileSync} from 'node:fs';
import {Gate} from './guard.mjs';
const home=process.env.HOME+'/.local/share/deepseek-harness/node_modules/@deepseek-ai/';
const {DeepSeekAdapter}=await import(home+'dsh-llm-deepseek/lib/index.js');
const {default:Tools}=await import(home+'dsh-tools/lib/index.js');
const {requiredImageOffload,LlmError,IMAGE_OFFLOAD_REQUIRED_CODE}=await import(home+'dsh-llm/lib/index.js');
const rows=JSON.parse(readFileSync(new URL('../phase4-step4a-evidence/images.json',import.meta.url),'utf8'));
const dir=process.env.PHASE4_EVIDENCE;
const log=x=>appendFileSync(dir+'/control.jsonl',JSON.stringify(x)+'\n');
const seen=()=>{try{return readFileSync(dir+'/control.jsonl','utf8').trim().split('\n').filter(Boolean).map(JSON.parse);}catch{return [];}};
const original=Tools.prototype.register;
Tools.prototype.register=function(d){
 if(d.name!=='read_image')return original.call(this,d);
 log({kind:'official_tool_definition',name:d.name,description:d.description,parameters:d.parameters});
 return original.call(this,{...d,async execute(args){
  const label=args.file_path.match(/image-([A-I])\.png$/)?.[1],r=rows.find(r=>r.label===label);
  if(!r)throw Error('invalid controlled source');
  log({kind:'controlled_effect',label});
  return {path:process.env.PHASE4_WORKSPACE+'/image-'+label+'.png',image:{attachmentId:'sha256:'+r.sha256,mediaType:r.media_type,bytes:r.bytes,width:r.width,height:r.height,name:'image-'+label+'.png'}};
 }});
};
const phrase='This controlled review records the inspection status, distinguishes evidence from interpretation, preserves item references and explains the limits of the descriptive comparison. ';
DeepSeekAdapter.prototype.generate=async function*(r){
 const history=seen(),reads=history.filter(x=>x.kind==='controlled_effect').length;
 if(!history.some(x=>x.kind==='actual_image_prices')){
  const blocks=rows.map(x=>({type:'image',attachment:{attachmentId:'sha256:'+x.sha256,mediaType:x.media_type,bytes:x.bytes,width:x.width,height:x.height,name:'image-'+x.label+'.png'}}));
  const pricing=this.imageRequestPricing('deepseek-official','deepseek-flash');
  log({kind:'actual_image_prices',source:'actual configured adapter plus mounted attachment/filesystem access',blocks,represented:pricing.priceImages(blocks),offloaded:pricing.priceImages(blocks.map(b=>({...b,offloaded:true})))});
 }
 if(r.purpose==='compaction'){
  log({kind:'summary_boundary',reads,maxTokens:r.maxTokens,messages:r.messages});
  throw Error('MODEL_FREE_STOP_BEFORE_SUMMARY');
 }
 const n=requiredImageOffload(r.messages,{maxImages:3,countQuantum:1,maxBytes:134217728,byteQuantum:67108864},b=>b.attachment.bytes);
 if(n){log({kind:'count_refusal',reads,n});throw new LlmError('Model-free official count budget',IMAGE_OFFLOAD_REQUIRED_CODE,{offloadImages:n});}
 const mode=process.env.PHASE4_SCENARIO;
 const goal=process.env.PHASE4_ROUND==='round1'?3:process.env.PHASE4_ROUND==='round2'?6:reads;
 const size=mode==='concise'?200:mode==='large'?4200:mode==='usage-high'?1000:mode==='representative'?(process.env.PHASE4_ROUND==='round1'?[188,584,704,1372][reads]:process.env.PHASE4_ROUND==='round2'?[380,552,716,1372][reads-3]:2080):800;
 const text=phrase.repeat(Math.ceil(size/phrase.length)).slice(0,size);
 // Tool arguments, output structure and full schema are genuine official paths;
 // this content has no visual answers and is not a native completion.
 yield {type:'block-start',index:0,blockType:'text'};
 yield {type:'block-end',index:0,block:{type:'text',text}};
 if(reads<goal){
  const label='ABCDEFGHI'[reads];
  yield {type:'block-start',index:1,blockType:'tool-call'};
  yield {type:'block-end',index:1,block:{type:'tool-call',id:'controlled-'+reads,name:'read_image',arguments:JSON.stringify({file_path:'image-'+label+'.png'})}};
 }
 // Usage is a controlled sensitivity input, never asserted measured native usage.
 const structural=globalThis.phase4Control?.measurement();
 if(mode==='usage-high'||mode==='usage-low'){
  const input=mode==='usage-high'?(structural?.totalTokens??0)+500:1;
  yield {type:'usage',usage:{inputTokens:input,outputTokens:Math.ceil(text.length/4),cacheReadTokens:0,cacheWriteTokens:0,totalTokens:input+Math.ceil(text.length/4)}};
 }
 log({kind:'controlled_response',reads,chars:text.length,usage_mode:mode,request_header:r.config??null,tools:r.tools});
 yield {type:'finish',reason:{kind:reads<goal?'tool-calls':'stop'}};
};
export const name='phase4-control-observer';
export const inject=['tokenMeter','sessions','systemPrompt'];
export function apply(ctx){
 let agent;
 globalThis.phase4Control={measurement:()=>agent?ctx.tokenMeter.measure(agent.session):null};
 ctx.on('agent/pre-step',({agent:a},next)=>{agent=a;log({kind:'pre_step',reads:seen().filter(x=>x.kind==='controlled_effect').length,measurement:ctx.tokenMeter.measure(a.session),header:a.session.requestHeader(),surface:a.session.surface.nodes});return next();});
 ctx.on('llm/stream',(r,next)=>{
  if(r.purpose==='compaction'){
   const measurement=ctx.tokenMeter.measure(agent.session);
   const selected=agent.session.surface.nodes.filter(seq=>agent.session.eventAt(seq).type!=='system/message'&&r.messages.slice(0,-1).some(m=>JSON.stringify(m)===JSON.stringify(agent.session.deriveEventMessage(agent.session.eventAt(seq)))));
   log({kind:'selected_range',selected,retained:agent.session.surface.nodes.filter(s=>!selected.includes(s)),measurement,header:agent.session.requestHeader(),groups:agent.session.snapshotEvents().filter(e=>e.type==='tool/result').map(e=>({result:e.seq,call:agent.session.snapshotEvents().find(a=>a.type==='assistant/message'&&a.data.message.content.some(b=>b.type==='tool-call'&&b.id===e.data.message.toolCallId))?.seq}))});
  }
  return next();
 });
 ctx.on('session/event',(s,e)=>{
  if(['image/offload','compaction/start','compaction/end','turn/end'].includes(e.type))log({kind:'official_event',event:e,pressure:ctx.tokenMeter.measure(s),header:s.requestHeader(),session_id:s.id});
  if(e.type==='turn/end')log({kind:'durable_snapshot',snapshot:s.snapshotEvents(),pressure:ctx.tokenMeter.measure(s),session_id:s.id,requestHeader:s.requestHeader()});
 });
 // Hard fail any attempted network access in this model-free process.
 globalThis.fetch=()=>{throw Error('MODEL_FREE_NETWORK_FORBIDDEN');};
}
