// Analytical possible surfaces ONLY. Never creates/edits a Session, generates a
// summary, calls compaction, or claims synthetic replacement/native success.
import {readFileSync,writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const out=new URL('.',import.meta.url),read=n=>JSON.parse(readFileSync(new URL(n,out),'utf8'));
const home=process.env.HOME+'/.local/share/deepseek-harness',require=createRequire(home+'/package.json');
const {resolveAdapterOptions,DeepSeekAdapter}=await import(home+'/node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js');
const config=require('yaml').parse(readFileSync(new URL('composed-config.yml',out),'utf8'));
const options=resolveAdapterOptions(config.find(x=>x.id==='llm-deepseek').config);
const adapter=new DeepSeekAdapter({options:()=>options,resolveFiles:()=>({})});
const pricing=adapter.imageRequestPricing('deepseek-official','deepseek-flash');
const src=readFileSync(home+'/node_modules/@deepseek-ai/dsh-token-meter/lib/index.js','utf8');
const estimator=new Function(src.slice(src.indexOf('const CHARS_PER_TOKEN'),src.indexOf('function estimateToolsTokens'))+';return {estimateMessage,estimateContent,estimateStructuralBlock};')();
let mountedPrices;
const price=message=>{
 let p=estimator.estimateMessage(message);
 for(const b of message.content)if(b.type==='image'){
  const index=mountedPrices.blocks.findIndex(x=>x.attachment.attachmentId===b.attachment.attachmentId);assert(index>=0);
  const x=mountedPrices[b.offloaded?'offloaded':'represented'][index];p-=estimator.estimateStructuralBlock(b);p+=x.visualTokens+estimator.estimateContent([{type:'text',text:x.text}]);
 }
 return p;
};
const tasks=JSON.parse(readFileSync(new URL('../phase4-step4a-evidence/tasks.json',out),'utf8'));
const inputs=read('gate-b-profile.json').results;
const representative=inputs.find(x=>x.scenario==='representative');
const projections=[];
for(const r of inputs){
 const a=r.selected_range;if(!a)continue;
 const selected=a.measurement.nodes.filter(n=>a.selected.includes(n.seq)).reduce((s,n)=>s+n.tokens,0);
 assert(a.groups.every(g=>a.selected.includes(g.call)===a.selected.includes(g.result)));
 const base=a.measurement.totalTokens-selected;
 projections.push({scenario:r.scenario,pressure:a.measurement.totalTokens,selected_tokens:selected,selected:a.selected,retained:a.retained,balanced_groups:a.groups,remaining_base_including_protected_header_and_anchor_excess:base,post_compaction_possible:[{summary:'small plausible complete (not predicted)',framed_meter_tokens:600,pressure:base+600},{summary:'prior REAL complete 1053-output reference',framed_meter_tokens:1241,pressure:base+1241},{summary:'larger density extrapolation (not universal upper bound)',framed_meter_tokens:2205,pressure:base+2205}],strict_shrink_candidate_2205:2205<selected,notes:'Official measure = repriced surface + tools + eligible usage excess. A valid replacement removes selected tokens and adds framed summary; neither header nor any positive usage excess may be silently dropped.'});
}
const r=representative,ctx=r.runtime_context_messages[0].data;mountedPrices=r.actual_image_prices;
assert(ctx.source.sections.some(s=>s.name==='sandbox:policy'));
const context_cost=price(ctx),round3_user=price({role:'user',content:[{type:'text',text:tasks.round3}]}),follow_user=price({role:'user',content:[{type:'text',text:tasks.compatible}]});
const base=projections.find(x=>x.scenario==='representative').remaining_base_including_protected_header_and_anchor_excess;
const trigger_cost=price({role:'user',content:[{type:'text',text:tasks.trigger}]});
const conciseBase=projections.find(x=>x.scenario==='concise').remaining_base_including_protected_header_and_anchor_excess;
const cases=[];
// A replacement alone is NOT Round3 entry. The pending trigger still needs an
// ordinary changed-projection continuation. Charge it and refreshed context
// before applying the 4608 entry screen. No successful response is fabricated.
for(const [name,replacementPressure,postContinuationChars,callChars,finalChars,followChars] of [
 ['concise with grounded summary',conciseBase+1241,200,200,800,200],
 ['representative with grounded summary',base+1241,1400,800,2400,400],
 ['larger summary plus representative continuation',base+2205,1400,800,2400,400],
 ['larger summary plus concise continuation',base+2205,200,800,2400,400],
 ['larger permitted Round3 descriptions',base+1241,1400,1800,3600,800],
 ['cap-equivalent ASCII output sensitivity',null,null,4800,4800,4800]
]){
 const continuation=postContinuationChars===null?0:price({role:'assistant',content:[{type:'text',text:'x'.repeat(postContinuationChars)}]});
 const start=replacementPressure===null?4608:replacementPressure+context_cost+trigger_cost+continuation;
 let pressure=start+round3_user;const steps=[{at:'immediate replacement before pending trigger (analytical)',pressure:replacementPressure},{at:'entry AFTER ordinary changed-projection trigger continuation + refreshed context',pressure:start},{at:'claimed Round3 task (context already current)',pressure}];let first;
 for(const [i,label] of [...'GHIA'].entries()){
  pressure+=price({role:'assistant',content:[{type:'text',text:'x'.repeat(callChars)},{type:'tool-call',id:'call_00_CONTROLLED_MODEL_FREE_'+i,name:'read_image',arguments:JSON.stringify({file_path:'image-'+label+'.png'})}]});
  const result=structuredClone(r.sample_tool_result);result.toolCallId='call_00_CONTROLLED_MODEL_FREE_'+i;
  result.content[0].text=result.content[0].text.replaceAll('image-A.png','image-'+label+'.png');result.content[1].attachment.name='image-'+label+'.png';
  // Geometry/size differs only trivially among fixtures; use the actual source
  // metadata for each label rather than a generic image-price constant.
  const manifest=JSON.parse(readFileSync(new URL('../phase4-step4a-evidence/images.json',out),'utf8'));const m=manifest.find(x=>x.label===label);
  Object.assign(result.content[1].attachment,{attachmentId:'sha256:'+m.sha256,bytes:m.bytes});
  const delta=price(result);pressure+=delta;
  if(!first)first=result;
  steps.push({at:'after '+label+' result, next pre-step BEFORE any budget offload',pressure,tool_result_meter_cost:delta});
  if(i===3){const offloaded=structuredClone(first);offloaded.content[1].offloaded=true;pressure+=price(offloaded)-price(first);steps.push({at:'official oldest-image offload retry if reached',pressure});}
 }
 pressure+=price({role:'assistant',content:[{type:'text',text:'x'.repeat(finalChars)}]});steps.push({at:'final comparison settled; next followup pre-step',pressure});
 pressure+=follow_user+price({role:'assistant',content:[{type:'text',text:'x'.repeat(followChars)}]});steps.push({at:'compatible followup completed',pressure});
 cases.push({name,start,replacementPressure,post_compaction_continuation_chars:postContinuationChars,post_compaction_continuation_cost:continuation,trigger_cost,entry_screen_pass:start<=4608,assumed_assistant_text_chars:{four_tool_generations_each:callChars,final_comparison:finalChars,compatible:followChars},steps,max_pressure:Math.max(...steps.filter(x=>x.pressure!==null).map(x=>x.pressure)),credible_no_second_pressure_trigger:start<=4608&&steps.every(x=>x.pressure<8192),uncertainty:'ASCII fixed-density analytical text lengths, not bounds on arbitrary native token-to-character ratio or predictions of actual model output. Native 1200-token cap and actual pressure fence remain authoritative.'});
}
assert(cases[0].credible_no_second_pressure_trigger&&cases[1].credible_no_second_pressure_trigger);
assert(!cases.at(-1).credible_no_second_pressure_trigger);
const effective={adapter:options,system_prompt_config:config.find(x=>x.id==='system-prompt').config,actual_system_text:r.system_messages[0].data.message.content,actual_context_sections:ctx.source.sections,actual_tool:r.latest_header.tools[0],actual_request_header:r.latest_header,actual_context_meter_cost:context_cost,trigger_task_meter_cost:trigger_cost,round3_task_meter_cost:round3_user,compatible_task_meter_cost:follow_user};
writeFileSync(new URL('gate-b-headroom.json',out),JSON.stringify({kind:'ANALYTICAL_POSSIBLE_SURFACES_NO_SESSION_REPLACEMENT',pass:true,weights_loaded:0,effective,projections,round3:cases,summary_completion_unknown:true,native_only:['actual complete summary text and framed size','actual committed replacement and persistence','actual post-compaction pressure/usage anchor excess','native historical Vision, paired selection and settlement']},null,2)+'\n');
console.log(JSON.stringify({post:projections.map(x=>({scenario:x.scenario,base:x.remaining_base_including_protected_header_and_anchor_excess,post:x.post_compaction_possible.map(p=>p.pressure)})),round3:cases.map(c=>({name:c.name,start:c.start,max:c.max_pressure,credible:c.credible_no_second_pressure_trigger}))},null,2));
