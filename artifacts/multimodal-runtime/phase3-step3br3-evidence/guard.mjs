// Passive official event/stream observation plus task-owned pre-step stop fence.
// Never dispatches tools, calls compaction, writes Session events, or owns shutdown.
import {appendFileSync,readFileSync,existsSync,realpathSync} from 'node:fs';
import {resolve} from 'node:path';
export const name='native-compaction-budget';
export const inject=['tools','sessions','llm','tokenMeter'];
const path=()=>process.env.PHASE3_EVIDENCE+'/lifecycle.jsonl';
function rows(){return existsSync(path())?readFileSync(path(),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];}
export function stopReason(events, pressure){
 const starts=events.filter(e=>e.type==='compaction/start');
 if(!starts.length)return null;
 if(starts.length>1)return 'second compaction prohibited';
 const ends=events.filter(e=>e.type==='compaction/end');
 if(ends.length!==starts.length)return 'unclosed compaction';
 if(ends.at(-1).data.error!==undefined)return 'failed summary: no retry';
 const summary=events.find(e=>e.type==='compaction/summary');
 if(!summary||!events.some(e=>e.surfaceOp?.op==='replace'&&e.sourceEventSeqs?.includes(summary.seq)))return 'incomplete replacement transaction';
 if(pressure>=4096)return 'pressure still high: no second summary';
 return null;
}
export async function boundary(next,events,pressure,record){
 let reason=stopReason(events(),pressure());
 if(reason){record(reason);return {kind:'reject'};}
 const decision=await next();
 reason=stopReason(events(),pressure());
 if(reason){record(reason);return {kind:'reject'};}
 return decision;
}
export function apply(ctx){
 const workspace=realpathSync(process.env.PHASE3_WORKSPACE);
 const record=x=>appendFileSync(path(),JSON.stringify({at:new Date().toISOString(),invocation:process.env.PHASE3_INVOCATION,...x})+'\n');
 let effects=rows().filter(r=>r.kind==='tool_guard'&&r.decision==='allow').length;
 ctx.tools.guard(exec=>{
  const args=exec.arguments??{},p=typeof args.file_path==='string'?resolve(workspace,args.file_path):'';
  const reason=exec.name!=='read_image'||!['A','B'].some(x=>p===workspace+'/image-'+x+'.png')||args.sandbox_permissions!==undefined||effects>=2?'frozen two-image effect budget':undefined;
  record({kind:'tool_guard',call_id:exec.callId,name:exec.name,arguments:args,decision:reason?'deny':'allow'});
  if(!reason)effects++;return reason;
 });
 ctx.on('session/event',(session,event)=>{
  record({kind:'session_event',session_id:session.id,event});
  if(['turn/end','compaction/start','compaction/end'].includes(event.type))
   record({kind:'surface',session_id:session.id,at_seq:event.seq,header:session.requestHeader(),measurement:ctx.tokenMeter.measure(session),nodes:[...session.surface.nodes],messages:session.deriveMessages()});
 });
 // prepend is the supported Cordis ordering option; wraps the unmodified engine.
 ctx.on('agent/pre-step',({agent,turn,step},next)=>{
  const session=agent.session;
  record({kind:'pre_step',session_id:session.id,turn,step,header:session.requestHeader(),measurement:ctx.tokenMeter.measure(session),nodes:[...session.surface.nodes]});
  return boundary(next,()=>session.snapshotEvents(),()=>ctx.tokenMeter.measure(session).totalTokens,reason=>record({kind:'safe_boundary_stop',reason,session_id:session.id}));
 },{prepend:true});
 ctx.on('llm/stream',(options,next)=>{
  record({kind:'llm_stream',purpose:options.purpose??'ordinary',provider:options.provider,model:options.model,maxTokens:options.maxTokens,session_id:options.sessionId,messages:options.messages});
  return next();
 });
 ctx.on('agent/request-error',({failure},next)=>{record({kind:'request_error',failure});return next();});
}
