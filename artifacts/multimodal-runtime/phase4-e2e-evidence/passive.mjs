// Task-only passive official lifecycle collector. No decisions or Session writes.
import {appendFileSync} from 'node:fs';
export const name='phase4-native-passive';
export const inject=['tokenMeter'];
export function apply(ctx){
 const record=x=>appendFileSync(process.env.PHASE4_EVIDENCE+'/lifecycle.jsonl',JSON.stringify({at:new Date().toISOString(),invocation:process.env.PHASE4_INVOCATION,...x})+'\n');
 const surface=(session,kind,extra={})=>record({kind,session_id:session.id,header:session.requestHeader(),measurement:ctx.tokenMeter.measure(session),nodes:[...session.surface.nodes],messages:session.deriveMessages(),...extra});
 ctx.on('session/event',(session,event)=>{record({kind:'session_event',session_id:session.id,event,...(event.surfaceOp?.op==='replace'?{framed_summary_tokens:ctx.tokenMeter.estimateMessage(event.data)}:{})});if(['turn/end','compaction/start','compaction/end'].includes(event.type))surface(session,'surface',{at_seq:event.seq});});
 ctx.on('agent/pre-step',async({agent,turn,step},next)=>{surface(agent.session,'pre_step',{turn,step});const result=await next();surface(agent.session,'post_step',{turn,step});return result;},{prepend:true});
 ctx.on('llm/stream',(options,next)=>{record({kind:'llm_stream',purpose:options.purpose??'ordinary',provider:options.provider,model:options.model,maxTokens:options.maxTokens,session_id:options.sessionId,messages:options.messages});return next();});
 ctx.on('agent/request-error',({failure},next)=>{record({kind:'request_error',failure});return next();},{prepend:true});
 record({kind:'passive_registered'});
}
