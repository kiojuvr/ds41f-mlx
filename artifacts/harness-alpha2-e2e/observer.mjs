// Passive application lifecycle evidence only.
import {appendFileSync} from 'node:fs';
export const name='alpha2-passive-observer';
export const inject=['sessions'];
export function apply(ctx){
 const log=x=>appendFileSync(process.env.ALPHA2_EVIDENCE+'/harness-events.jsonl',JSON.stringify({at:new Date().toISOString(),...x})+'\n');
 ctx.on('session/event',(session,event)=>log({kind:'session_event',session_id:session.id,event}));
 ctx.on('llm/stream',(options,next)=>{log({kind:'llm_stream',provider:options.provider,model:options.model,purpose:options.purpose??'ordinary',session_id:options.sessionId,messages:options.messages});return next();});
 ctx.on('agent/request-error',({failure},next)=>{log({kind:'request_error',failure});return next();},{prepend:true});
 log({kind:'observer_registered'});
}
