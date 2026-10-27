// Task path/effect budget only. Official tools and lifecycle remain owners.
import {realpathSync,appendFileSync} from 'node:fs';
import {resolve} from 'node:path';
export const name='protocol-budget';
export const inject=['tools','sessions'];
export function apply(ctx){
 const workspace=realpathSync(process.env.PHASE3_WORKSPACE); let admitted=0;
 const record=x=>{try{appendFileSync(process.env.PHASE3_EVIDENCE+'/lifecycle.jsonl',JSON.stringify({at:new Date().toISOString(),...x})+'\n');}catch{}};
 ctx.tools.guard(exec=>{
  const a=exec.arguments??{}; const p=typeof a.file_path==='string'?resolve(workspace,a.file_path):'';
  const reason=exec.name!=='read_image'||!['A','B','C','D'].some(x=>p===workspace+'/image-'+x+'.png')||a.sandbox_permissions!==undefined||admitted>=5?'Step 3A path/effect budget':undefined;
  record({kind:'tool_guard',call_id:exec.callId,name:exec.name,arguments:a,decision:reason?'deny':'allow'}); if(!reason)admitted++; return reason;
 });
 ctx.on('session/event',(session,event)=>{record({kind:'session_event',session_id:session.id,event});});
 ctx.on('agent/request-error',({failure},next)=>{record({kind:'request_error',failure});return next();});
}
