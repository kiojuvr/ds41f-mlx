// Minimal monotonic budget fence, using the same official tools.guard API as Phase 3.
// No tools registered/dispatched, no call scheduling or result/history changes.
import {realpathSync, appendFileSync} from 'node:fs';
import {resolve} from 'node:path';
export const name = 'protocol-budget';
export const inject = ['tools'];
export function apply(ctx) {
  const workspace=realpathSync(process.env.PHASE3_WORKSPACE);
  let admitted=0;
  ctx.tools.guard(exec => {
    const a=exec.arguments??{};
    const p=typeof a.file_path==='string'?resolve(workspace,a.file_path):'';
    const reason=exec.name!=='read_image' || !['A','B','C'].some(x=>p===workspace+'/image-'+x+'.png') || a.sandbox_permissions!==undefined || admitted>=3
      ? 'Protocol gate: only three read_image effects on the supplied images are admitted' : undefined;
    appendFileSync(process.env.PHASE3_EVIDENCE+'/tool-policy.jsonl',JSON.stringify({at:new Date().toISOString(),call_id:exec.callId,name:exec.name,arguments:a,decision:reason?'deny':'allow',reason})+'\n');
    if(!reason) admitted++;
    return reason;
  });
}
