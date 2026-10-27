// Existing supported pre-step fence, no new lifecycle owner.
import {boundary} from '../phase3-step3br3-evidence/guard.mjs';
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
const {Context}=await import(process.env.HOME+'/.local/share/deepseek-harness/node_modules/@deepseek-ai/cordis/lib/index.js');
const ctx=new Context(),checks=[];
try{
 const success=[{type:'compaction/start'},{type:'compaction/summary',seq:2},{surfaceOp:{op:'replace'},sourceEventSeqs:[2]},{type:'compaction/end',data:{}}];
 // boundary's retained implementation has a 4096 threshold. Do NOT silently
 // reuse it for 8192: verify its exact boundary and record required extension.
 for(const [name,before,after,p,expected,callsExpected] of [
 ['first hook',[],null,8192,'enter',1],
 ['failed summary stops after listener',[],[{type:'compaction/start'},{type:'compaction/end',data:{error:'failed'}}],5000,'reject',1],
 ['later failed hook cannot retry',[{type:'compaction/start'},{type:'compaction/end',data:{error:'failed'}}],null,5000,'reject',0],
 ['successful safe continuation',success,null,3000,'enter',1],
 ['residual pressure blocked',success,null,8192,'reject',0],
 ['unclosed transaction',[{type:'compaction/start'}],null,1000,'reject',0],
 ['incomplete transaction',[{type:'compaction/start'},{type:'compaction/end',data:{}}],null,1000,'reject',0]
 ]){
  let events=before,calls=0;
  const a=ctx.on('check',(_,next)=>boundary(next,()=>events,()=>p,()=>{}),{prepend:true});
  const b=ctx.on('check',(_,next)=>{calls++;if(after)events=after;return next();});
  const r=await ctx.waterfall('check',{},()=>Promise.resolve({kind:'enter'}));
  assert.equal(r.kind,expected);assert.equal(calls,callsExpected);checks.push(name);a();b();
 }
}finally{await ctx.fiber.dispose();}
writeFileSync(new URL('stop-checks.json',import.meta.url),JSON.stringify({checks,pass:true,existing_guard_threshold:4096,required_step4b_extension:'parameterize acceptance threshold 8192 and post-compaction entry headroom 4608; keep post-listener check and one-attempt fence',native_guard_not_yet_qualified:true},null,2)+'\n');
console.log('PASS: seven existing stop-fence checks; threshold extension explicitly still required');
