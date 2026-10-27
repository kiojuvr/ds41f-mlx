// Phase4-only extension of Step3B-R3 guard. Observes/rejects; never dispatches,
// appends Session events, selects compaction, mutates native state or shuts down.
import {readFileSync,appendFileSync,existsSync,realpathSync,statSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
const prior=new URL('../phase4-step4a-evidence/',import.meta.url);
export const budgets=JSON.parse(readFileSync(new URL('budgets.json',prior),'utf8'));
export const manifest=JSON.parse(readFileSync(new URL('images.json',prior),'utf8'));
export const name='phase4-acceptance-budget';
export const inject=['tools','sessions','llm','tokenMeter'];
const order=[...'ABCDEFGHI','A'];
export class Gate {
 constructor(snapshot={}){Object.assign(this,{effects:[],pending:null,ordinary:0,summary:0,attempts:0,http:0,httpOrdinary:0,httpSummary:0,allocation:0,offloads:0,starts:0,round3Entered:false,stopped:null},snapshot);}
 deny(reason){this.stopped??=reason;throw Error('PHASE4_STOP: '+this.stopped);}
 alive(){if(this.stopped)this.deny(this.stopped);}
 uncertainty(reason){this.stopped??=reason;}
 effect(exec,workspace,authenticate,phase=''){
  this.alive();if(this.pending)this.deny('uncertain/uncompleted previous effect');
  if((phase==='round1'&&this.effects.length>=3)||(phase==='round2'&&this.effects.length>=6)||['review','trigger','compatible'].includes(phase)||(phase==='round3'&&!this.round3Entered))this.deny('effect outside frozen round');
  if(this.effects.length>=4&&this.offloads===0)this.deny('required official offload did not occur');
  const p=resolve(workspace,exec.arguments?.file_path??'');
  const label=order[this.effects.length];
  if(exec.name!=='read_image'||!label||p!==resolve(workspace,'image-'+label+'.png')||Object.keys(exec.arguments??{}).some(k=>k!=='file_path'))this.deny('tool/path/order/effect ceiling');
  if(!exec.callId||this.effects.some(x=>x.id===exec.callId))this.deny('duplicate Tool Call identity');
  authenticate(label,p);this.pending={label,id:exec.callId};this.effects.push(this.pending);
 }
 assistant(message){if(message.content.filter(b=>b.type==='tool-call').length>1)this.deny('one read_image call per response');}
 result(id,isError){if(!this.pending||this.pending.id!==id||isError)this.deny('uncertain/mismatched/failed effect');this.pending=null;}
 attempt(purpose,maxTokens){
  this.alive();if(this.pending)this.deny('effect not completed');
  if(this.attempts>=budgets.adapter_attempts_including_offload)this.deny('adapter attempt ceiling');
  const summary=purpose==='compaction',cap=summary?2048:1200;
  if(maxTokens!==cap)this.deny('frozen output policy');
  // Ordinary attempts reserve a generation; a confirmed pre-HTTP count refusal
  // refunds ONLY that reservation, never an actual/incomplete HTTP request.
  if(summary?this.summary>=1:this.ordinary>=19)this.deny('ordinary/summary request ceiling');
  this.attempts++;this.lastAttempt={purpose:summary?'compaction':'ordinary',http:this.http,refunded:false};if(summary)this.summary++;else this.ordinary++;
  this.allocation+=cap;if(this.allocation>24848)this.deny('total output allocation');
 }
 budgetRefusal(failure){
  this.alive();if(failure.code!=='IMAGE_OFFLOAD_REQUIRED'||failure.offloadImages!==1||this.ordinary<1||this.lastAttempt?.purpose!=='ordinary'||this.lastAttempt.refunded||this.lastAttempt.http!==this.http)this.deny('non-count refusal: no retry');
  this.lastAttempt.refunded=true;this.ordinary--;this.allocation-=1200;
 }
 offload(){this.alive();if(++this.offloads>7)this.deny('offload ceiling');}
 httpStart(purpose,bodyBytes,maxTokens){
  this.alive();const s=purpose==='compaction';
  if(this.http>=20||(s?this.httpSummary>=1:this.httpOrdinary>=19)||bodyBytes>16777216||maxTokens!==(s?2048:1200))this.deny('HTTP/body/output ceiling');
  this.http++;if(s)this.httpSummary++;else this.httpOrdinary++;
 }
 prepared(input,output,images){this.alive();if(input+output>32768||images>3)this.deny('prepared positions/current image ceiling');}
 transaction(events,pressure,round3=false){
  this.alive();const starts=events.filter(e=>e.type==='compaction/start');
  if(starts.length>1)this.deny('second compaction prohibited');
  if(starts.length){
   const ends=events.filter(e=>e.type==='compaction/end');
   if(ends.length!==1||ends[0].data?.error!==undefined)this.deny('failed/unclosed compaction');
   const summary=events.find(e=>e.type==='compaction/summary');
   if(!summary||!events.some(e=>e.surfaceOp?.op==='replace'&&e.sourceEventSeqs?.includes(summary.seq)))this.deny('incomplete replacement');
   if(pressure>=8192)this.deny('second pressure compaction prohibited');
  }else if(pressure>=8192&&(this.effects.length<4||this.offloads<1))this.deny('premature compaction: required offload not observed');
  if(round3&&!this.round3Entered){
   if(!starts.length||this.effects.length!==6||this.pending||pressure>4608)this.deny('Round3 entry prerequisites/headroom');
   this.round3Entered=true;
  }
 }
}
export async function boundary(next,gate,events,pressure,round3){
 try{gate.transaction(events(),pressure(),round3());const decision=await next();gate.transaction(events(),pressure(),round3());return decision;}
 catch(e){if(!String(e.message).startsWith('PHASE4_STOP'))throw e;return {kind:'reject'};}
}
export function install(ctx,{gate,workspace,record=()=>{},authenticate=()=>{},round=()=>'',storageCheck=()=>{}}){
 const checked=()=>{storageCheck();return ctx.tokenMeter.measure(current.session).totalTokens;};let current;
 ctx.tools.guard(exec=>{
  try{if(current)gate.transaction(current.session.snapshotEvents(),checked());gate.effect(exec,workspace,authenticate,round());record('effect_reserved');}
  catch(e){gate.uncertainty(e.message);record('stop');return e.message;}
 });
 ctx.on('session/event',(session,event)=>{
  try{
   if(event.type==='assistant/message')gate.assistant(event.data.message);
   if(event.type==='tool/result')gate.result(event.data.message.toolCallId,event.data.message.isError);
   if(event.type==='image/offload')gate.offload();
   if(event.type==='turn/end'&&event.data.reason.kind!=='completed')gate.uncertainty('non-complete official turn: no resubmission');
   if(event.type==='compaction/start'){if(++gate.starts>1)gate.uncertainty('second compaction start');}
  }catch(e){gate.uncertainty(e.message);}
  record(event.type);
 });
 ctx.on('agent/pre-step',({agent},next)=>{current=agent;return boundary(next,gate,()=>agent.session.snapshotEvents(),checked,()=>round()==='round3').finally(()=>record('pre_step'));},{prepend:true});
 ctx.on('llm/stream',(options,next)=>{
  try{storageCheck();gate.attempt(options.purpose??'ordinary',options.maxTokens??1200);record('adapter_attempt');}
  catch(e){record('stop');throw e;}
  return next();
 });
 ctx.on('agent/request-error',({failure},next)=>{
  try{if(failure.code==='IMAGE_OFFLOAD_REQUIRED')gate.budgetRefusal(failure);else gate.uncertainty('provider/protocol failure: no retry');}
  catch(e){gate.uncertainty(e.message);}
  record('request_error');
  // Undefined is the official non-retry outcome. Short-circuit recognized
  // overflow recovery too: this finite acceptance authorizes only count retry.
  return gate.stopped?Promise.resolve(undefined):next();
 },{prepend:true});
}
export function checkReceipt(gate,s,round,now=Date.now()){
 if(!s||!Number.isFinite(s.at_ms)||!Number.isSafeInteger(s.session_bytes)||s.session_bytes<0||!Number.isSafeInteger(s.attachment_bytes)||s.attachment_bytes<0||now-s.at_ms>10000||s.at_ms>now+1000||s.stop||s.session_bytes>67108864||s.attachment_bytes>8388608)gate.deny('Supervisor/storage boundary');
 if(round==='round3'&&!gate.round3Entered&&s.persistence_ok!==true)gate.deny('Round3 requires official persisted replacement receipt');
}
export function apply(ctx){
 const dir=process.env.PHASE4_EVIDENCE,workspace=realpathSync(process.env.PHASE4_WORKSPACE);
 const journal=dir+'/gate.jsonl';const rows=existsSync(journal)?readFileSync(journal,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];
 const gate=new Gate(rows.at(-1)?.state);
 const record=kind=>appendFileSync(journal,JSON.stringify({kind,state:gate,at:new Date().toISOString()})+'\n');
 const authenticate=(label,p)=>{const row=manifest.find(r=>r.label===label);if(realpathSync(p)!==p||statSync(p).size!==row.bytes||createHash('sha256').update(readFileSync(p)).digest('hex')!==row.sha256)gate.deny('authenticated source mismatch');};
 const storageCheck=()=>{if(!['round1','round2','review','trigger','round3','compatible'].includes(process.env.PHASE4_ROUND))gate.deny('unknown invocation phase');const path=dir+'/supervision.json';if(!existsSync(path))gate.deny('missing external Supervisor receipt');let s;try{s=JSON.parse(readFileSync(path,'utf8'));}catch{gate.deny('malformed Supervisor receipt');}checkReceipt(gate,s,process.env.PHASE4_ROUND);};
 install(ctx,{gate,workspace,record,authenticate,round:()=>process.env.PHASE4_ROUND,storageCheck});
 // Passive hooks for existing wire/preparation collectors, not semantic owners.
 globalThis.phase4Gate={httpStart:(...a)=>{try{gate.httpStart(...a);}finally{record('HTTP');}},prepared:(...a)=>gate.prepared(...a),uncertain:r=>{gate.uncertainty(r);record('uncertain');}};
}
