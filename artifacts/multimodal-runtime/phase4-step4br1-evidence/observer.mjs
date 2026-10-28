// Passive process-local observer; original fetch args and responses pass unchanged.
// No credential headers, base64 bodies, provider/tool changes, or retry.
import {appendFileSync,readFileSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {validateTerminal} from './terminal.mjs';
const require = createRequire(process.env.HOME+'/.local/share/deepseek-harness/package.json');
const sharp = require('sharp');
const hash = b => createHash('sha256').update(b).digest('hex');
const record = x => appendFileSync(process.env.PHASE4_EVIDENCE+'/wire.jsonl', JSON.stringify({at:new Date().toISOString(),...x})+'\n');
const nativeFetch = globalThis.fetch;
let sequence=0;
const previous=()=>existsSync(process.env.PHASE4_EVIDENCE+'/wire.jsonl')?readFileSync(process.env.PHASE4_EVIDENCE+'/wire.jsonl','utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];
// Completion handles are task diagnostics only; never awaited by fetch/Agent.
const active=new Set();
export const observerIdle=()=>Promise.all([...active]);
function uncertain(reason) {
  try {globalThis.phase4Gate?.uncertain(reason);}
  catch(e) {console.error('Phase4 uncertainty receipt failure:',e);}
}
export async function observe(response, request, emit=record) {
  let reader, captured=false, buffer='', bytes=0;
  const decoder=new TextDecoder('utf-8',{fatal:true});
  const events=[], usage=[];
  let stop=null;
  const consume=async raw=>{
    const frames=[];
    const {createParser}=require('eventsource-parser');
    createParser({onEvent:e=>frames.push(e),onError:e=>{throw e;}}).feed(raw+'\n\n');
    for(const frame of frames) {
      const e=JSON.parse(frame.data);
      if(!e||typeof e!=='object'||Array.isArray(e)||typeof e.type!=='string'||(frame.event!==undefined&&frame.event!==e.type))throw Error('Malformed SSE event identity');
      emit({kind:'sse_event',request,event:e});
      if(captured)throw Error('Event after validated message_stop');
      if(e.type==='error')throw Error('Provider error event');
      events.push(e);
      if(e.usage||e.message?.usage)usage.push(e.usage??e.message.usage);
      if(e.type==='message_delta')stop=e.delta?.stop_reason??stop;
      if(e.type==='message_stop') {
        const terminal=await validateTerminal(events);
        emit({kind:'native_message_stop',request,events:events.map(e=>e.type),usage,stop_reason:stop,...terminal,terminal_validated:true});
        // Only after translator, completed tool JSON AND durable receipt succeed.
        captured=true;
      }
    }
  };
  try {
    reader=response.body.getReader();
    while(true) {
      let next;
      // Only an error from this read can qualify as transport cleanup. Parser,
      // validation, logging and other exceptions take the outer fail-closed path.
      try {next=await reader.read();}
      catch(e) {
        if(e.name==='AbortError'&&captured&&!buffer.trim()) {
          decoder.decode(); // Reject an unresolved/malformed UTF-8 tail, too.
          emit({kind:'observer_read_error',request,error_class:e.name,message_stop_already_captured:true,classification:'POST_TERMINAL_CLONE_ABORT',transport_cleanup_after_terminal:true});
          return;
        }
        throw e;
      }
      const {value,done}=next;
      bytes+=value?.byteLength??0;
      if(bytes>16777216)throw Error('Observer evidence size ceiling');
      buffer=(buffer+decoder.decode(value,{stream:!done})).replace(/\r\n/g,'\n');
      let i;
      while((i=buffer.indexOf('\n\n'))>=0) {
        const raw=buffer.slice(0,i);buffer=buffer.slice(i+2);
        await consume(raw);
      }
      if(done) {
        if(buffer.trim())throw Error('Unterminated SSE frame');
        emit({kind:'observer_eof',request,message_stop:captured,classification:captured?'VALIDATED_TERMINAL_EOF':'INCOMPLETE_SSE'});
        if(!captured)uncertain('incomplete native SSE');
        break;
      }
    }
  } catch(e) {
    // Set the sticky fence before attempting diagnostics, which may itself fail.
    uncertain('native protocol observer failure');
    try {emit({kind:'observer_read_error',request,error_class:e.name,message_stop_already_captured:captured,classification:'OBSERVER_FAILURE',detail:e.message});}
    catch(loggingError) {console.error('Phase4 observer evidence failure:',loggingError);}
  } finally {
    // Do not await tee cancellation: that waits for the ORIGINAL branch. Cancelling
    // only our branch does not cancel the original consumer. Handle its promise.
    if(reader) {reader.cancel().catch(e=>console.error('Phase4 clone cancellation:',e.name));reader.releaseLock();}
  }
}
function launch(response,request) {
  const task=observe(response,request).catch(e=>{
    uncertain('observer asynchronous failure');
    console.error('Phase4 observer asynchronous failure:',e);
  });
  active.add(task);task.then(()=>active.delete(task));
}
globalThis.fetch=async function(input,init) {
  const url=new URL(typeof input==='string'||input instanceof URL?input:input.url);
  if(url.hostname!=='127.0.0.1'||url.port!=='18083') return nativeFetch(input,init);
  const request=process.env.PHASE4_INVOCATION+':'+(++sequence), body=init?.body;
  const compact=new Headers(init?.headers??input?.headers).get('x-deepseek-harness-compact')==='1';
  const r={kind:'request',request,purpose:compact?'compaction':'ordinary',method:init?.method??input?.method??'GET',path:url.pathname};
  if(typeof body==='string') {
    const j=JSON.parse(body);
    r.messages_projection=j;
    Object.assign(r,{body_bytes:Buffer.byteLength(body),body_sha256:hash(body),model:j.model,max_tokens:j.max_tokens,thinking:j.thinking,tools:(j.tools??[]).map(t=>t.name),extension_names:Object.keys(j).filter(k=>k.startsWith('dsh_')),tool_uses:[],tool_results:[],images:[]});
    const walk=async (blocks,path)=>{
      for(const [i,b] of (Array.isArray(blocks)?blocks:[]).entries()) {
        const p=path+'.'+i;
        if(b.type==='tool_use') r.tool_uses.push({id:b.id,name:b.name});
        if(b.type==='tool_result') {r.tool_results.push({tool_use_id:b.tool_use_id,is_error:b.is_error??false,content_types:Array.isArray(b.content)?b.content.map(x=>x.type):typeof b.content});await walk(b.content,p+'.tool_result');}
        if(b.type==='image') {
          const image={location:p,source_type:b.source?.type,file_id:b.source?.file_id,media_type:b.source?.media_type};
          if(b.source?.type==='base64') {
            const bytes=Buffer.from(b.source.data,'base64'), m=await sharp(bytes).metadata();
            Object.assign(image,{bytes:bytes.length,sha256:hash(bytes),width:m.width,height:m.height,format:m.format});
          }
          if(b.source?.type==='file') image.native_block={type:b.type,source:{...b.source}};
          r.images.push(image);
        }
      }
    };
    for(const [i,m] of (j.messages??[]).entries()) await walk(m.content,'messages.'+i+'.content');
    const sanitize=x=>{if(!x||typeof x!=='object')return; if(x.type==='base64'&&typeof x.data==='string'){x.sha256=hash(Buffer.from(x.data,'base64'));delete x.data;} for(const v of Object.values(x))if(typeof v==='object')sanitize(v);}; sanitize(r.messages_projection);
  } else if(body instanceof FormData) {
    r.multipart=true; r.fields=[...body.entries()].map(([name,value])=>({name,...(value instanceof Blob?{mime:value.type,filename:value.name}:{value})}));
    const file=body.get('file');
    if(file instanceof Blob) {const bytes=Buffer.from(await file.arrayBuffer()),m=await sharp(bytes).metadata();r.upload={media_type:file.type,bytes:bytes.length,sha256:hash(bytes),width:m.width,height:m.height};}
  }
  record(r);
  const calls=previous().filter(x=>x.kind==='request'&&x.path==='/v1/messages');
  if (url.pathname==='/v1/messages' && (calls.filter(x=>x.purpose===r.purpose).length>(compact?1:19) || r.body_bytes>16777216 || r.max_tokens>(compact?2048:1200))) {
    record({kind:'budget_stop',request});
    throw new Error('Protocol gate request/token/body budget exhausted; no retry authorized');
  }
  if(url.pathname==='/v1/messages') {
    if(!globalThis.phase4Gate) throw new Error('Missing Phase4 guard; no inference submission');
    globalThis.phase4Gate.httpStart(r.purpose,r.body_bytes,r.max_tokens);
  }
  const started=performance.now();
  let response;
  try {response=await nativeFetch(input,init);} catch(e) {record({kind:'transport_error',request,error_class:e.name});globalThis.phase4Gate?.uncertain('uncertain HTTP outcome');throw e;}
  record({kind:'response',request,status:response.status,headers_latency_ms:Math.round(performance.now()-started)});
  if(response.headers.get('content-type')?.includes('text/event-stream')) launch(response.clone(),request);
  else if(response.ok && url.pathname==='/v1/files') response.clone().json().then(j=>record({kind:'file_receipt',request,metadata:j})).catch(e=>record({kind:'file_receipt_error',request,error_class:e.name}));
  else if(!response.ok) response.clone().json().then(j=>record({kind:'http_error',request,type:j.error?.type,code:j.error?.code,error:j.error})).catch(()=>{});
  return response;
};
record({kind:'observer_loaded',node:process.version});
