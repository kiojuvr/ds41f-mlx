// Temporary approved passive observer, scoped to this isolated official CLI process.
// It does not rewrite bodies/headers, proxy requests, dispatch tools, or retry.
import {appendFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const require = createRequire(process.env.HOME + '/.local/share/deepseek-harness/package.json');
const sharp = require('sharp');
const output = '/tmp/ds41f-phase0-harness.oLw1RR/wire-summary.jsonl';
const hash = data => createHash('sha256').update(data).digest('hex');
const record = value => appendFileSync(output, JSON.stringify({at: new Date().toISOString(), ...value})+'\n', {mode:0o600});
const nativeFetch = globalThis.fetch;
let sequence = 0;
async function observeSSE(response, request) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', events = [], blocks = {}, stopReason = null, messageStop = false;
  const event = raw => {
    const data = raw.split('\n').filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n');
    if (!data || data === '[DONE]') return;
    let obj; try { obj=JSON.parse(data); } catch { events.push('invalid_json'); return; }
    events.push(obj.type);
    if (obj.type === 'content_block_start') {
      const b=obj.content_block;
      blocks[obj.index]={type:b.type, ...(b.id?{id:b.id}:{}), ...(b.name?{name:b.name}:{}), text:b.text??'', args:''};
    }
    if (obj.type === 'content_block_delta') {
      const b=blocks[obj.index];
      if (b && obj.delta.type==='text_delta') b.text += obj.delta.text;
      if (b && obj.delta.type==='input_json_delta') b.args += obj.delta.partial_json;
    }
    if (obj.type==='message_delta') stopReason=obj.delta?.stop_reason??stopReason;
    if (obj.type==='message_stop') messageStop=true;
  };
  while (true) {
    const {value,done}=await reader.read();
    buffer+=decoder.decode(value,{stream:!done}).replace(/\r\n/g,'\n');
    let split;
    while ((split=buffer.indexOf('\n\n'))>=0) { event(buffer.slice(0,split)); buffer=buffer.slice(split+2); }
    if (done) {if(buffer.trim())event(buffer);break;}
  }
  const result=Object.values(blocks).map(b=> b.type==='tool_use'?{type:b.type,id:b.id,name:b.name,arguments:JSON.parse(b.args||'{}')}:b.type==='text'?{type:b.type,text:b.text.slice(0,2048)}:{type:b.type});
  record({kind:'sse_complete',request,events,stop_reason:stopReason,message_stop:messageStop,blocks:result});
}
globalThis.fetch = async function(input, init) {
  const url = new URL(typeof input==='string'||input instanceof URL?input:input.url);
  if (url.hostname !== '127.0.0.1' || url.port !== '8000') return nativeFetch(input,init);
  const request=++sequence;
  const method=init?.method??input?.method??'GET';
  const body=init?.body;
  const summary={kind:'request',request,method,path:url.pathname};
  if (typeof body==='string') {
    summary.body_bytes=Buffer.byteLength(body);
    const json=JSON.parse(body);
    summary.model=json.model;
    summary.max_tokens=json.max_tokens;
    summary.thinking=json.thinking;
    summary.stream=json.stream;
    summary.tools=(json.tools??[]).map(t=>t.name);
    summary.extension_names=Object.keys(json).filter(k=>k.startsWith('dsh_'));
    summary.tool_uses=[];summary.tool_results=[];summary.images=[];
    for (const [mi,m] of (json.messages??[]).entries()) {
      const walk=async(blocks,parent)=>{
        for (const [bi,b] of (Array.isArray(blocks)?blocks:[]).entries()) {
          const location=`messages[${mi}].content${parent}[${bi}]`;
          if(b.type==='tool_use')summary.tool_uses.push({id:b.id,name:b.name});
          if(b.type==='tool_result') {
            summary.tool_results.push({tool_use_id:b.tool_use_id,is_error:b.is_error??false,content_types:Array.isArray(b.content)?b.content.map(c=>c.type):typeof b.content});
            await walk(b.content,`${parent}[${bi}].content`);
          }
          if(b.type==='image') {
            const image={location,source_type:b.source?.type,media_type:b.source?.media_type};
            if(b.source?.type==='base64') {
              const bytes=Buffer.from(b.source.data,'base64');const m=await sharp(bytes).metadata();
              Object.assign(image,{bytes:bytes.length,sha256:hash(bytes),width:m.width,height:m.height,format:m.format});
            }
            summary.images.push(image);
          }
        }
      };
      await walk(m.content,'');
    }
  } else if (body instanceof FormData) {
    summary.multipart=true;
    const file=body.get('file');
    if(file instanceof Blob) {
      const bytes=Buffer.from(await file.arrayBuffer());const m=await sharp(bytes).metadata();
      summary.upload={media_type:file.type,bytes:bytes.length,sha256:hash(bytes),width:m.width,height:m.height};
    }
  }
  record(summary);
  const started=performance.now();
  let response;
  try {response=await nativeFetch(input,init);} catch(error) {record({kind:'transport_error',request,error_class:error.name});throw error;}
  record({kind:'response',request,status:response.status,headers_latency_ms:Math.round(performance.now()-started)});
  if(response.headers.get('content-type')?.includes('text/event-stream')) {
    observeSSE(response.clone(),request).catch(error=>record({kind:'observer_error',request,error_class:error.name}));
  } else if(!response.ok) {
    response.clone().json().then(j=>record({kind:'http_error',request,type:j.error?.type??null,code:j.error?.code??null})).catch(()=>{});
  }
  return response;
};
record({kind:'observer_loaded',node:process.version});
