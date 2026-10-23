// Passive process-local observer; original fetch args and responses pass unchanged.
// No credential headers, base64 bodies, provider/tool changes, or retry.
import {appendFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const require = createRequire(process.env.HOME+'/.local/share/deepseek-harness/package.json');
const sharp = require('sharp');
const hash = b => createHash('sha256').update(b).digest('hex');
const record = x => appendFileSync(process.env.PHASE3_EVIDENCE+'/wire.jsonl', JSON.stringify({at:new Date().toISOString(),...x})+'\n');
const nativeFetch = globalThis.fetch;
let sequence=0;
async function observe(response, request) {
  const reader=response.body.getReader(), decoder=new TextDecoder();
  let buffer='', events=[], blocks={}, stop=null, captured=false;
  const consume = raw => {
    const data=raw.split('\n').filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n');
    if (!data || data==='[DONE]') return;
    const e=JSON.parse(data); events.push(e.type);
    if(e.type==='content_block_start') blocks[e.index]={...e.content_block, arguments:''};
    if(e.type==='content_block_delta') {
      const b=blocks[e.index];
      if(e.delta.type==='text_delta') b.text=(b.text??'')+e.delta.text;
      if(e.delta.type==='input_json_delta') b.arguments+=e.delta.partial_json;
    }
    if(e.type==='message_delta') stop=e.delta?.stop_reason??stop;
    // Persist immediately at the actual event; do not wait for cloned EOF.
    if(e.type==='message_stop') {
      captured=true;
      record({kind:'native_message_stop',request,events,stop_reason:stop,blocks:Object.values(blocks).map(b=>b.type==='tool_use'?{type:b.type,id:b.id,name:b.name,input:JSON.parse(b.arguments||'{}')}:b.type==='text'?{type:b.type,text:b.text}:{type:b.type})});
    }
  };
  try {
    while(true) {
      const {value,done}=await reader.read();
      buffer+=decoder.decode(value,{stream:!done}).replace(/\r\n/g,'\n');
      let i;
      while((i=buffer.indexOf('\n\n'))>=0) {consume(buffer.slice(0,i));buffer=buffer.slice(i+2);}
      if(done) {if(buffer.trim())consume(buffer);break;}
    }
    record({kind:'observer_eof',request,message_stop:captured});
  } catch(e) { record({kind:'observer_read_error',request,error_class:e.name,message_stop_already_captured:captured}); }
}
globalThis.fetch=async function(input,init) {
  const url=new URL(typeof input==='string'||input instanceof URL?input:input.url);
  if(url.hostname!=='127.0.0.1'||url.port!=='18083') return nativeFetch(input,init);
  const request=++sequence, body=init?.body;
  const r={kind:'request',request,method:init?.method??input?.method??'GET',path:url.pathname};
  if(typeof body==='string') {
    const j=JSON.parse(body);
    Object.assign(r,{body_bytes:Buffer.byteLength(body),body_sha256:hash(body),model:j.model,max_tokens:j.max_tokens,thinking:j.thinking,tools:(j.tools??[]).map(t=>t.name),extension_names:Object.keys(j).filter(k=>k.startsWith('dsh_')),tool_uses:[],tool_results:[],images:[]});
    const walk=async (blocks,path)=>{
      for(const [i,b] of (Array.isArray(blocks)?blocks:[]).entries()) {
        const p=path+'.'+i;
        if(b.type==='tool_use') r.tool_uses.push({id:b.id,name:b.name});
        if(b.type==='tool_result') {r.tool_results.push({tool_use_id:b.tool_use_id,is_error:b.is_error??false,content_types:Array.isArray(b.content)?b.content.map(x=>x.type):typeof b.content});await walk(b.content,p+'.tool_result');}
        if(b.type==='image') {
          const image={location:p,source_type:b.source?.type,media_type:b.source?.media_type};
          if(b.source?.type==='base64') {
            const bytes=Buffer.from(b.source.data,'base64'), m=await sharp(bytes).metadata();
            Object.assign(image,{bytes:bytes.length,sha256:hash(bytes),width:m.width,height:m.height,format:m.format});
          }
          r.images.push(image);
        }
      }
    };
    for(const [i,m] of (j.messages??[]).entries()) await walk(m.content,'messages.'+i+'.content');
  } else if(body instanceof FormData) {
    r.multipart=true;
    const file=body.get('file');
    if(file instanceof Blob) {const bytes=Buffer.from(await file.arrayBuffer()),m=await sharp(bytes).metadata();r.upload={media_type:file.type,bytes:bytes.length,sha256:hash(bytes),width:m.width,height:m.height};}
  }
  record(r);
  const started=performance.now();
  let response;
  try {response=await nativeFetch(input,init);} catch(e) {record({kind:'transport_error',request,error_class:e.name});throw e;}
  record({kind:'response',request,status:response.status,headers_latency_ms:Math.round(performance.now()-started)});
  if(response.headers.get('content-type')?.includes('text/event-stream')) observe(response.clone(),request);
  else if(!response.ok) response.clone().json().then(j=>record({kind:'http_error',request,type:j.error?.type,code:j.error?.code})).catch(()=>{});
  return response;
};
record({kind:'observer_loaded',node:process.version});
