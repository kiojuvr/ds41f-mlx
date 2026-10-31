// Unmodified installed provider serialization, no network or durable Harness state.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const [root, imageRoot] = process.argv.slice(2);
const provider = await import(pathToFileURL(`${root}/@deepseek-ai/dsh-llm-deepseek/lib/index.js`));
const local = await import(pathToFileURL(`${root}/@deepseek-ai/dsh-attachment-local/lib/index.js`));
const llm = await import(pathToFileURL(`${root}/@deepseek-ai/dsh-llm/lib/index.js`));
const model = {id:'deepseek-flash', inputModalities:['text','image']};
const results = [];
for (const [size, count, fallback] of [[1048576,13,true],[2097152,33,false]]) {
  const data = new Uint8Array(readFileSync(`${imageRoot}/${size}.png`));
  const ref = {attachmentId:`sha256:${createHash('sha256').update(data).digest('hex')}`,name:'audit.png',width:32,height:32};
  const target = provider.resolveRequestImageTarget(model,ref);
  const version = await local.readRequestImageFile(`${imageRoot}/cache`, {
    data, mediaType:'image/png',ref,depth:'uchar',space:'srgb',hasAlpha:false,
  }, target);
  assert.equal(version.bytes,size); // Prove the official request transform preserves these bytes.
  const history = [{role:'user',content:Array.from({length:count},()=>({type:'image',attachment:ref}))}];
  let reads=0, resolutions=0, captured;
  const stop = new Error('captured without network');
  const adapter = new provider.DeepSeekAdapter({
    options:()=>({}), resolveAuth:async()=>({headers:{}}),resolveUserId:()=> 'audit',
    prepareExtensions:async()=>({fields:{},accept:async()=>{}}),
    resolveAttachments:()=>({readImageRequest:async()=>{reads++;return version;}}),
    resolveFiles:()=>({ensureUploaded:async()=>{
      resolutions++;
      if(fallback) throw new Error('controlled Files unavailable');
      return {record:{fileId:`file_${'a'.repeat(32)}`}};
    }}),
  });
  const connection={baseURL:'https://audit.invalid/v1',models:[model],maxImagesPerRequest:600,
    maxRequestFilesBytes:128*2**20,maxInlineRequestImageBytes:20*2**20,
    imageOffloadCountQuantum:20,imageOffloadByteQuantum:8*2**20,inlineImageOffloadByteQuantum:2*2**20,
    filesApiTimeoutMs:1000,filePolicy:{},defaults:{thinking:'disabled'},maxTokens:64};
  const previous=globalThis.fetch;
  globalThis.fetch=async(url,init)=>{captured=init.body;throw stop;};
  try {
    await assert.rejects(adapter.request({model:model.id,messages:history,reasoningEffort:'off'},
      connection,new AbortController().signal,()=>{}).next(),e=>e===stop);
  } finally {globalThis.fetch=previous;}
  const wire=JSON.parse(captured);
  const images=wire.messages.flatMap(m=>m.content).filter(b=>b.type==='image');
  assert.equal(images.length,count);assert.equal(reads,1);
  assert.ok(images.every(b=>b.source.type===(fallback?'base64':'file')));
  const rawBytes=count*size;
  const representedBytes=images.reduce((n,b)=>n+(b.source.data?.length??0),0);
  assert.equal(llm.requiredImageOffload(history,{representation:'raw',maxBytes:128*2**20,maxImages:600,countQuantum:20},()=>size),0);
  if(fallback){assert.ok(representedBytes<20*2**20);assert.ok(Buffer.byteLength(captured)>16*2**20);}
  else {assert.ok(rawBytes>64*2**20);assert.ok(Buffer.byteLength(captured)<16*2**20);}
  results.push({size,count,fallback,target,reads,resolutions,rawBytes,representedBytes,
    wireBytes:Buffer.byteLength(captured),sourceType:images[0].source.type,passed:true});
}
console.log(JSON.stringify(results,null,2));
