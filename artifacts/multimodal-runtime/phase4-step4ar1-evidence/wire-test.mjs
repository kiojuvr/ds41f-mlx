// Existing passive wire collector extension tested with Response fixtures only;
// no HTTP server, network, official Session events or native generation.
import {mkdtempSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';
import {Gate} from './guard.mjs';
const dir=mkdtempSync(join(tmpdir(),'phase4-wire-model-free-'));
process.env.PHASE4_EVIDENCE=dir;process.env.PHASE4_INVOCATION='control';
let calls=0,incomplete=false,gate=new Gate();
globalThis.phase4Gate={httpStart:(...a)=>gate.httpStart(...a),uncertain:r=>gate.uncertainty(r)};
globalThis.fetch=async()=>{calls++;return incomplete?new Response('data: {"type":"message_start","message":{"usage":{}}}\n\n',{headers:{'content-type':'text/event-stream'}}):new Response('{}',{headers:{'content-type':'application/json'}});};
await import('./observer.mjs');
const send=summary=>fetch('http://127.0.0.1:18083/v1/messages',{method:'POST',headers:summary?{'x-deepseek-harness-compact':'1'}:{},body:JSON.stringify({model:'deepseek-flash',max_tokens:summary?2048:1200,messages:[]})});
try{
 for(let i=0;i<19;i++)await send(false);await send(true);
 assert.equal(calls,20);await assert.rejects(send(false));assert.equal(calls,20);
 writeFileSync(dir+'/wire.jsonl','');gate=new Gate();incomplete=true;await send(false);await new Promise(r=>setTimeout(r,30));
 assert(gate.stopped?.includes('incomplete native SSE'));await assert.rejects(send(false));assert.equal(calls,21);
 writeFileSync(new URL('wire-test.json',import.meta.url),JSON.stringify({kind:'MODEL_FREE_RESPONSE_FIXTURES_NOT_NATIVE_PROTOCOL_PROOF',pass:true,checks:['existing collector gates before transport at nineteen ordinary plus one summary','twenty-first request never reaches transport','incomplete SSE marks sticky uncertainty and next request is blocked'],actual_network_calls:0},null,2)+'\n');
 console.log('PASS: wire budget and incomplete-response stop, no actual network');
}finally{rmSync(dir,{recursive:true,force:true});}
