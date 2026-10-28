// Task-only validation, reusing the exact installed translator, not a second
// Messages implementation. No official file is edited or alternate request made.
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
const require=createRequire(process.env.HOME+'/.local/share/deepseek-harness/package.json');
const path=require.resolve('@deepseek-ai/dsh-llm-deepseek');
const source=readFileSync(path,'utf8');
export const adapterIdentity={path,sha256:createHash('sha256').update(source).digest('hex')};
if(adapterIdentity.sha256!=='eba35025ed1eedda89c51f2da46a3a968f7be1d59719f2ceeadfc4518dff37d6') throw Error('Unqualified installed Messages translator');
const {LlmError,ToolCallId}=require('@deepseek-ai/dsh-llm');
const helpers=source.slice(source.indexOf('function object('),source.indexOf('/** Validate native replay,'));
const translator=source.slice(source.indexOf('function string(value)'),source.indexOf('//#region lib/types/adapter.js'));
const translate=Function('LlmError','ToolCallId',helpers+'\n'+translator+'\nreturn translate;')(LlmError,ToolCallId);
export async function validateTerminal(events,model='deepseek-flash') {
 const chunks=[];
 async function* receipt(){yield* events;}
 for await(const c of translate(receipt(),model)) chunks.push(c);
 const finish=chunks.filter(c=>c.type==='finish');
 if(finish.length!==1)throw Error('Missing unique translated finish');
 const blocks=chunks.filter(c=>c.type==='block-end').map(c=>c.block);
 // Acceptance is deliberately stricter than the translator's max_tokens escape:
 // unresolved/invalid tool JSON never qualifies a benign clone abort.
 const ids=new Set();
 for(const b of blocks)if(b.type==='tool-call') {
  if(ids.has(b.id))throw Error('Duplicate Tool Call identity');ids.add(b.id);
  const input=JSON.parse(b.arguments);
  if(!input||typeof input!=='object'||Array.isArray(input))throw Error('Tool input must be an object');
 }
 return {blocks:blocks.map(b=>b.type==='tool-call'?{type:'tool_use',id:b.id,name:b.name,input:JSON.parse(b.arguments)}:b),finish:finish[0],translated_usage:chunks.find(c=>c.type==='usage')?.usage};
}
