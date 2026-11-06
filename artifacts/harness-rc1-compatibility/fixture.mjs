// Disposable genuine tool-plugin registry; no request/protocol/history edits.
import {readFileSync, appendFileSync} from 'node:fs';
import {setTimeout as delay} from 'node:timers/promises';
import {defineTool} from '@deepseek-ai/dsh-tools';
import * as toolFs from '@deepseek-ai/dsh-tool-fs';
import * as toolSkill from '@deepseek-ai/dsh-tool-skill';
import * as skillFs from '@deepseek-ai/dsh-skill-filesystem';
import * as toolGoal from '@deepseek-ai/dsh-tool-goal';
export const name='alpha2-acceptance-fixture';
export const inject=['tools','skills'];
const root='/tmp/ds41f-rc1-e2e';
const log=x=>appendFileSync(process.env.ALPHA2_EVIDENCE+'/fixture.jsonl',JSON.stringify({at:new Date().toISOString(),...x})+'\n');
export function apply(ctx){
 const register=(name,file)=>ctx.tools.register(defineTool({name,description:'Read the harmless acceptance receipt file and return its exact text.',parameters:{},
  output:{schema:{type:'string'},render:(_args,value)=>[{type:'text',text:value}]},
  async execute(_args,exec){const text=readFileSync(root+'/workspace/'+file,'utf8').trim();log({kind:'effect',name,text,agent:exec.agent?.id});return text;}}));
 let baseline=register('baseline_receipt','baseline.txt'),late,full=false,slow=false,stage='baseline';
 const timer=setInterval(()=>{
  const next=readFileSync(root+'/registry-stage','utf8').trim(); if(next===stage)return;
  if(next==='add'&&!late)late=register('late_receipt','late.txt');
  if(next==='drop'&&baseline){baseline();baseline=null;}
  if(next==='full'&&!full){full=true;ctx.plugin(toolFs,{read:true,readImage:true,write:false,edit:false});ctx.plugin(skillFs);ctx.plugin(toolSkill);ctx.plugin(toolGoal);}
  if(next==='slow'&&!slow){slow=true;ctx.tools.register(defineTool({name:'slow_receipt',description:'Read a harmless receipt after a 30-second cancellable wait. Never retry automatically.',parameters:{},
   output:{schema:{type:'string'},render:(_args,value)=>[{type:'text',text:value}]},
   async execute(_args,exec){log({kind:'slow_begin'});try{await delay(30000,undefined,{signal:exec.signal});}catch(error){log({kind:'slow_cancelled',aborted:exec.signal.aborted});throw error;}
    const text=readFileSync(root+'/workspace/late.txt','utf8').trim();log({kind:'effect',name:'slow_receipt',text});return text;}}));}
  stage=next;log({kind:'registry',stage});
 },200);
 ctx.on('dispose',()=>clearInterval(timer));
 log({kind:'mounted',stage});
}
