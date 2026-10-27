// Actual normal composed config + verbatim installed policy and safe-boundary checks.
import {createRequire} from 'node:module';
import {readFileSync,writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {boundary} from './guard.mjs';
const install=process.env.HOME+'/.local/share/deepseek-harness';
const require=createRequire(install+'/package.json');
const load=n=>import(install+'/node_modules/@deepseek-ai/'+n+'/lib/index.js');
const {deepFreeze}=await load('dsh-util-values');
const {resolveAdapterOptions,DeepSeekAdapter}=await load('dsh-llm-deepseek');
const out=new URL('.',import.meta.url);
const tree=require('yaml').parse(readFileSync(new URL('composed-config.yml',out),'utf8'));
const row=id=>{const r=tree.find(x=>x.id===id);assert(r&&!r.disabled,id);return r;};
for(const id of ['headless-runner','token-meter','compaction-basic','protocol-budget'])row(id);
assert.equal(row('tool-fs').config.readImage,true);assert.equal(row('tool-fs').config.read,false);
assert.equal(row('session-log-deepseek').config.enabled,false);
const provider=resolveAdapterOptions(row('llm-deepseek').config);
assert.equal(provider.maxTokens,1200);assert.equal(provider.models[0].contextWindow,1048576);
const src=readFileSync(install+'/node_modules/@deepseek-ai/dsh-compaction-basic/lib/index.js','utf8');
const funcs=new Function('deepFreeze',src.slice(src.indexOf('const DEFAULT_THRESHOLD_RATIO'),src.indexOf('//#endregion'))+';return {resolveConfig,resolveTargetPolicy,resolveCompactSpec};')(deepFreeze);
const target={provider:'deepseek-official',model:'deepseek-flash'};
const config=funcs.resolveConfig(row('compaction-basic').config);
const policy=funcs.resolveTargetPolicy(config,target);
const adapter=new DeepSeekAdapter({options:()=>provider,resolveFiles:()=>({})});
const info=await adapter.resolveModel(target.provider,target.model);
const spec=funcs.resolveCompactSpec(policy,info.context.contextWindow,info.defaultMaxTokens);
assert.deepEqual([spec.thresholdTokens,spec.retainTokens,spec.maxTokens,spec.compactionRetries],[4096,1024,2048,0]);
assert.equal(config.auto,true);assert.match(src,/ctx.on\("agent\/pre-step"/);
const {Context}=await load('cordis');const ctx=new Context();
const checks=[];
try{
 for(const [name,before,after,pressure,expected,expectedCalls] of [
  ['first eligible hook',[],[],4096,'enter',1],
  ['failed summary returns safe reject after listener',[],[{type:'compaction/start'},{type:'compaction/end',data:{error:'failed'}}],5000,'reject',1],
  ['later hook after failure cannot enter listener',[{type:'compaction/start'},{type:'compaction/end',data:{error:'failed'}}],null,5000,'reject',0],
  ['successful summary but residual pressure cannot enter listener',[{type:'compaction/start'},{type:'compaction/summary',seq:2},{surfaceOp:{op:'replace'},sourceEventSeqs:[2]},{type:'compaction/end',data:{}}],null,4096,'reject',0],
  ['successful summary with safe pressure permits continuation',[{type:'compaction/start'},{type:'compaction/summary',seq:2},{surfaceOp:{op:'replace'},sourceEventSeqs:[2]},{type:'compaction/end',data:{}}],null,2500,'enter',1],
  ['incomplete transaction cannot continue',[{type:'compaction/start'},{type:'compaction/end',data:{}}],null,2000,'reject',0],
  ['unclosed transaction cannot continue',[{type:'compaction/start'}],null,2000,'reject',0],
 ]){
  let events=before,calls=0;
  const stop=ctx.on('test/boundary',(_,next)=>boundary(next,()=>events,()=>pressure,()=>{}),{prepend:true});
  const engine=ctx.on('test/boundary',(_,next)=>{calls++;if(after)events=after;return next();});
  const result=await ctx.waterfall('test/boundary',{},()=>Promise.resolve({kind:'enter'}));
  assert.equal(result.kind,expected,name);assert.equal(calls,expectedCalls,name);checks.push(name);stop();engine();
 }
}finally{await ctx.fiber.dispose();}
writeFileSync(new URL('effective-configuration.json',out),JSON.stringify({provider,config,policy,info,spec,guardChecks:checks,priorEventProof:'phase3-step3br1-evidence/result.json',weightsLoaded:0},null,2)+'\n');
const enlarged=resolveAdapterOptions({...row('llm-deepseek').config,maxTokens:2048});
assert.equal(enlarged.maxTokens,2048);
const prior=JSON.parse(readFileSync(new URL('../phase3-step3br2-evidence/compaction-result.json',out),'utf8'));
const partial=prior.summary_response.blocks;
const meterSource=readFileSync(install+'/node_modules/@deepseek-ai/dsh-token-meter/lib/index.js','utf8');
const estimate=new Function(meterSource.slice(meterSource.indexOf('const CHARS_PER_TOKEN'),meterSource.indexOf('function estimateToolsTokens'))+';return estimateMessage;')();
const frame=new Function(src.slice(src.indexOf('const SUMMARY_OPEN_TAG'),src.indexOf('/**\n* Run the default cache-reusing'))+src.slice(src.indexOf('function frameSummary('),src.indexOf('function finishError('))+';return frameSummary;')();
const partialPrice=estimate({role:'user',content:frame(partial)});
const framing=estimate({role:'user',content:frame([])});
const text=partial.map(b=>b.text).join('');
assert(text.endsWith('## Errors and Fixes'));
const extrapolated=framing+Math.ceil((partialPrice-framing)*2047/512);
assert(extrapolated<prior.selected_route_tokens);
writeFileSync(new URL('summary-budget-preflight.json',out),JSON.stringify({weights_loaded:0,provider_accepts_2048:true,summary_cap:2048,ordinary_cap:1200,total_output_reservation:12848,prior_selected_route_tokens:prior.selected_route_tokens,prior_partial_output_tokens:512,prior_partial_characters:text.length,partial_framed_meter_tokens:partialPrice,framing_meter_tokens:framing,unfinished_structure:'Errors and Fixes heading unfinished; Pending Jobs, Current Work, Next Step and Critical Context absent',plausible_2047_token_price_at_observed_density:extrapolated,not_a_prediction_or_shrink_proof:true,actual_installed_strict_shrink_required:true},null,2)+'\n');
console.log('PASS: actual normal CLI config/policy resolution; five supported waterfall safe-stop checks; provider 2048 and framed price plausibility');
