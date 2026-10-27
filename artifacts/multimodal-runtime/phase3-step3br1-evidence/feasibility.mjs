// Model-free controls only. No HTTP, weights, successful summary or replacement.
// Service mounting follows pinned official agent-loop-testkit/src/index.ts.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const install=process.env.HOME+'/.local/share/deepseek-harness';
const require=createRequire(install+'/package.json');
const load=name=>import(install+'/node_modules/@deepseek-ai/'+name+'/lib/index.js');
const {Context}=await load('cordis');
const {default:LlmRuntime,LlmAdapter,createUserMessage}=await load('dsh-llm');
const {default:SessionStore,Session,SessionId}=await load('dsh-session');
const {default:Projection}=await load('dsh-session-projection');
const {default:SystemPrompt}=await load('dsh-system-prompt');
const {default:Tools,defineContentToolFixture}=await load('dsh-tools');
const {default:Agents}=await load('dsh-agent');
const {default:AgentLoop}=await load('dsh-agent-loop');
const {default:TokenMeter}=await load('dsh-token-meter');
const {default:Basic}=await load('dsh-compaction-basic');
const {deepFreeze}=await load('dsh-util-values');
const {DeepSeekAdapter,resolveAdapterOptions}=await load('dsh-llm-deepseek');
const out=new URL('.',import.meta.url);
const prior=new URL('../phase3-step3b-evidence/',out);
const composition=require('yaml').parse(readFileSync(new URL('composed-config.yml',prior),'utf8'));
for(const id of ['headless-runner','compaction-basic','token-meter']) assert(composition.some(p=>p.id===id&&!p.disabled));
const config={auto:true,modelPolicies:[{provider:'deepseek-official',model:'deepseek-flash',thresholdRatio:1/256,headroomTokens:2048,retainTokens:1024,maxTokens:512,compactionRetries:0}]};
// Execute the verbatim installed pure policy region, not a reimplementation.
const source=readFileSync(install+'/node_modules/@deepseek-ai/dsh-compaction-basic/lib/index.js','utf8');
const policySource=source.slice(source.indexOf('const DEFAULT_THRESHOLD_RATIO'),source.indexOf('//#endregion'));
const policy=new Function('deepFreeze',policySource+';return {resolveConfig,resolveTargetPolicy,resolveCompactSpec};')(deepFreeze);
const resolved=policy.resolveConfig(config);
const target={provider:'deepseek-official',model:'deepseek-flash'};
const options=resolveAdapterOptions(composition.find(p=>p.id==='llm-deepseek').config);
const actualAdapter=new DeepSeekAdapter({options:()=>options,resolveFiles:()=>({})});
const info=await actualAdapter.resolveModel(target.provider,target.model);
const targetPolicy=policy.resolveTargetPolicy(resolved,target);
const spec=policy.resolveCompactSpec(targetPolicy,info.context.contextWindow,info.defaultMaxTokens);
assert.equal(info.context.contextWindow,1048576); assert.equal(info.defaultMaxTokens,1200);
assert.equal(spec.thresholdTokens,4096); assert.equal(spec.retainTokens,1024); assert(spec.retainTokens<spec.thresholdTokens);
assert.throws(()=>policy.resolveConfig({retainRatio:.9}),/retainRatio/);
assert.throws(()=>policy.resolveCompactSpec({...targetPolicy,retainTokens:4096},1048576,1200),/retainTokens/);
assert.throws(()=>policy.resolveCompactSpec({...targetPolicy,headroomTokens:1048576},1048576,1200),/no pressure budget/);
const ctx=new Context();
const hooks=[],requests=[],summaryAttempts=[];
let toolDone=false, subject;
class ControlledAdapter extends LlmAdapter {
  async resolveModel(provider,model){return actualAdapter.resolveModel(provider,model);}
  async *stream(request){
    if(request.purpose==='compaction'){
      const selected=subject.session.surface.nodes.filter(seq=>{
        const event=subject.session.eventAt(seq);
        return event.type!=='system/message' && request.messages.slice(0,-1).some(m=>JSON.stringify(m)===JSON.stringify(subject.session.deriveEventMessage(event)));
      });
      summaryAttempts.push({purpose:request.purpose,maxTokens:request.maxTokens,messages:request.messages.length,selected,retained:subject.session.surface.nodes.filter(seq=>!selected.includes(seq))});
      throw new Error('MODEL_FREE_STOP_BEFORE_SUMMARY');
    }
    requests.push({purpose:request.purpose??'ordinary',messages:request.messages.length});
    if(requests.length===1){
      yield {type:'block-start',index:0,blockType:'tool-call'};
      yield {type:'block-end',index:0,block:{type:'tool-call',id:'controlled-call',name:'work',arguments:'{}'}};
      yield {type:'finish',reason:{kind:'tool-calls'}};
    }else{
      yield {type:'block-start',index:0,blockType:'text'};
      yield {type:'block-end',index:0,block:{type:'text',text:'Controlled response; not native evidence.'}};
      yield {type:'finish',reason:{kind:'stop'}};
    }
  }
}
try{
  for(const plugin of [LlmRuntime,SessionStore,Projection]) await ctx.plugin(plugin);
  await ctx.plugin(SystemPrompt,{includeHarnessIdentity:false,includeRuntimeContext:false});
  await ctx.plugin(Tools); await ctx.plugin(Agents); await ctx.plugin(TokenMeter);
  ctx.llm.registerAdapter([target.provider],new ControlledAdapter());
  ctx.tools.register(defineContentToolFixture({name:'work',description:'Controlled no-effect tool.',parameters:{},async execute(){toolDone=true;return [{type:'text',text:'Controlled tool output.'}];}}));
  await ctx.plugin(Basic,config);
  // Passive observations only. The engine's actual event listener and methods stay unchanged.
  ctx.on('agent/pre-step',({agent,turn,step},next)=>{
    hooks.push({turn,step,header:agent.session.requestHeader()?.config??null,toolDone,measurement:ctx.tokenMeter.measure(agent.session),surface:[...agent.session.surface.nodes]});
    return next();
  });
  await ctx.plugin(AgentLoop,{agents:[]});
  const agent=subject=await ctx.agentLoop.create(SessionId('model-free-control'),target);
  const follow=async text=>{agent.followup(createUserMessage({content:[{type:'text',text}],source:{kind:'user'}}));await agent.whenIdle();};
  await follow('Use the controlled work tool once, then finish.');
  assert.equal(hooks[0].header,null);
  assert.equal(hooks[1].header.provider,target.provider);
  assert(hooks[1].toolDone); assert.equal(summaryAttempts.length,0);
  const completed=Session.create(SessionId('snapshot-control'),agent.session.snapshotEvents());
  assert.equal(completed.requestHeader().config.model,target.model);
  for(let part=0;part<3;part++){
    await follow('Controlled user document section '+part+': '+('Control-only ordinary user input paragraph. '.repeat(150)));
    assert.equal(summaryAttempts.length,0); // pending user text was not yet admitted at pre-step
  }
  const before=ctx.tokenMeter.measure(agent.session);
  assert(before.totalTokens>=4096);
  await follow('Continue with one short conclusion.');
  assert.equal(summaryAttempts.length,1); assert.equal(summaryAttempts[0].maxTokens,512);
  const events=agent.session.snapshotEvents();
  const start=events.find(e=>e.type==='compaction/start');
  const end=events.find(e=>e.type==='compaction/end');
  assert(start&&end); assert.match(end.data.error,/MODEL_FREE_STOP_BEFORE_SUMMARY/);
  assert(!events.some(e=>e.type==='compaction/summary'||e.surfaceOp?.op==='replace'));
  assert.equal(events.at(-1).type,'turn/end'); assert.equal(events.at(-1).data.reason.kind,'completed');
  const selected=summaryAttempts[0].selected;
  const system=events.find(e=>e.type==='system/message').seq;
  const call=events.find(e=>e.type==='assistant/message'&&e.data.message.content.some(b=>b.type==='tool-call')).seq;
  const toolResult=events.find(e=>e.type==='tool/result').seq;
  assert(!selected.includes(system)); assert(selected.includes(call)); assert(selected.includes(toolResult));
  assert(summaryAttempts[0].retained.length>=2);
  const selectedRouteTokens=before.nodes.filter(n=>selected.includes(n.seq)).reduce((n,node)=>n+node.tokens,0);
  assert(selectedRouteTokens>spec.maxTokens+512); // real selected history, not just a tiny first exchange
  await follow('One further controlled continuation to check absence of pressure-failure cooldown.');
  assert.equal(summaryAttempts.length,2); // each later eligible hook retries; no successful replacement
  assert(!agent.session.snapshotEvents().some(e=>e.type==='compaction/summary'||e.surfaceOp?.op==='replace'));
  // Reaching stream proves real range validation accepted both boundaries,
  // protected system head and constructed a real compaction request.
  const result={kind:'MODEL_FREE_CONTROLS_NOT_NATIVE_SESSION_EVIDENCE',config,resolved,targetPolicy,adapterInfo:info,
    messageBudgetTokens:info.context.contextWindow-info.defaultMaxTokens,pressureBudgetTokens:info.context.contextWindow-info.defaultMaxTokens-targetPolicy.headroomTokens,
    spec,compositionMountedInRetainedNormalProfile:true,policySourceSha256:createHash('sha256').update(policySource).digest('hex'),
    hooks,requests,summaryAttempts,completedHeaderSurvivesSnapshot:completed.requestHeader(),aboveThresholdMeasurement:before,
    realCompactRegionReached:true,realSummaryRequestReached:true,summarySucceeded:false,replacement:false,
    protectedSystemSeq:system,balancedSelectedToolPair:[call,toolResult],selectedRouteTokens,pressureFailureRetriesOnLaterHook:true,
    failedAttemptClosed:true,ordinaryTurnContinued:true,
    counts:{hooks:hooks.length,ordinaryRequests:requests.length,summaryAttempts:summaryAttempts.length},
    checks:['missing initial routed header skips','below threshold skips','completed tool boundary reached','completed header snapshot persistence','pending followup not yet priced','above threshold selects actual compactRegion','real range validation and input assembly','512 summary cap','summary failure closes attempt; turn continues','no summary or replacement','protected system and balanced selected tool pair','later hook retries failed pressure compaction','invalid retention/headroom rejected']};
  writeFileSync(new URL('result.json',out),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify({PASS:result.checks,counts:result.counts,spec},null,2));
}finally{await ctx.fiber.dispose();}
