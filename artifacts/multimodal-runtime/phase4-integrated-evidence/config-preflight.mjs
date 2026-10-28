// Installed pure policy resolution and real official read_image declaration; no Session.
import {readFileSync,writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const install=process.env.HOME+'/.local/share/deepseek-harness';
const require=createRequire(install+'/package.json');
const load=n=>import('file://'+install+'/node_modules/@deepseek-ai/'+n+'/lib/index.js');
const tree=require('yaml').parse(readFileSync(new URL('composed-config.yml',import.meta.url),'utf8'));
const row=id=>{const r=tree.find(x=>x.id===id);assert(r&&!r.disabled,id);return r;};
for(const id of ['headless-runner','token-meter','compaction-basic','phase4-budget','phase4-native-passive'])row(id);
const {resolveAdapterOptions,DeepSeekAdapter}=await load('dsh-llm-deepseek');
const {deepFreeze}=await load('dsh-util-values');
const provider=resolveAdapterOptions(row('llm-deepseek').config);
assert.equal(provider.maxImagesPerRequest,3);assert.equal(provider.imageOffloadCountQuantum,1);assert.equal(provider.maxTokens,1200);assert.equal(provider.retryPolicy.maxRetries,0);assert.equal(provider.models[0].contextWindow,1048576);
const src=readFileSync(install+'/node_modules/@deepseek-ai/dsh-compaction-basic/lib/index.js','utf8');
const funcs=new Function('deepFreeze',src.slice(src.indexOf('const DEFAULT_THRESHOLD_RATIO'),src.indexOf('//#endregion'))+';return {resolveConfig,resolveTargetPolicy,resolveCompactSpec};')(deepFreeze);
const target={provider:'deepseek-official',model:'deepseek-flash'};
const config=funcs.resolveConfig(row('compaction-basic').config);const policy=funcs.resolveTargetPolicy(config,target);
const adapter=new DeepSeekAdapter({options:()=>provider,resolveFiles:()=>({})});const info=await adapter.resolveModel(target.provider,target.model);const spec=funcs.resolveCompactSpec(policy,info.context.contextWindow,info.defaultMaxTokens);
assert.deepEqual([spec.thresholdTokens,spec.retainTokens,spec.maxTokens,spec.compactionRetries],[8192,1024,2048,0]);assert.equal(config.auto,true);
assert.equal(row('tool-fs').config.readImage,true);assert.equal(row('tool-fs').config.read,false);assert.equal(row('session-log-deepseek').config.enabled,false);
const fsSource=readFileSync(install+'/node_modules/@deepseek-ai/dsh-tool-fs/lib/index.js','utf8');
const {defineTool}=await load('dsh-tools');
const schemaSource=fsSource.slice(fsSource.indexOf('const IMAGE_VALUE_SCHEMA ='),fsSource.indexOf('/**\n* Map a model-supplied path'));
const functionSource=fsSource.slice(fsSource.indexOf('function applyReadImageTool(ctx)'),fsSource.indexOf('//#endregion',fsSource.indexOf('function applyReadImageTool(ctx)')));
let definition;
new Function('defineTool','ctx',schemaSource+functionSource+';applyReadImageTool(ctx);')(defineTool,{tools:{register:d=>definition=d}});
assert.equal(definition.name,'read_image');assert.equal(definition.parameters.properties.file_path.type,'string');assert(definition.parameters.required.includes('file_path'));
// Registration only: the execute function is never called by this preflight.
const {Context}=await load('cordis');const ctx=new Context();
const modules=await Promise.all(['dsh-llm','dsh-session','dsh-session-projection','dsh-system-prompt','dsh-tools','dsh-agent','dsh-token-meter'].map(load));
try{
 for(const m of modules)await ctx.plugin(m.default);
 ctx.tools.register(definition);
 await ctx.plugin(await import(new URL('../phase4-step4ar1-evidence/guard.mjs',import.meta.url)));
 await ctx.plugin(await import(new URL('./passive.mjs',import.meta.url)));
 assert(globalThis.phase4Gate);
}finally{await ctx.fiber.dispose();}
writeFileSync(new URL('effective-configuration.json',import.meta.url),JSON.stringify({provider,config,policy,info,spec,tool:{name:definition.name,description:definition.description,parameters:definition.parameters},guard_registered:true,passive_registered:true,weights_loaded:0,sessions_created:0,tool_effects:0},null,2)+'\n');
console.log('PASS: exact installed policy; real read_image declaration; task guard and passive collector registered; no Session/effect');
