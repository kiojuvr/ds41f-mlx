// Read-only installed composition check; never mounts a runner or invokes a command.
import {createRequire} from 'node:module';
import {readFileSync, writeFileSync} from 'node:fs';
import {dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const install=process.env.HOME+'/.local/share/deepseek-harness';
const require=createRequire(install+'/package.json');
const YAML=require('yaml');
const out=dirname(fileURLToPath(import.meta.url));
const tree=YAML.parse(readFileSync(out+'/composed-config.yml','utf8'));
const {resolveAdapterOptions}=await import(install+'/node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js');
const byId=id=>{const row=tree.find(x=>x.id===id);if(!row||row.disabled)throw Error('missing '+id);return row;};
const provider=resolveAdapterOptions(byId('llm-deepseek').config);
for(const id of ['commands','command-compact','compaction-basic','token-meter','headless-runner','tool-fs','attachment-local'])byId(id);
if(provider.maxImagesPerRequest!==3||provider.imageOffloadCountQuantum!==1)throw Error('changed provider budget');
const config=byId('compaction-basic').config??{};
if(Object.keys(config).length)throw Error('unexpected compaction overrides');
const result={
  provider,
  plugins:['commands','command-compact','compaction-basic','token-meter','headless-runner','tool-fs','attachment-local'].map(byId),
  compactionExplicitConfiguration:config,
  compactionDefaultsSource:'installed dsh-compaction-basic/lib/index.js: resolveConfig (source inspection, not engine execution)',
  summarizerMaxTokensDefault:65536,
  manualRetentionTokens:0,
  manualRetentionSource:'compactNow passes literal 0, independent of automatic retainRatio',
  commandRegisteredInComposition:true,
  commandDispatchExposedByHeadless:false,
  executionAuthorized:false,
  reason:'headless task positional/stdin always calls agent.followup(createUserMessage); no human-command adapter'
};
writeFileSync(out+'/effective-configuration.json',JSON.stringify(result,null,2)+'\n');
