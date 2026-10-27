// Resolve the actual composed provider through the installed official validator.
import {createRequire} from 'node:module';
import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname} from 'node:path';
const require=createRequire(process.env.HOME+'/.local/share/deepseek-harness/package.json');
const {resolveAdapterOptions}=await import(process.env.HOME+'/.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js');
const YAML=require('yaml');const out=dirname(fileURLToPath(import.meta.url));
const tree=YAML.parse(readFileSync(out+'/composed-config.yml','utf8'));
const provider=tree.find(x=>x.id==='llm-deepseek');
const resolved=resolveAdapterOptions(provider.config);
if(resolved.maxImagesPerRequest!==3||resolved.imageOffloadCountQuantum!==1)throw Error('configuration mismatch');
if(resolved.maxRequestFilesBytes!==134217728||resolved.maxInlineRequestImageBytes!==20971520||resolved.imageOffloadByteQuantum!==67108864||resolved.inlineImageOffloadByteQuantum!==10485760)throw Error('byte thresholds changed');
if(!tree.some(x=>x.name==='@deepseek-ai/dsh-compaction-image-offload'&&!x.disabled))throw Error('official lifecycle plugin absent');
writeFileSync(out+'/effective-provider.json',JSON.stringify({provider:provider.name,resolved,officialOffloadPluginEnabled:true,weightsLoaded:false},null,2)+'\n');
