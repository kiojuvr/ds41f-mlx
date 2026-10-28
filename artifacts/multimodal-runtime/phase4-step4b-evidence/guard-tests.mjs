// Re-execute unchanged qualified model-free registration checks, outputs local.
import {readFileSync} from 'node:fs';
const prior=new URL('../phase4-step4ar1-evidence/guard-tests.mjs',import.meta.url);
let source=readFileSync(prior,'utf8').replace("from './guard.mjs'",'from '+JSON.stringify(new URL('../phase4-step4ar1-evidence/guard.mjs',import.meta.url).href));
source=source.replace("const root=process.env.HOME", "const root='file://'+process.env.HOME");
source=source.replace("new URL('gate-a-tests.json',import.meta.url)",JSON.stringify(new URL('guard-registration.json',import.meta.url).pathname));
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
