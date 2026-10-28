// Unchanged R3 read-only official inspector, resolved against this attempt's root.
import {readFileSync} from 'node:fs';
let source=readFileSync(new URL('../phase3-step3br3-evidence/persisted.mjs',import.meta.url),'utf8');
source=source.replace("const install=process.env.HOME", "const install='file://'+process.env.HOME").replace("const out=new URL('.',import.meta.url)", 'const out=new URL('+JSON.stringify(new URL('.',import.meta.url).href)+')');
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
