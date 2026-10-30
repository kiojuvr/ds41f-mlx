// Read-only installed official code; no network or attachment publication.
import { pathToFileURL } from 'node:url';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
const root = '/Users/kioju/.local/share/deepseek-harness/node_modules/@deepseek-ai';
const load = name => import(pathToFileURL(`${root}/${name}/lib/index.js`));
const a = await load('dsh-attachment-local');
const d = await load('dsh-llm-deepseek');
const { createRequire } = await import('node:module');
const require = createRequire(`${root}/dsh-attachment-local/lib/index.js`);
const sharp = require('sharp');
const data = await sharp({create:{width:2600,height:2000,channels:3,background:'red'}}).png().toBuffer();
const limits = {maxImageBytes:a.DEFAULT_MAX_IMAGE_BYTES,maxImagePixels:a.DEFAULT_MAX_IMAGE_PIXELS,maxImageDimension:a.DEFAULT_MAX_IMAGE_DIMENSION};
const policy = {maxPixels:a.DEFAULT_NORMALIZED_IMAGE_MAX_PIXELS,maxBytes:a.DEFAULT_NORMALIZED_IMAGE_MAX_BYTES,maxDimension:a.DEFAULT_NORMALIZED_IMAGE_MAX_DIMENSION};
const input = {data, mediaType:'image/png'};
const normal = await a.prepareImageFile(input, limits, policy);
assert(normal.ref.width * normal.ref.height <= 4194304);
const configured = await a.prepareImageFile(input, limits, {...policy,maxPixels:6000000});
assert.equal(configured.ref.width * configured.ref.height, 5200000);
const defaultTarget = d.resolveRequestImageTarget({},configured.ref);
const explicitTarget = d.resolveRequestImageTarget({imagePixelBudget:6000000},configured.ref);
assert(defaultTarget.width * defaultTarget.height <= 4194304);
assert(explicitTarget.width * explicitTarget.height > 4194304);
console.log(JSON.stringify({version:JSON.parse(readFileSync(`${root}/dsh/package.json`)).version,
  sha256:Object.fromEntries(['dsh-attachment-local','dsh-attachment','dsh-llm-deepseek'].map(n=>[n,createHash('sha256').update(readFileSync(`${root}/${n}/lib/index.js`)).digest('hex')])),
  limits,policy,input:{width:2600,height:2000,bytes:data.length},
  defaultNormalized:normal.ref,configuredNormalized:configured.ref,defaultTarget,explicitTarget,
  result:'PASS; larger valid output requires explicit supported configuration, not defaults'},null,2));
