// Model/network-free qualification against installed official Harness code.
// Usage: node tools/check_harness_image_envelope.mjs /path/to/node_modules
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
const root = resolve(process.argv[2]);
const load = name => import(pathToFileURL(`${root}/@deepseek-ai/${name}/lib/index.js`));
const { Config } = await load('dsh-llm-deepseek');
const { requiredImageOffload, projectOffloadedImages } = await load('dsh-llm');
const { apply } = await load('dsh-compaction-image-offload');
const config = Config({});
const limits = Object.fromEntries(['maxImagesPerRequest', 'maxRequestFilesBytes',
  'maxInlineRequestImageBytes', 'imageOffloadCountQuantum'].map(k => [k, config[k].get()]));
assert.deepEqual(Object.values(limits), [600, 128*2**20, 20*2**20, 20]);
const image = { type: 'image', attachment: { attachmentId: 'same' } };
const history = n => [{ role: 'user', content: Array(n).fill(image) }];
const raw = { representation: 'raw', maxImages: limits.maxImagesPerRequest,
  maxBytes: limits.maxRequestFilesBytes, countQuantum: limits.imageOffloadCountQuantum };
assert.equal(requiredImageOffload(history(600), raw, () => 1), 0);
assert.equal(requiredImageOffload(history(601), raw, () => 1), 20);
assert.equal(requiredImageOffload(history(1), raw, () => raw.maxBytes), 0);
assert.equal(requiredImageOffload(history(1), raw, () => raw.maxBytes+1), 1);
const inline = { ...raw, representation: 'base64', maxBytes: limits.maxInlineRequestImageBytes };
assert.equal(requiredImageOffload(history(1), inline, () => inline.maxBytes/4*3), 0);
assert.equal(requiredImageOffload(history(1), inline, () => inline.maxBytes/4*3+1), 1);
// Raw Files fit does not imply inline fallback fits.
assert.equal(requiredImageOffload(history(1), raw, () => 16*2**20), 0);
assert.equal(requiredImageOffload(history(1), inline, () => 16*2**20), 1);
const listeners = {}, events = [];
let projection;
apply({ sessions: { registerMessageProjection: p => projection = p },
  on: (name, callback) => listeners[name] = callback });
const message = history(21)[0];
const source = { type: 'user/message', data: message };
const session = { surface: { nodes: [0] }, eventAt: () => source,
  deriveEventMessage: () => message, append: (type, data) => events.push({type, data}) };
const recover = failure => listeners['agent/request-error']({ agent: {session}, failure,
  signal: new AbortController().signal }, () => 'not-recovered');
assert.equal(recover({code: 'INVALID_REQUEST', status: 400}), 'not-recovered');
assert.equal(events.length, 0);
assert.deepEqual(await recover({code: 'IMAGE_OFFLOAD_REQUIRED', offloadImages: 20}), {kind:'retry'});
const marked = projection.project(events[0], { nodes: [0], events: [source], baseSeq: 0, messages: new Map() }).get(0);
const projected = projectOffloadedImages([marked], () => '[offloaded]');
assert.equal(projected[0].content.filter(b => b.type === 'image').length, 1);
assert.equal(message.content.filter(b => b.offloaded).length, 0);
const version = JSON.parse(readFileSync(`${root}/@deepseek-ai/dsh/package.json`)).version;
const sourceSha256 = Object.fromEntries(['dsh-llm', 'dsh-llm-deepseek', 'dsh-compaction-image-offload']
  .map(name => [name, createHash('sha256').update(readFileSync(`${root}/@deepseek-ai/${name}/lib/index.js`)).digest('hex')]));
console.log(JSON.stringify({version, sourceSha256, limits, boundaries: 'PASS', projection: '20 oldest occurrences become text',
  runtime400: 'not recovered by image-offload plugin'}, null, 2));
