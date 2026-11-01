// Offline wire validation with the standard parser already shipped by Harness.
// Does not modify Harness or send another inference request.
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const require = createRequire(path.resolve(process.argv[2], 'package.json'));
const { createParser } = require('eventsource-parser');
const root = path.dirname(new URL(import.meta.url).pathname);
function parse(name) {
  const events = [], comments = [];
  const parser = createParser({onEvent: e => events.push({event:e.event, data:e.data}),
    onComment: c => comments.push(c), onError: error => {throw error;}});
  const wire = fs.readFileSync(path.join(root, name), 'utf8');
  // Deliberately split inside comment, JSON and event boundaries.
  for(let offset=0; offset<wire.length; offset+=7) parser.feed(wire.slice(offset, offset+7));
  return {events, comments};
}
const controlled = parse('controlled.sse'), semantic = parse('controlled-semantic.sse');
assert.deepEqual(controlled.events, semantic.events);
assert(controlled.comments.length >= 6);
assert.equal(semantic.comments.length, 0);
const native = ['tool', 'continuation'].map(name => {
  const parsed = parse(name+'.sse');
  const events = parsed.events.map(e => JSON.parse(e.data));
  assert.equal(events[0].type, 'message_start');
  assert.equal(events.at(-1).type, 'message_stop');
  assert(events.every(e => e.type !== 'keepalive'));
  const response = JSON.parse(fs.readFileSync(path.join(root, name+'-response.json')));
  // Test-only wire projection compared to the SAME native canonical response.
  // Tool IDs/arguments and text must agree with the canonical accumulator.
  const blocks = [];
  const argumentsByIndex = new Map();
  for(const event of events) {
    if(event.type === 'content_block_start') blocks[event.index] = structuredClone(event.content_block);
    if(event.type === 'content_block_delta') {
      const block = blocks[event.index], delta = event.delta;
      if(delta.type === 'text_delta') block.text += delta.text;
      else if(delta.type === 'input_json_delta') argumentsByIndex.set(event.index,
        (argumentsByIndex.get(event.index) ?? '') + delta.partial_json);
      else throw new Error('Unexpected delta in nonthinking acceptance: '+delta.type);
    }
  }
  for(const [index, text] of argumentsByIndex) blocks[index].input = JSON.parse(text);
  assert.deepEqual(blocks, response.content);
  return {name, semanticEvents:events.length, comments:parsed.comments.length,
    canonicalContentMatches:true};
});
console.log(JSON.stringify({status:'PASS', parser:'eventsource-parser',
  packageVersion:require('eventsource-parser/package.json').version,
  controlled:{semanticEvents:controlled.events.length, comments:controlled.comments.length,
    eventsIdentical:true}, native}, null, 2));
