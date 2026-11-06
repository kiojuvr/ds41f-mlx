// Read-only check of official persisted bytes against passive application events.
import {readFileSync,writeFileSync,readdirSync} from 'node:fs';
import {zstdDecompressSync} from 'node:zlib';
import assert from 'node:assert/strict';
const out='/Volumes/SDXC-512/ds41f-mlx/artifacts/harness-rc1-compatibility';
const sid='session-1066f8e2-5b35-4dac-b7d8-08bd0d7e2be1';
const path=process.argv[2];
const records=zstdDecompressSync(readFileSync(path)).toString().trim().split('\n').map(l=>JSON.parse(l));
const live=readFileSync(out+'/harness-events.jsonl','utf8').trim().split('\n').map(l=>JSON.parse(l)).filter(r=>r.kind==='session_event'&&r.session_id===sid).map(r=>r.event);
assert.equal(records[0].id,sid);
const events=records.slice(1);
for(let i=0;i<events.length;i++)assert.equal(events[i].seq,i);
for(const event of live)assert.deepEqual(events[event.seq],event);
const passiveMissing=events.filter(e=>!live.some(x=>x.seq===e.seq));
// Official restore adds the seed-end sentinel without a live session/event emission.
assert.equal(passiveMissing.length,3);assert.ok(passiveMissing.every(e=>e.type==='session/end-seed'));
writeFileSync(out+'/persistence.json',JSON.stringify({pass:true,session_id:sid,event_count:events.length,passive_events:live.length,passive_missing:passiveMissing,header:records[0],all_passive_events_equal_persisted:true},null,2)+'\n');
console.log('PASS persisted official Session:',live.length,'exact passive events;',events.length,'total including seed-end sentinel');
