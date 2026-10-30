// Uses the unmodified installed official client/store over real loopback HTTP.
// The injected index is process-local so verification never writes Harness state.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const [root, inputPath] = process.argv.slice(2);
const input = JSON.parse(readFileSync(inputPath));
const { DeepSeekFileStore, DeepSeekFilesClient } = await import(pathToFileURL(
  `${root}/@deepseek-ai/dsh-llm-deepseek/lib/index.js`));
const events = [];
const cancellation = new AbortController();
const transport = async (url, init) => {
  const response = await fetch(url, init);
  events.push({ method: init.method, path: new URL(url).pathname + new URL(url).search,
    status: response.status });
  if (init.method === 'GET' && input.cancelDuringList) cancellation.abort(new Error('cancel during recovery'));
  if (init.method === 'GET' && input.raceDelete) {
    await fetch(`${input.url}/files/${input.raceDelete}`, { method: 'DELETE' });
  }
  return response;
};
const connection = { baseURL: input.url, headers: {} };
const client = new DeepSeekFilesClient({ ...connection, fetch: transport });
const store = new DeepSeekFileStore({ fetch: transport, index: {
  get: async () => undefined,
  commit: async record => ({ accepted: true, record }),
} });
const data = new Uint8Array(readFileSync(input.image));
const version = { data, bytes: data.length, mediaType: 'image/png',
  attachment: { attachmentId: 'sha256:0123456789abcdef' }, variantId: 'sha256:abcdef0123456789' };
const policy = { expiresAfterSeconds: 3600, refreshMarginSeconds: 0, quotaCleanupBatch: input.batch ?? 1 };
if (input.fill) {
  // Exhaust the real default 64 MiB retained-allocation budget, not a count cap.
  for (let i = 0; i < 10; i++) {
    try {
      await client.upload({ data, mediaType: 'image/png', filename: `dsh-fill-${i}.png`, expiresAfterSeconds: 3600 });
    } catch (error) {
      assert.equal(error.failure.status, 429, String(error) + JSON.stringify(error));
      assert.match(error.detail, /quota|storage/iu);
      break;
    }
    assert.ok(i < 9, 'finite quota exhaustion bound');
  }
  events.length = 0;
}
if (input.cancel) {
  const abort = new AbortController();
  abort.abort(new Error('verification cancelled'));
  assert.throws(() => store.ensureUploaded(version, connection, policy, abort.signal), /verification cancelled/);
  await assert.rejects(client.list({ signal: abort.signal }));
  assert.equal(events.length, 0);
} else {
  let result;
  try { result = await store.ensureUploaded(version, connection, policy, cancellation.signal); }
  catch (error) {
    assert.equal(input.success, false, String(error) + JSON.stringify(error));
    if (input.cancelDuringList) assert.match(String(error), /cancel during recovery/);
    else assert.equal(error.failure.status, input.raceDelete ? 404 : 429);
  }
  if (input.success) assert.equal(result.uploaded, true);
  else assert.equal(result, undefined);
  const uploads = events.filter(e => e.method === 'POST');
  assert.equal(uploads[0].status, 429);
  assert.equal(uploads.length, input.attempts ?? 2);
  const lists = events.filter(e => e.method === 'GET');
  assert.ok(lists.length >= (input.pages ?? 1));
  assert.ok(lists.every(e => new URL(e.path, input.url).searchParams.get('limit') === '1000'));
  if (input.pages > 1) assert.ok(lists.slice(1).every(e => e.path.includes('after_id=')));
  const deleted = events.filter(e => e.method === 'DELETE').map(e => e.path.split('/').at(-1));
  if (input.deleted) assert.deepEqual(deleted, input.deleted);
  else assert.equal(deleted.length, 1);
  assert.ok(events.filter(e => e.method === 'DELETE').every(e => e.status === (input.raceDelete ? 404 : 200)));
}
console.log(JSON.stringify({ scenario: input.name, events, outcome: 'PASS' }, null, 2));
