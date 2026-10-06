const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const {webcrypto} = require('node:crypto');
const source = fs.readFileSync('ds41f_mlx/web_static/platform.js', 'utf8');
function platform(navigator) {
  const context = {window: {}, navigator, crypto: {getRandomValues: webcrypto.getRandomValues.bind(webcrypto)}, setTimeout, clearTimeout};
  vm.runInNewContext(source, context);
  return context.window.ChatPlatform;
}
test('HTTP fallback UUIDs use cryptographic v4 randomness', () => {
  const api = platform({});
  const ids = Array.from({length: 1000}, () => api.uuid());
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.every(id => /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id)));
});
test('secure-origin Web Locks remain ifAvailable with no fallback/stealing', async () => {
  let called = false;
  const api = platform({locks: {request: async (name, options, callback) => {
    assert.equal(name, 'conversation'); assert.equal(options.ifAvailable, true);
    return callback(null);
  }}});
  await assert.rejects(api.withLock('conversation', () => { called = true; }), /Another browser tab/);
  assert.equal(called, false);
});
test('secure-origin lock keeps the complete asynchronous action', async () => {
  const events = [];
  const api = platform({locks: {request: async (_, __, callback) => {
    events.push('acquire'); const value = await callback({}); events.push('release'); return value;
  }}});
  assert.equal(await api.withLock('conversation', async () => {
    events.push('begin'); await new Promise(resolve => setTimeout(resolve, 10)); events.push('end'); return 42;
  }), 42);
  assert.deepEqual(events, ['acquire', 'begin', 'end', 'release']);
});
