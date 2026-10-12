const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const {webcrypto} = require('node:crypto');
const source = fs.readFileSync('ds41f_mlx/web_static/platform.js', 'utf8');
test('HTTP LAN identities use cryptographic v4 randomness without persistent locks', () => {
  const context = {window: {}, crypto: {getRandomValues: webcrypto.getRandomValues.bind(webcrypto)}};
  vm.runInNewContext(source, context);
  const api = context.window.ChatPlatform;
  const ids = Array.from({length: 1000}, () => api.uuid());
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.every(id => /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id)));
  assert.deepEqual(Object.keys(api), ['uuid']);
});

test('conversation store is tab memory only; a fresh page has no archive', async () => {
  const source = fs.readFileSync('ds41f_mlx/web_static/store.js', 'utf8');
  const first = {window: {}}; vm.runInNewContext(source, first);
  await first.window.ChatStore.put('sessions', {id: 'a', messages: ['hello']});
  await first.window.ChatStore.put('sessions', {id: 'b', messages: []});
  assert.equal((await first.window.ChatStore.all('sessions')).length, 2);
  const fresh = {window: {}}; vm.runInNewContext(source, fresh);
  assert.equal((await fresh.window.ChatStore.all('sessions')).length, 0);
  await assert.rejects(first.window.ChatStore.all('saves'), /archives are not supported/);
});
