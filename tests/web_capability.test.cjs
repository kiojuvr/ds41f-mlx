const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('ds41f_mlx/web_static/app.js', 'utf8');
const functions = source.slice(source.indexOf('function production()'), source.indexOf('async function createConversation()'));
const defaults = {temperature: 0, top_p: 1, reasoning_effort: 'none', max_tokens: 'auto'};
function context(runtime) {
  const nodes = {};
  const c = vm.createContext({state: {runtime}, $: id => nodes[id] ||= {}});
  vm.runInContext(functions, c);
  vm.runInContext(source.slice(source.indexOf('function settings()'), source.indexOf('function generationRequest()')), c);
  return {c, nodes};
}
test('Browser consumes runtime authority, not profile or preferences', () => {
  const {c, nodes} = context({status: 'ready', profile: 'not-authority', capabilities: {stateful_chat: 'production-sessions-v1'}, chat_defaults: defaults});
  assert.deepEqual(JSON.parse(vm.runInContext('JSON.stringify(negotiate())', c)), defaults);
  vm.runInContext('applyDefaults(negotiate())', c);
  assert.equal(nodes.temperature.value, 0); assert.equal(nodes.topP.value, 1);
  assert.equal(nodes.thinking.value, 'off'); assert.equal(nodes.outputLimit.value, 'auto');
  c.state.session = {generation: {...defaults}};
  nodes.temperature.value = 1; nodes.topP.value = .95;
  assert.deepEqual(JSON.parse(vm.runInContext('JSON.stringify(settings())', c)), defaults);
  c.state.session = {};
  assert.throws(() => vm.runInContext('settings()', c), /lacks negotiated/);
});
test('Browser fails closed and retains explicit standard-OFF compatibility', () => {
  for (const runtime of [
    {status: 'ready', profile: 'mtp-serving-v1'},
    {status: 'ready', capabilities: {stateful_chat: 'future'}},
    {status: 'ready', capabilities: {stateful_chat: 'production-sessions-v1'}, chat_defaults: {...defaults, temperature: 1}}
  ]) {
    const {c} = context(runtime);
    assert.throws(() => vm.runInContext('negotiate()', c), /Unsupported/);
    assert.throws(() => vm.runInContext('settings()', c), /Unsupported/);
  }
  assert.equal(vm.runInContext('negotiate().temperature', context({status: 'ready', profile: 'standard-off'}).c), 1);
});
