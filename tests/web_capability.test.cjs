const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {execFileSync} = require('node:child_process');
const policy = require('../ds41f_mlx/web_static/policy.js');
const contract = JSON.parse(execFileSync('.venv/bin/python', ['-c', 'import json; from ds41f_mlx.serving.generation_policy import capability_contract; print(json.dumps(capability_contract()))']));
const runtime = () => ({status: 'ready', capabilities: {stateful_chat: 'production-sessions-v1'}, generation_policy: structuredClone(contract), chat_defaults: {...contract.defaults, reasoning_effort: 'none', max_tokens: 'auto'}});
const source = fs.readFileSync('ds41f_mlx/web_static/app.js', 'utf8');
function context(r = runtime()) {
  const nodes = {};
  const c = vm.createContext({state: {runtime: r}, GenerationPolicy: policy, $: id => nodes[id] ||= {get value() { return this._value ?? ''; }, set value(v) { this._value = String(v); }}});
  vm.runInContext(source.slice(source.indexOf('function production()'), source.indexOf('async function createConversation()')), c);
  vm.runInContext(source.slice(source.indexOf('function settings()'), source.indexOf('async function generate()')), c);
  c.pendingCalls = () => [];
  vm.runInContext(source.slice(source.indexOf('function policyDisplay()'), source.indexOf('function controls()')), c);
  vm.runInContext("$('reasoning'); applyDefaults(negotiate())", c);
  c.state.session = {generation: {...r.chat_defaults}, policyVersion: contract.version, messages: []};
  return {c, nodes};
}
test('Browser renders authoritative defaults and serializes all eight controls', () => {
  const r = runtime(); r.generation_policy.defaults.temperature = r.chat_defaults.temperature = .7;
  const {c, nodes} = context(r);
  assert.equal(nodes.policy_temperature.value, '0.7');
  nodes.policy_top_k.value = '40'; nodes.policy_min_p.value = '.05';
  const request = JSON.parse(vm.runInContext('JSON.stringify(generationRequest())', c));
  assert.equal(request.top_k, 40); assert.equal(request.min_p, .05);
  for (const name of policy.parameters) assert.ok(name in request);
});
test('Browser reasoning locks, sampling stays request-owned; continuation snapshots survive recovery', () => {
  const {c, nodes} = context();
  nodes.thinking.value = 'on'; nodes.reasoning.value = 'low';
  vm.runInContext('state.session.workflow = settings(); state.session.protocol = {reasoning: settings().reasoning_effort, tools: []}; generationRequest()', c);
  nodes.policy_temperature.value = '.6';
  assert.equal(vm.runInContext('generationRequest().temperature', c), contract.defaults.temperature);
  vm.runInContext('state.session.generation = settings(); state.busy = true', c);
  assert.match(vm.runInContext('policyDisplay()', c), /Effective workflow:.*Next user turn:/);
  vm.runInContext('state.busy = false', c);
  assert.match(vm.runInContext('policyDisplay()', c), /Next user turn:.*Continuation snapshot:/);
  vm.runInContext('state.session.workflow = settings()', c);
  assert.equal(vm.runInContext('generationRequest().temperature', c), .6);
  nodes.reasoning.value = 'high';
  assert.throws(() => vm.runInContext('settings()', c), /session-owned/);
  nodes.reasoning.value = 'low';
  c.state.session = JSON.parse(JSON.stringify(c.state.session)); // durable/reconnect journal
  assert.equal(vm.runInContext('generationRequest().reasoning_effort', c), 'low');
  assert.equal(vm.runInContext('generationRequest().temperature', c), .6);
});
test('Browser rejects unknown contract/capability, disagreement and invalid ranges', () => {
  for (const mutate of [r => delete r.generation_policy, r => r.generation_policy.version = 'future', r => r.capabilities.stateful_chat = 'future', r => r.chat_defaults.temperature += .1, r => r.generation_policy.reasoning.change = 'allow', r => r.generation_policy.parameters.future = {}]) {
    const r = runtime(); mutate(r); assert.throws(() => policy.negotiate(r), /Unsupported|disagree/);
  }
  for (const name of policy.parameters) {
    const spec = contract.parameters[name];
    if ('maximum' in spec) assert.throws(() => policy.validate(contract, {...runtime().chat_defaults, [name]: spec.maximum + 1}), /range/);
    if ('exclusiveMinimum' in spec) assert.throws(() => policy.validate(contract, {...runtime().chat_defaults, [name]: spec.exclusiveMinimum}), /range/);
  }
});
