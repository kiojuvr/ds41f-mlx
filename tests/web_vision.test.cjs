const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const policy = require('../ds41f_mlx/web_static/policy.js');
const source = fs.readFileSync('ds41f_mlx/web_static/app.js', 'utf8');
function setup(capability = 'bounded-inline-v1') {
  const nodes = {};
  const state = {runtime: {status: 'ready', capabilities: {stateful_chat: 'production-sessions-v1', chat_vision: capability}}, attachments: [], session: {messages: []}};
  let notice = '', previews = 0;
  const c = vm.createContext({state, GenerationPolicy: {...policy, negotiate() {}},
    scoped: () => ({notice: text => notice = text}), windowActive: () => state,
    preview: () => previews++, $: id => nodes[id] ||= {closest: () => ({})},
    createImageBitmap: async file => ({width: file.width || 800, height: 800, close() {}}),
    FileReader: class {readAsDataURL() {this.result = 'data:image/png;base64,ORIGINAL'; this.onload();}},
    addPDF: async () => {}, resizeInput() {}, send() {}});
  vm.runInContext(source.slice(source.indexOf('async function attach('), source.indexOf('function preview()')), c);
  vm.runInContext(source.slice(source.indexOf("$('images').onchange"), source.indexOf('function resizeInput()')), c);
  return {c, nodes, state, notice: () => notice, previews: () => previews};
}
const file = (extra = {}) => ({name: 'source.png', size: 20, arrayBuffer: async () => new Uint8Array([137,80,78,71,13,10,26,10,0,0,0,0]).buffer, ...extra});
test('MTP picker, drop and paste retain original bounded attachments and preview', async () => {
  const {c, nodes, state, previews} = setup();
  await nodes.images.onchange({target: {files: [file()]}});
  assert.equal(state.attachments.length, 1);
  await nodes.composer.ondrop({preventDefault() {}, dataTransfer: {files: [file()]}});
  nodes.input.onpaste({preventDefault() {}, clipboardData: {items: [{kind: 'file', getAsFile: () => file()}]}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(state.attachments.length, 3);
  assert.equal(state.attachments[0].url, 'data:image/png;base64,ORIGINAL');
  assert.equal(previews(), 3);
  assert.equal(state.session.messages.length, 0); // acquisition is not executable state
});
test('Missing/unknown/unready Vision fails closed; no profile-based OFF fallback', async () => {
  for (const capability of [null, 'future']) {
    const {c, state, notice} = setup(capability);
    c.files = [file()]; await vm.runInContext('attach(files)', c);
    assert.equal(state.attachments.length, 0); assert.match(notice(), /does not advertise/);
  }
  assert.equal(policy.visionAvailable({status: 'ready', profile: 'standard-off'}), false);
  assert.equal(policy.visionAvailable({status: 'alive', capabilities: {chat_vision: 'bounded-inline-v1'}}), false);
});
test('Existing count, encoded size and source geometry preflights are unchanged', async () => {
  for (const [files, expected] of [[Array(5).fill(file()), /four/], [[file({size: 16*1024*1024+1})], /16 MiB/], [[file({width: 4000})], /aspect ratio/]]) {
    const {c, state, notice} = setup(); c.files = files;
    await vm.runInContext('attach(files)', c);
    assert.equal(state.attachments.length, 0); assert.match(notice(), expected);
  }
});
