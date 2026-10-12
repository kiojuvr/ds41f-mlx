const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('ds41f_mlx/web_static/app.js', 'utf8');

test('an awaited file batch remains owned by its originating conversation after Switch', async () => {
  const a = {name: 'A'}, b = {name: 'B'}, owners = [];
  const c = vm.createContext({state: a, addPDF: async (file, ctx) => {
    owners.push(ctx); c.state = b; // User switches while upload awaits.
  }, attach: async (files, ctx) => owners.push(ctx)});
  vm.runInContext(source.slice(source.indexOf('async function attachFiles(files)'), source.indexOf("$('composer').ondragover")), c);
  await c.attachFiles([{name: 'a.pdf'}, {name: 'b.pdf'}, {name: 'image.png', type: 'image/png'}]);
  assert.equal(owners.length, 3);
  assert.ok(owners.every(ctx => ctx === a));
});

test('hidden conversation tool results never mutate the active conversation', () => {
  const a = {session: {messages: [], count: 1}}, b = {session: {messages: [], count: 2}};
  const c = vm.createContext({state: b});
  vm.runInContext(source.slice(source.indexOf('function appendResults('), source.indexOf('async function reconcile(')), c);
  c.appendResults({messages: [{role: 'tool', tool_call_id: 'A-call', content: 'result A'}], results: []}, a);
  assert.equal(a.session.messages.length, 1);
  assert.equal(a.session.effects.count, 1);
  assert.equal(b.session.messages.length, 0);
  assert.equal(b.session.effects, undefined);
});
