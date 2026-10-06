// Optional renderer tests using an existing Node installation; no npm/build deps.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
class Node {
  constructor(tag, value = '') { this.tag = tag; this.value = value; this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  get firstChild() { return this.children[0]; }
  get length() { return this.value.length; }
  get data() { return this.value; }
  set data(value) { this.value = value; }
  appendData(value) { this.value += value; }
  querySelector(selector) { return descendants(this).find(n => (n.className || '').split(' ').includes(selector.slice(1))); }
  set innerHTML(_) { throw new Error('HTML parser sink forbidden'); }
}
const context = {URL, window: {}, document: {
  createElement: tag => new Node(tag), createTextNode: value => new Node('#text', value),
}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../ds41f_mlx/web_static/render.js'), 'utf8'), context);
const render = context.window.ChatRender;
const descendants = node => [node, ...node.children.flatMap(descendants)];
test('untrusted text, reasoning, code and calls stay text', () => {
  const root = new Node('main');
  const attack = '<img src=x onerror="alert(1)">';
  render.message(root, {role: 'assistant', content: attack + '\n```html\n' + attack + '\n```',
    reasoning_content: attack, tool_calls: [{id: attack, function: {name: attack, arguments: attack}}]});
  assert.equal(descendants(root).filter(n => n.tag === 'img').length, 0);
  assert.ok(descendants(root).filter(n => n.tag === '#text' && n.value.includes(attack)).length >= 4);
  assert.equal(descendants(root).filter(n => n.tag === 'pre').length, 1);
});
test('links reject active schemes, relative URLs and credentials', () => {
  for (const url of ['javascript:alert(1)', 'data:text/html,<script>', '/relative', 'https://user:pass@example.com']) {
    assert.equal(render.sourceURL(url), null);
  }
  assert.equal(render.sourceURL('https://example.com/a'), 'https://example.com/a');
  const root = new Node('main');
  render.activity(root, {tool: 'web_search', results: [
    {url: 'javascript:alert(1)', title: '<script>alert(1)</script>'},
    {url: 'https://example.com/" onclick="alert(1)', title: 'safe'},
  ]});
  const links = descendants(root).filter(n => n.tag === 'a');
  assert.equal(links.length, 1);
  assert.equal(links[0].rel, 'noopener noreferrer');
  assert.ok(links[0].href.startsWith('https://example.com/'));
  assert.equal(links[0].onclick, undefined);
});
test('tool sources survive normal transcript rendering', () => {
  const root = new Node('main');
  render.message(root, {role: 'tool', tool_call_id: 'call1', content: JSON.stringify({results: [
    {url: 'https://example.com', title: 'Example'},
  ]})});
  assert.equal(descendants(root).filter(n => n.tag === 'a').length, 1);
});
test('long unbroken backticks remain text without regex backtracking', () => {
  const root = new Node('main');
  const value = '`'.repeat(100000);
  render.message(root, {role: 'assistant', content: value});
  assert.ok(descendants(root).some(n => n.tag === '#text' && n.value === value));
});
test('images never fetch network URLs from transcript', () => {
  const root = new Node('main');
  render.message(root, {role: 'user', content: [
    {type: 'image_url', image_url: {url: 'https://example.com/tracking.png'}},
    {type: 'image_url', image_url: {url: 'data:image/png;base64,aGVsbG8='}},
    {type: 'image_url', image_url: {url: 'data:image/svg+xml;base64,aGVsbG8='}},
  ]});
  const images = descendants(root).filter(n => n.tag === 'img');
  assert.equal(images.length, 1);
  assert.equal(images[0].src, 'data:image/png;base64,aGVsbG8=');
});
test('safe Markdown blocks and inline formatting', () => {
  const root = new Node('main');
  render.prose(root, '# Heading\n\nA **bold** and *italic* [link](https://example.com) with `code`.\n\n- first\n- second\n\n1. one\n2. two\n\n> quoted\n\n---\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n```python\nprint(42)\n```');
  for (const tag of ['h1', 'p', 'strong', 'em', 'a', 'code', 'ul', 'ol', 'blockquote', 'hr', 'table', 'th', 'td', 'pre', 'button']) {
    assert.ok(descendants(root).some(n => n.tag === tag), tag);
  }
});
test('Markdown rejects executable links and keeps HTML literal', () => {
  const root = new Node('main');
  render.prose(root, '[x](javascript:alert(1)) [x](data:text/html,bad) [x](https://u:p@example.com) <script>alert(1)</script>');
  assert.equal(descendants(root).filter(n => ['a', 'script'].includes(n.tag)).length, 0);
});
test('stream updates append to stable text nodes, not whole response DOM', () => {
  const root = new Node('main');
  const node = render.message(root, {role: 'assistant', content: '', reasoning_content: ''}, [], true);
  const body = node.querySelector('.liveContent').firstChild;
  render.updateLive(node, {content: 'hello', reasoning_content: 'public'});
  render.updateLive(node, {content: 'hello world', reasoning_content: 'public reasoning'});
  assert.equal(node.querySelector('.liveContent').firstChild, body);
  assert.equal(body.value, 'hello world');
});
test('completed streaming Markdown blocks finalize once; unfinished fences stay literal', () => {
  const root = new Node('main');
  const node = render.message(root, {role: 'assistant', content: '', reasoning_content: ''}, [], true);
  render.updateLive(node, {content: '# Heading\n\n```js\nconst x = 1;\n', reasoning_content: ''});
  const blocks = node.querySelector('.liveBlocks');
  const heading = blocks.children[0];
  assert.equal(heading.tag, 'h1');
  assert.equal(descendants(blocks).filter(n => n.tag === 'pre').length, 0);
  render.updateLive(node, {content: '# Heading\n\n```js\nconst x = 1;\n```\n\nTail', reasoning_content: ''});
  assert.equal(blocks.children[0], heading);
  assert.equal(descendants(blocks).filter(n => n.tag === 'pre').length, 1);
  assert.equal(node.querySelector('.liveContent').firstChild.value, 'Tail');
});
test('deep blockquotes have bounded nesting', () => {
  const root = new Node('main');
  render.prose(root, '>'.repeat(100000) + 'literal');
  assert.ok(descendants(root).filter(n => n.tag === 'blockquote').length <= 16);
  assert.ok(descendants(root).some(n => n.tag === '#text' && n.value.endsWith('literal')));
});
test('invalid table delimiter with huge whitespace stays linear/literal', () => {
  const root = new Node('main');
  render.prose(root, 'a | b\n| --- | ---' + ' '.repeat(100000) + 'X');
  assert.equal(descendants(root).filter(n => n.tag === 'table').length, 0);
});
test('live reasoning stays hidden until public content arrives', () => {
  const root = new Node('main');
  const node = render.message(root, {role: 'assistant', content: '', reasoning_content: ''}, [], true);
  assert.equal(node.querySelector('.reasoning').hidden, true);
  render.updateLive(node, {content: 'answer', reasoning_content: 'Published field'});
  assert.equal(node.querySelector('.reasoning').hidden, false);
});
