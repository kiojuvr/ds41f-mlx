// Model-free probe of unmodified, pinned Harness source. No Harness session or inference.
// Usage: node --experimental-strip-types --loader ./loader.mjs projection.mjs <harness-root>
import { pathToFileURL } from 'node:url'
import assert from 'node:assert/strict'
const root = process.argv[2]
const { projectToolUpdates } = await import(pathToFileURL(`${root}/packages/llm/llm/src/content.ts`))
const search = { name: 'search', description: 'baseline', parameters: { type: 'object' } }
const fetch = { name: 'fetch', description: 'later addition', parameters: { type: 'object' } }
const user = id => ({ id, role: 'user', content: [{ type: 'text', text: id }] })
const update = (id, content) => ({ id, role: 'developer', content })
const swapped = update('swap', [{ type: 'tool-addition', toolName: 'fetch' }, { type: 'tool-removal', toolName: 'search' }])
const messages = [user('before'), swapped, user('after')]
const history = { tools: [search], updates: [{ messageId: swapped.id, additions: [fetch] }] }
const results = Object.fromEntries(['addition-only', 'in-history'].map(mode => [mode, projectToolUpdates(messages, [fetch], mode, history)]))
assert.deepEqual(results['addition-only'].tools, [{ ...fetch, deferLoading: true }])
assert.deepEqual(results['addition-only'].messages[1].content, [{ type: 'tool-addition', toolName: 'fetch' }])
assert.deepEqual(results['in-history'].tools, [search, { ...fetch, deferLoading: true }])
assert.deepEqual(results['in-history'].messages[1].content, swapped.content)
console.log(JSON.stringify(results, null, 2))
