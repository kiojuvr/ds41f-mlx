/* Web Chat is the sole frontend. Tab-local conversations; runtime owns settlement. */
'use strict';
const $ = id => document.getElementById(id);
const conversations = new Map();
function context(session = null) { return {session, busy: false, stop: false, attachments: [], tools: [], runtime: null, live: null, controller: null, notice: '', phase: '', draft: '', toolsMode: 'auto'}; }
let state = context();
// Bind asynchronous workflows to their conversation, not the visible selector.
// Switch never aborts another conversation or redirects its canonical outcome.
function scoped(ctx) {
  return {save: () => save(ctx), render: () => render(ctx), controls: () => controls(ctx),
    notice: value => notice(value, ctx), setPhase: value => setPhase(value, ctx),
    followLatest: () => { if (ctx === state) followLatest(); },
    reconcile: wait => reconcile(wait, ctx), generate: () => generate(ctx),
    toolLoop: explicit => toolLoop(explicit, ctx), pendingCalls: () => pendingCalls(ctx),
    appendResults: result => appendResults(result, ctx), generationRequest: () => generationRequest(ctx),
    status: () => status(ctx), settings: () => ctx === state ? settings() : {...ctx.session.generation}};
}
let visible = 100, follow = true, phase = '', liveNode = null, scrollIntentUntil = 0, layoutScrollUntil = 0;
const rendered = new Map();
function followLatest() { if (follow) { const node = $('messages'); if (node.scrollHeight - node.clientHeight - node.scrollTop > 1) node.scrollTop = node.scrollHeight; } $('latest').hidden = follow; }
$('messages').addEventListener('scroll', () => {
  const node = $('messages'), nearBottom = node.scrollHeight - node.clientHeight - node.scrollTop < 70;
  if (!nearBottom && (performance.now() > layoutScrollUntil || performance.now() < scrollIntentUntil)) follow = false;
  else if (follow || performance.now() < scrollIntentUntil) follow = true;
  $('latest').hidden = follow;
}, {passive: true});
for (const type of ['wheel', 'touchmove', 'keydown', 'pointerdown']) $('messages').addEventListener(type, () => { scrollIntentUntil = performance.now() + 600; }, {passive: true});
$('messages').addEventListener('pointermove', event => { if (event.buttons) scrollIntentUntil = performance.now() + 600; }, {passive: true});
$('latest').onclick = () => { follow = true; followLatest(); };
// Disclosure changes are deliberate reading actions, never token-follow triggers.
$('messages').addEventListener('click', event => { if (event.target.closest('summary')) { scrollIntentUntil = 0; follow = false; $('latest').hidden = false; } });
new ResizeObserver(() => { $('latest').style.bottom = ($('composer').offsetHeight + 28) + 'px'; followLatest(); }).observe($('composer'));
$('messages').addEventListener('load', event => { if (event.target.tagName === 'IMG') { layoutScrollUntil = performance.now() + 200; followLatest(); } }, true);
function setPhase(value, ctx = state) { ctx.phase = value; if (ctx === state) { phase = value; controls(); } }
const path = id => '/api/session/' + encodeURIComponent(id);
function notice(value, ctx = state) { ctx.notice = String(value || ''); if (ctx === state) $('notice').textContent = ctx.notice; }
async function api(url, method = 'GET', value) {
  const response = await fetch(url, {method, headers: {'content-type': 'application/json'}, body: value === undefined ? undefined : JSON.stringify(value)});
  const data = await response.json();
  if (!response.ok) throw Object.assign(new Error(data.error?.message || response.statusText), {status: response.status, code: data.error?.code, effectNotStarted: data.error?.effect_not_started});
  return data;
}
async function save(ctx = state) { if (ctx.session) await ChatStore.put('sessions', ctx.session); }
function policyDisplay() {
  const s = state.session;
  if (!s?.generation) return '';
  const active = state.busy || !!s.pending || pendingCalls().length > 0 || !!s.toolAwaitingResponse;
  const effective = active ? (s.workflow || s.generation) : s.generation;
  const different = s.workflow && JSON.stringify(s.workflow) !== JSON.stringify(s.generation);
  return (active ? 'Effective workflow: ' : 'Next user turn: ') + JSON.stringify(effective)
    + (different ? ' · ' + (active ? 'Next user turn: ' + JSON.stringify(s.generation) : 'Continuation snapshot: ' + JSON.stringify(s.workflow)) : '')
    + (s.protocol ? ' · reasoning session-owned' : '');
}
function controls(ctx = state) {
  if (ctx !== state) return;
  for (const id of ['send', 'closeSession', 'images', 'pdfs', 'refreshDocuments']) $(id).disabled = state.busy;
  $('newSession').disabled = false;
  $('sessions').disabled = false;
  for (const button of document.querySelectorAll('[data-action="remove-document"]')) button.disabled = state.busy;
  $('send').disabled ||= !state.session || !!state.session.problem || !!state.session.pending || state.session.effects?.state === 'reserved' || state.session.toolAwaitingResponse || pendingCalls().length > 0;
  const active = state.busy || state.session?.record?.state === 'busy';
  $('send').hidden = active; $('stop').hidden = !active;
  $('stop').disabled = !active || (state.stop && state.busy);
  $('status').textContent = state.stop && state.busy ? 'Stopping' : phase || (state.busy ? 'Recovering' : state.session?.record?.state === 'busy' ? 'Generating' : state.session?.problem || state.session?.toolPause === 'error' || state.session?.effects?.state === 'reserved' ? 'Error' : state.runtime?.status === 'ready' ? 'Ready' : ['error', 'unreachable', 'unavailable'].includes(state.runtime?.status) ? 'Error' : 'Recovering');
  const locked = !!state.session?.protocol;
  if (locked) { const effort = state.session.protocol.reasoning; $('thinking').value = effort === 'none' ? 'off' : 'on'; if (effort !== 'none') $('reasoning').value = effort; }
  const restricted = production();
  const workflowLocked = state.busy || !!state.session?.pending || pendingCalls().length > 0 || !!state.session?.toolAwaitingResponse;
  for (const name of GenerationPolicy.parameters) { if ($('policy_' + name)) $('policy_' + name).disabled = workflowLocked || !state.runtime?.generation_policy; }
  for (const id of ['outputLimit', 'maxTokens', 'resetGeneration']) $(id).disabled = workflowLocked;
  $('images').disabled ||= restricted;
  $('thinking').disabled = locked || workflowLocked;
  $('reasoning').disabled = locked || workflowLocked || $('thinking').value === 'off';
  $('thinkLabel').textContent = $('thinking').value === 'off' ? 'Think Off' : 'Think ' + $('reasoning').value;
  $('effectivePolicy').textContent = policyDisplay();
  $('thinkingLock').textContent = locked ? 'Locked for this conversation. Start a new chat to change.' : '';
  $('toolsLabel').textContent = 'Tools ' + {auto: 'Auto', ask: 'Ask', off: 'Off'}[$('toolsMode').value];
  $('discardPending').hidden = !state.session?.pending || !state.session?.problem;
  $('skipTools').hidden = !state.session || !pendingCalls().length || !!state.session.problem;
  $('diagnostics').textContent = JSON.stringify({runtime: state.runtime, settlement: state.session?.record, count: state.session?.count, pending: state.session?.pending, effects: state.session?.effects}, null, 2);
  termination();
}
async function menus() {
  $('sessions').replaceChildren();
  let index = 0;
  for (const ctx of conversations.values()) {
    const session = ctx.session; index++;
    const option = document.createElement('option'); option.value = session.id;
    option.textContent = (session.title || `Chat ${index}`) + (ctx.busy ? ' · Working' : '');
    option.selected = ctx === state;
    $('sessions').appendChild(option);
  }
}
function render(ctx = state) {
  if (ctx !== state) { menus(); return; }
  layoutScrollUntil = performance.now() + 200;
  const main = $('messages'), top = main.scrollTop;
  const anchor = [...main.querySelectorAll('article[data-index]')].find(node => node.offsetTop + node.offsetHeight >= top + main.offsetTop);
  const anchorIndex = anchor?.dataset.index, anchorOffset = anchor ? anchor.offsetTop - top : 0;
  const liveReasoningOpen = !!(liveNode?.isConnected && liveNode.querySelector('.reasoning')?.open);
  const open = new Map([...main.querySelectorAll('article')].map(node => [node.dataset.index, {reasoning: !!node.querySelector('.reasoning')?.open, toolcall: !!node.querySelector('.toolcall')?.open}]));
  const fragment = document.createDocumentFragment();
  const keep = new Set();
  const messages = state.session?.messages || [];
  const start = Math.max(0, messages.length - visible);
  if (start) {
    const older = document.createElement('button'); older.textContent = `Show 100 earlier messages (${start} hidden)`;
    older.onclick = () => { scrollIntentUntil = 0; follow = false; visible += 100; render(); }; fragment.appendChild(older);
  }
  const toolResults = new Map(messages.filter(m => m.role === 'tool').map(m => [m.tool_call_id, m]));
  for (let index = start; index < messages.length; index++) {
    if (messages[index].role === 'tool') continue;
    const key = index;
    let display = messages[index], interrupted = state.session.interruptions?.includes(index), interruptionIndex = interrupted ? index : -1, limited = state.session.endings?.[index] === 'length' && !interrupted;
    if (display.role === 'assistant') {
      const group = [display];
      while (index + 1 < messages.length && ['assistant', 'tool'].includes(messages[index + 1].role)) {
        index++; if (messages[index].role === 'assistant') group.push(messages[index]);
        if (state.session.interruptions?.includes(index)) { interrupted = true; interruptionIndex = index; } limited ||= state.session.endings?.[index] === 'length' && !state.session.interruptions?.includes(index);
      }
      display = {role: 'assistant', content: group.map(m => m.content || '').filter(Boolean).join('\n\n'), reasoning_content: group.map(m => m.reasoning_content || '').filter(Boolean).join('\n\n'), tool_calls: group.flatMap(m => m.tool_calls || [])};
    }
    if (display.role === 'user' && state.session.userPresentation?.[key]) display = {...display, presentation: state.session.userPresentation[key]};
    const lateStop = state.session.lateStops?.includes(interruptionIndex), userStopped = state.session.interruptionKinds?.[interruptionIndex] === 'user';
    const results = (display.tool_calls || []).map(call => toolResults.get(call.id)).filter(Boolean);
    // User payloads are immutable; avoid serializing historical original image bytes on each paint boundary.
    const fingerprint = display.role === 'user' ? (typeof display.content === 'string' ? state.session.id + ':' + key + ':' + display.content : display.content) : JSON.stringify([display, results, interrupted, lateStop, userStopped, limited]);
    let entry = rendered.get(key);
    if (!entry || entry.fingerprint !== fingerprint) {
      const node = ChatRender.message(fragment, display, results); node.dataset.index = String(key);
      for (const kind of ['reasoning', 'toolcall']) { const details = node.querySelector('.' + kind); if (details) details.open = open.get(String(key))?.[kind] || (kind === 'reasoning' && liveReasoningOpen && index === messages.length - 1); }
      if (interrupted) { const label = document.createElement('div'); label.className = 'small'; label.textContent = lateStop ? 'Response completed before Stop took effect. Canonical committed state has been reconciled.' : (userStopped ? 'Stopped by user. ' : 'Response interrupted. ') + 'Canonical committed state has been reconciled; a small committed suffix may have been recovered.'; node.appendChild(label); }
      if (limited) { const label = document.createElement('div'); label.className = 'small'; label.textContent = 'Output limit reached.'; node.appendChild(label); }
      entry = {node, fingerprint}; rendered.set(key, entry);
    }
    fragment.appendChild(entry.node); keep.add(key);
  }
  for (const index of rendered.keys()) if (!keep.has(index)) rendered.delete(index);
  if (state.live) { liveNode = ChatRender.message(fragment, state.live, [], true); liveNode.querySelector('.reasoning').open = liveReasoningOpen; }
  else if (state.session?.visibleUnsettled) {
    const node = ChatRender.message(fragment, state.session.visibleUnsettled);
    const label = document.createElement('div'); label.className = 'error';
    label.textContent = 'This partial response is unconfirmed. See the recovery actions below.';
    node.appendChild(label);
  }
  if (!messages.length && !state.live) { const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'What can I help with?'; fragment.appendChild(empty); }
  main.replaceChildren(fragment);
  const restoredAnchor = anchorIndex === undefined ? null : main.querySelector(`article[data-index="${anchorIndex}"]`);
  main.scrollTop = !follow && restoredAnchor ? restoredAnchor.offsetTop - anchorOffset : top;
  renderDocuments(); controls(); followLatest();
}
function termination() {
  document.querySelectorAll('.termination').forEach(node => node.remove());
  const s = state.session; if (!s || state.busy) return;
  let label = '', actions = [];
  if (['unrecoverable', 'closed'].includes(s.record?.state)) { label = 'This conversation cannot safely continue. Start a new chat. This tab keeps the existing history for reading.'; actions = [['New chat', () => $('newSession').click()], ['Recovery', () => $('settingsDialog').showModal()]]; }
  else if (s.record?.state === 'busy') { label = 'The response is still running. Wait for it to settle or Stop.'; actions = [['Reconcile', () => $('reconnect').click()], ['Stop', () => $('stop').click()]]; }
  else if (s.problem || s.pending) { label = 'The last request has an uncertain outcome. The runtime will not automatically retry it.'; actions = [['Reconcile', () => $('reconnect').click()], ['Details', () => $('settingsDialog').showModal()]]; }
  else if (s.capacityStop || s.terminationReason === 'context_capacity') { label = s.capacityStop ? 'Context/admission boundary: ' + s.capacityStop : 'Qualified total context capacity reached. This history cannot continue in the same envelope.'; actions = [['Settings / Recovery', () => $('settingsDialog').showModal()], ['Check / Continue', pendingCalls().length || s.toolAwaitingResponse ? continueTools : continueResponse], ['New chat', () => $('newSession').click()]]; }
  else if (s.effects?.state === 'reserved') { label = (s.toolError ? 'Tool failed: ' + s.toolError + '. ' : '') + 'Tool outcome unavailable. It will not be retried.'; actions = [['Continue without result', () => $('skipTools').click()], ['Details', () => $('settingsDialog').showModal()]]; }
  else if (s.toolPause === 'error') { label = 'Tool failed: ' + (s.toolError || 'Result unavailable'); actions = [['Continue without result', () => $('skipTools').click()]]; if (!pendingCalls().length) actions = [['Continue without result', continueResponse]]; }
  else if (pendingCalls().length) {
    label = s.toolPause === 'ceiling' ? `Runaway circuit breaker reached (${s.toolRounds} rounds). Pending calls are preserved; only explicit Continue resets it.` : s.toolPause === 'stopped' ? 'Tool workflow stopped. Pending tools have not been run.' : s.interrupted ? 'Stopped by user. Requested tools have not been run.' : $('toolsMode').value === 'ask' ? 'Allow requested tools to run?' : 'Tool execution paused.';
    actions = [['Continue tools', continueTools], ['Stop', pauseTools], ['Continue without result', () => $('skipTools').click()]];
  } else if (s.toolAwaitingResponse) { label = s.interrupted ? 'Stopped by user. Tool results were received safely.' : 'Tool results received. Continue the response?'; actions = [['Continue tools', continueTools]];
  } else if (s.finishReason === 'length' && !s.interrupted) { label = 'Output limit reached.'; actions = [['Continue response', continueResponse]]; }
  if (!label) return;
  const parent = [...$('messages').querySelectorAll('.assistant')].pop() || $('messages');
  const box = document.createElement('div'); box.className = 'termination'; const message = document.createElement('div'); message.textContent = label; box.appendChild(message);
  for (const [name, action] of actions) { const button = document.createElement('button'); button.type = 'button'; button.textContent = name; button.onclick = action; box.appendChild(button); }
  parent.appendChild(box);
}
async function pauseTools() { await exclusive(async ctx => { ctx.session.toolPause = 'stopped'; ctx.session.interrupted = true; await save(ctx); }); }
async function continueResponse() { await exclusive(async ctx => { const state = ctx; const {reconcile, notice, pendingCalls, generate, toolLoop} = scoped(ctx); await reconcile(); if (state.stop) { notice('Stopped before continuing.'); return; } if (state.session.problem || state.session.pending || pendingCalls().length || state.session.effects?.state === 'reserved') throw new Error('Reconcile the current outcome first'); state.session.toolPause = ''; await generate(); await toolLoop(); }); }
async function continueTools() { await exclusive(async ctx => { const state = ctx; const {reconcile, notice, pendingCalls, generate, toolLoop} = scoped(ctx); await reconcile(); if (state.stop) { notice('Stopped before continuing tools.'); return; } const s = state.session; if (s.problem || s.pending || s.effects?.state === 'reserved') throw new Error('Reconcile the tool outcome first'); if (s.toolAwaitingResponse && !pendingCalls().length) await generate(); await toolLoop(true); }); }
async function select(id) {
  const next = conversations.get(id);
  if (!next) return;
  for (const dialog of ['documentDialog', 'documentsDrawer']) if ($(dialog).open) $(dialog).close();
  state.draft = $('input').value;
  state = next;
  phase = state.phase;
  visible = 100; rendered.clear(); liveNode = null; follow = true;
  $('input').value = state.draft; resizeInput();
  $('toolsMode').value = state.toolsMode;
  if (state.session?.generation) applyDefaults(state.session.generation);
  notice(state.notice); preview(); render(); await menus();
}
function production() { return state.runtime?.capabilities?.stateful_chat === 'production-sessions-v1'; }
function negotiate() { return GenerationPolicy.negotiate(state.runtime); }
function buildPolicyControls() {
  negotiate();
  const contract = state.runtime.generation_policy;
  if ($('samplingControls').dataset.contract === JSON.stringify(contract)) return;
  if ($('samplingControls').dataset.contract && state.session) throw new Error('Runtime policy contract changed; close this conversation');
  $('samplingControls').replaceChildren();
  const advanced = document.createElement('details'), summary = document.createElement('summary'); summary.textContent = 'Advanced sampling'; advanced.appendChild(summary);
  GenerationPolicy.parameters.forEach((name, index) => {
    const spec = contract.parameters[name], label = document.createElement('label'), input = document.createElement('input');
    label.textContent = name + ' · default ' + JSON.stringify(contract.defaults[name]);
    input.id = 'policy_' + name; input.type = 'number'; input.step = spec.type === 'integer' ? '1' : 'any';
    if ('minimum' in spec) input.min = spec.minimum;
    if ('exclusiveMinimum' in spec) { input.min = spec.exclusiveMinimum; input.title = 'Must be greater than ' + spec.exclusiveMinimum; }
    if ('maximum' in spec) input.max = spec.maximum;
    if (name === 'seed') input.placeholder = 'null (' + spec.semantics + ')';
    input.onchange = policyChanged; label.appendChild(input);
    (index < 4 ? $('samplingControls') : advanced).appendChild(label);
  });
  $('samplingControls').appendChild(advanced);
  $('reasoning').replaceChildren();
  for (const effort of contract.reasoning.efforts.filter(e => e !== 'none')) { const option = document.createElement('option'); option.value = effort; option.textContent = effort; $('reasoning').appendChild(option); }
  $('samplingControls').dataset.contract = JSON.stringify(contract);
}
function applyDefaults(d) {
  GenerationPolicy.validate(state.runtime.generation_policy, d);
  for (const name of GenerationPolicy.parameters) $('policy_' + name).value = d[name] === null ? '' : d[name];
  $('thinking').value = d.reasoning_effort === 'none' ? 'off' : 'on';
  if (d.reasoning_effort !== 'none') $('reasoning').value = d.reasoning_effort;
  $('outputLimit').value = d.max_tokens === 'auto' ? 'auto' : 'custom';
  $('maxTokens').hidden = d.max_tokens === 'auto';
  if (d.max_tokens !== 'auto') $('maxTokens').value = d.max_tokens;
}
async function createConversation() {
  const preference = state.policyPreference || negotiate();
  const id = 'web-' + ChatPlatform.uuid();
  const session = {id, title: '', messages: [], count: 0, frontier: 0, problem: '', native: false,
    generation: {...preference}, policyVersion: state.runtime.generation_policy.version};
  const ctx = context(session);
  ctx.runtime = state.runtime; ctx.tools = state.tools; ctx.toolsMode = state.toolsMode;
  conversations.set(id, ctx); await save(ctx); await select(id);
}
async function ensureRuntime(ctx) {
  const s = ctx.session;
  if (s.native) return;
  if (s.creationUnknown) throw new Error('Conversation creation could not be confirmed. Reconcile; never automatically repeat it.');
  s.creationUnknown = true; // Known correlation ID BEFORE create I/O, no retry.
  try {
    const rec = await api('/api/session', 'POST', {id: s.id});
    if (rec.id !== s.id) throw new Error('Conversation creation identity mismatch');
    s.native = true; s.creationUnknown = false;
  } catch (error) { s.problem = error.message; throw error; }
}
async function status(ctx = state) {
  const result = await api('/api/status');
  ctx.runtime = result.runtime;
  for (const item of conversations.values()) item.runtime = result.runtime;
  if (ctx !== state) return;
  $('documentCapability').textContent = result.documents?.vision_available ? 'Retained PDF · isolated page Vision available · no chat image slots used' : 'Retained PDF text/search available. ' + (result.documents?.vision_notice || 'Isolated Vision unavailable.');
  state.tools = (result.tools || []).filter(tool => tool.function?.name !== 'fetch_pdf' && (!production() || tool.function?.name !== 'fetch_image'));
  buildPolicyControls();
  if (!state.session) applyDefaults(state.policyPreference || negotiate());
  controls();
}
function appendResults(result, ctx = state) {
  const s = ctx.session;
  for (const message of result.messages) {
    // Only current canonical batch; each result is appended once.
    if (!s.messages.some(m => m.role === 'tool' && m.tool_call_id === message.tool_call_id)) s.messages.push(message);
  }
  for (const item of result.results || []) {
    if (item.artifact_id && (item.tool === 'fetch_pdf' || item.tool === 'document_pdf')) {
      s.documents ||= [];
      let doc = s.documents.find(d => d.artifact_id === item.artifact_id);
      if (!doc) { doc = {artifact_id: item.artifact_id, announced: true}; s.documents.push(doc); }
      Object.assign(doc, {name: item.filename || doc.name || 'PDF', page_count: item.page_count || doc.page_count, state: item.error ? 'inspection-error' : 'retained'});
      if (item.action === 'info') doc.pageStates = item.pages || [];
      for (const page of item.pages || []) {
        if (!page.observation) continue;
        doc.pageStates ||= [];
        let material = doc.pageStates.find(p => p.page === page.page);
        if (!material) { material = {page: page.page}; doc.pageStates.push(material); }
        material.observation_state = page.observation.state;
      }
      if (item.tool === 'fetch_pdf' || ['acquire', 'read', 'observe', 'observation'].includes(item.action)) {
        doc.readPages = [...new Set([...(doc.readPages || []), ...(item.pages || []).filter(p => !p.observation || p.observation.state === 'completed').map(p => p.page)])];
      }
    }
  }
  s.effects = {count: s.count, state: 'completed'};
  s.toolAwaitingResponse = true;
  s.capacityStop = result.budget_error || '';
  const failure = result.results?.find(item => item.error);
  s.toolPause = failure ? 'error' : ''; s.toolError = failure ? (typeof failure.error === 'string' ? failure.error : JSON.stringify(failure.error)) : '';
}
async function reconcile(wait = false, ctx = state) {
  const state = ctx;
  const {save, render, notice, controls, appendResults} = scoped(ctx);
  const s = state.session;
  if (!s || (!s.native && !s.creationUnknown)) return {state: 'empty', request_count: 0};
  try {
    let rec;
    for (let i = 0; ; i++) {
      rec = await api(path(s.id));
      if (!wait || rec.state !== 'busy' || i >= 600) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    s.native = true; s.creationUnknown = false;
    if (ctx === windowActive()) $('sessionInfo').textContent = `${rec.state} · turns ${rec.request_count} · replay ${rec.diagnostics?.m8?.total_prompt_replay_count ?? 0} · repack ${rec.diagnostics?.m8?.total_full_cache_repack_count ?? 0}`;
    if (rec.state === 'busy') { s.problem = 'Runtime request is still active; wait or Stop with its active request identity.'; }
    else if (rec.state === 'unrecoverable' || rec.state === 'closed') { s.problem = 'This conversation cannot safely continue. Start a new chat; existing history remains readable. No automatic replay.' + (rec.last_error ? '\n' + rec.last_error : ''); }
    else if (s.pending && rec.request_count === s.pending.base + 1) {
      const turn = rec.last_turn;
      if (s.pending.nonce !== turn?.application_request_id || (s.pending.requestId && s.pending.requestId !== turn?.request_id)) throw new Error('Request identity mismatch; application history cannot be reconciled');
      if (!turn?.response_json) throw new Error('No canonical response for pending turn');
      const userStop = state.stop || !!s.pending.userStopRequested;
      s.messages.push(turn.response_json.choices[0].message);
      s.toolAwaitingResponse = false;
      s.finishReason = turn.response_json.choices[0].finish_reason;
      s.terminationReason = turn.termination_reason || s.finishReason; s.capacity = turn.capacity || s.capacity; s.capacityStop = '';
      s.endings ||= {}; s.endings[s.messages.length - 1] = turn.cancelled ? 'cancelled' : s.finishReason;
      if (turn.cancelled || userStop) { s.interruptions ||= []; s.interruptions.push(s.messages.length - 1); s.interruptionKinds ||= {}; s.interruptionKinds[s.messages.length - 1] = userStop ? 'user' : 'interrupted'; if (!turn.cancelled) { s.lateStops ||= []; s.lateStops.push(s.messages.length - 1); } }
      s.count = rec.request_count; s.pending = null; s.problem = ''; s.visibleUnsettled = null;
      s.interrupted = !!turn.cancelled || userStop;
      notice(turn.cancelled ? 'Interrupted at a committed transaction. Canonical response recovered from runtime.' : '');
    } else if (rec.request_count !== s.count) {
      s.problem = 'Runtime/application turn count differs. Missing ordinary history; continuation disabled (no replay).';
    } else if (s.pending) {
      s.problem = 'No settled turn for the frozen request. No automatic retry. Reconcile, then explicitly discard the unadmitted UI request if appropriate.' + (s.pending.error ? '\n' + s.pending.error : '');
    } else s.problem = '';
    s.frontier = rec.diagnostics?.m8?.frontier || 0;
    s.record = {state: rec.state, requestId: rec.active_request_id};
    if (s.effects?.state === 'reserved' && !s.problem) {
      const recovered = await api(`/api/tools/result?session_id=${encodeURIComponent(s.id)}&request_count=${s.effects.count}`);
      if (recovered.result) appendResults(recovered.result);
      else notice('Tool effect outcome unavailable. It will NOT be retried; explicitly continue with unavailable results or close.');
    }
    if (s.problem) notice(s.problem);
    await save(); render();
    return rec;
  } catch (error) {
    s.problem = error.status === 404 ? 'This conversation expired or the LLM Runtime restarted. History remains readable; start a new chat to continue. No replay or automatic resend.' : error.message;
    notice(s.problem); controls(); throw error;
  }
}
function windowActive() { return state; }
async function exclusive(action, ctx = state) {
  if (ctx.busy) return;
  ctx.busy = true; ctx.stop = false; setPhase('Recovering', ctx);
  try { await action(ctx); }
  catch (error) { notice(error.message, ctx); }
  finally {
    ctx.busy = false; ctx.controller = null; ctx.live = null; setPhase('', ctx);
    render(ctx); await menus();
  }
}
function settings() {
  negotiate();
  if (state.session && state.session.policyVersion !== state.runtime.generation_policy.version) throw new Error('Conversation lacks supported generation policy; start a new chat');
  const values = {};
  for (const name of GenerationPolicy.parameters) { const text = $('policy_' + name).value; if (!text && name !== 'seed') throw new Error(name + ' is required'); values[name] = !text ? null : Number(text); }
  values.max_tokens = $('outputLimit').value === 'auto' ? 'auto' : Number($('maxTokens').value);
  values.reasoning_effort = $('thinking').value === 'off' ? 'none' : $('reasoning').value;
  if (state.session?.protocol && values.reasoning_effort !== state.session.protocol.reasoning) throw new Error('Reasoning is session-owned. Start a new chat to change it.');
  return GenerationPolicy.validate(state.runtime.generation_policy, values);
}
function policyChanged() {
  try {
    const values = settings(); state.policyPreference = {...values};
    if (state.session) { state.session.generation = {...values}; save().catch(error => notice(error.message)); }
    controls();
  } catch (error) { notice(error.message); }
}
function generationRequest(ctx = state) {
  const s = ctx.session;
  GenerationPolicy.negotiate(ctx.runtime);
  const options = GenerationPolicy.validate(ctx.runtime.generation_policy, s.workflow || s.generation);
  if (s.protocol && options.reasoning_effort !== s.protocol.reasoning) throw new Error('Reasoning is session-owned. Start a new chat to change it.');
  return {model: 'deepseek-v4.1-flash', messages: s.messages, stream: true, ...options,
    tools: s.protocol?.tools || ctx.tools, tool_choice: 'auto'};
}
async function generate(ctx = state) {
  const state = ctx;
  const {save, render, notice, reconcile, setPhase, followLatest, generationRequest} = scoped(ctx);
  const s = state.session, request = generationRequest();
  // Nonbinding observation, using the actual request. Actual admission rechecks
  // it before native reservation; preview never grants execution permission.
  try {
    s.capacity = await api('/api/budget', 'POST', {session_id: s.id, expected_count: s.count, request});
    s.capacityStop = '';
  } catch (error) {
    s.capacityStop = error.message; s.pending = null; await save(); render();
    throw new Error('No generation submitted: ' + error.message);
  }
  if (state.stop) { s.pending = null; s.interrupted = true; await save(); notice('Stopped before generation admission.'); return; }
  // Recipe tool declarations are part of the historical prompt prefix. Keep
  // them stable; Tools mode controls client execution, not prefix rewriting.
  s.pending ||= {base: s.count, requestId: null, nonce: ChatPlatform.uuid(), addedUser: false};
  s.protocol ||= {tools: request.tools, reasoning: request.reasoning_effort};
  await save(); // freeze tab-lifetime journal BEFORE request/effect; no archive
  state.controller = new AbortController();
  s.visibleUnsettled = null;
  state.live = {role: 'assistant', content: '', reasoning_content: ''};
  let outputNode = ctx === windowActive() ? ChatRender.message($('messages'), state.live, [], true) : null;
  if (ctx === windowActive()) liveNode = outputNode;
  setPhase('Queued / Generating');
  // Append to persistent text nodes while streaming; Markdown parses once at settlement.
  const paintTimer = setInterval(() => {
    if (ctx !== windowActive() || !ctx.live) return;
    if (!outputNode?.isConnected || liveNode !== outputNode) { render(); outputNode = liveNode; }
    if (outputNode) ChatRender.updateLive(outputNode, ctx.live);
    followLatest();
  }, 80);
  try {
    if (state.stop) throw new Error('Stopped before request submission. No request was retried.');
    const response = await fetch('/api/stream', {method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({session_id: s.id, expected_count: s.count, application_id: s.pending.nonce, request}), signal: state.controller.signal});
    if (!response.ok) { const data = await response.json(); throw new Error(data.error?.message || response.statusText); }
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
    async function frame(frame) {
      const lines = frame.split('\n'); const event = lines.find(x => x.startsWith('event:'))?.slice(6).trim();
      const data = lines.filter(x => x.startsWith('data:')).map(x => x.slice(5).trimStart()).join('\n');
      if (!data || data === '[DONE]') return;
      const value = JSON.parse(data);
      if (event === 'admitted') { s.pending.requestId = value.request_id; await save(); return; }
      if (event === 'error') throw new Error(value.message);
      const delta = value.choices?.[0]?.delta || {};
      state.live.content += delta.content || ''; state.live.reasoning_content += delta.reasoning_content || '';
      for (const call of delta.tool_calls || []) {
        state.live.tool_calls ||= [];
        const item = state.live.tool_calls[call.index] ||= {id: '', type: 'function', function: {name: '', arguments: ''}};
        if (call.id) item.id = call.id;
        if (call.function?.name) item.function.name += call.function.name;
        if (call.function?.arguments) item.function.arguments += call.function.arguments;
      }
      // Painting is independent of token cadence and does not rebuild the response DOM.
    }
    while (true) {
      const chunk = await reader.read(); if (chunk.done) break;
      buffer += decoder.decode(chunk.value, {stream: true});
      let split; while ((split = buffer.indexOf('\n\n')) >= 0) { await frame(buffer.slice(0, split)); buffer = buffer.slice(split + 2); }
    }
    buffer += decoder.decode();
    if (buffer.trim()) throw new Error('Incomplete SSE frame; reconcile committed runtime outcome');
  } catch (error) {
    if (error.name !== 'AbortError' && s.pending) s.pending.error = error.message;
    notice(error.name === 'AbortError' ? 'Stop/disconnect requested; waiting for canonical settlement…' : error.message);
  }
  finally {
    clearInterval(paintTimer); setPhase(state.stop ? 'Stopping' : 'Recovering');
    const visibleOutput = state.live; state.live = null;
    try { await reconcile(true); }
    finally {
      if (s.problem && (visibleOutput?.content || visibleOutput?.reasoning_content || visibleOutput?.tool_calls)) {
        s.visibleUnsettled = visibleOutput; await save(); render();
      }
    }
  }
}
function pendingCalls(ctx = state) {
  const s = ctx.session;
  const last = [...(s?.messages || [])].reverse().find(m => m.role === 'assistant');
  return (last?.tool_calls || []).filter(call => !s.messages.some(m => m.role === 'tool' && m.tool_call_id === call.id));
}
async function toolLoop(explicit = false, ctx = state) {
  const state = ctx;
  const {save, render, notice, generate, setPhase, pendingCalls, appendResults, generationRequest} = scoped(ctx);
  const s = state.session;
  if (!pendingCalls().length) return;
  if (!explicit && ctx.toolsMode !== 'auto') { s.toolPause = 'approval'; await save(); return; }
  if (explicit && ctx.toolsMode === 'off') { notice('Tools are Off. Choose Ask or Auto to execute them.'); return; }
  const ceiling = 128; // runaway circuit breaker, not an ordinary UX round quota
  s.toolPause = ''; s.toolRounds = 0;
  for (let round = 0; round < ceiling && !state.stop; round++) {
    const calls = pendingCalls(); if (!calls.length || state.session.problem) return;
    if (ctx.toolsMode === 'off') { s.toolPause = 'approval'; await save(); return; }
    if (state.session.effects?.state === 'reserved') { notice('Tool reservation is uncertain; no automatic re-execution'); return; }
    setPhase('Using tool'); notice(calls.some(c => c.function.name === 'web_search') ? 'Searching web…' : 'Fetching URL…');
    state.session.effects = {count: state.session.count, state: 'reserved'}; await save();
    let result;
    try { result = await api('/api/tools', 'POST', {session_id: state.session.id, request_count: state.session.count, request: generationRequest()}); }
    catch (error) { if (error.effectNotStarted) s.effects = null; s.toolPause = 'error'; s.toolError = error.message; await save(); throw error; }
    appendResults(result); s.toolRounds = round + 1; await save(); render(); notice('');
    if (result.budget_error) { s.capacityStop = result.budget_error; await save(); render(); return; }
    const failed = result.results?.find(item => item.error);
    if (failed) { s.toolPause = 'error'; s.toolError = typeof failed.error === 'string' ? failed.error : JSON.stringify(failed.error); await save(); return; }
    if (state.stop) { s.interrupted = true; s.interruptions ||= []; const index = s.messages.findLastIndex(m => m.role === 'assistant'); if (!s.interruptions.includes(index)) s.interruptions.push(index); s.interruptionKinds ||= {}; s.interruptionKinds[index] = 'user'; await save(); render(); return; }
    await generate();
    if (ctx.toolsMode === 'ask' && pendingCalls().length) { s.toolPause = 'approval'; await save(); return; }
  }
  if (pendingCalls().length && !state.stop) { s.toolPause = 'ceiling'; await save(); }
  if (state.stop) { s.interrupted = true; await save(); }
}
async function send() {
  const ctx = windowActive();
  const state = ctx;
  const {save, render, notice, reconcile, generate, toolLoop, settings, pendingCalls} = scoped(ctx);
  const newDocuments = (state.session?.documents || []).filter(d => !d.announced && d.state === 'retained');
  const reference = newDocuments.map(d => `Retained PDF: ${d.name}; artifact_id=${d.artifact_id}; pages=${d.page_count}. Use document_pdf to search/read/observe additional pages without reacquisition. Original PDF bytes are source authority; parser text and model observations are derived. Cite page numbers.`).join('\n');
  const userText = $('input').value.trim();
  const text = [userText, reference].filter(Boolean).join('\n\n');
  if (!text && !state.attachments.length) return;
  await exclusive(async () => {
    settings(); await ensureRuntime(ctx); await reconcile();
    if (state.stop) { notice('Stopped before sending. Your draft was kept.'); return; }
    const s = state.session;
    if (s.problem || s.pending || s.effects?.state === 'reserved') throw new Error(s.problem || 'Unresolved application request/effect');
    if (pendingCalls().length || s.toolAwaitingResponse) throw new Error('Continue the pending tool response before another user turn');
    const parts = [];
    if (text) parts.push({type: 'text', text});
    if (ctx.runtime?.capabilities?.stateful_chat === 'production-sessions-v1' && state.attachments.length) throw new Error('Images unsupported by runtime');
    for (const file of state.attachments) parts.push({type: 'image_url', image_url: {url: file.url}});
    s.generation = settings(); s.workflow = {...s.generation};
    s.toolPause = ''; s.toolError = ''; s.interrupted = false;
    s.userPresentation ||= {};
    s.userPresentation[s.messages.length] = {content: state.attachments.length ? [{type: 'text', text: userText}, ...parts.filter(p => p.type === 'image_url')] : userText,
      documents: newDocuments.map(d => ({name: d.name, page_count: d.page_count, artifact_id: d.artifact_id})), context: reference};
    s.messages.push({role: 'user', content: state.attachments.length ? parts : text});
    for (const reference of newDocuments) {
      const doc = (s.documents || []).find(d => d.artifact_id === reference.artifact_id);
      if (doc) doc.announced = true;
    }
    s.title ||= userText.slice(0, 60) || 'Document conversation';
    s.pending = {base: s.count, requestId: null, nonce: ChatPlatform.uuid(), addedUser: true};
    await save(); state.draft = ''; state.attachments = [];
    if (ctx === windowActive()) { $('input').value = ''; resizeInput(); follow = true; preview(); }
    render();
    await generate(); await toolLoop();
  });
}
async function attach(files, ctx = state) {
  const state = ctx;
  const {notice} = scoped(ctx);
  if (state.busy) return;
  try {
    GenerationPolicy.negotiate(ctx.runtime);
    if (ctx.runtime?.capabilities?.stateful_chat === 'production-sessions-v1') throw new Error('Images unsupported by runtime');
    const historical = (state.session?.messages || []).flatMap(m => Array.isArray(m.content) ? m.content : []).filter(p => p.type === 'image_url').length;
    if (historical + state.attachments.length + files.length > 4) throw new Error('At most four images in the full conversation');
    const additions = [];
    for (const file of files) {
      if (file.size > 16 * 1024 * 1024) throw new Error(`${file.name}: exceeds 16 MiB encoded bytes`);
      const bytes = new Uint8Array(await file.arrayBuffer());
      const signature = String.fromCharCode(...bytes.slice(0, 12));
      const type = bytes[0] === 255 && bytes[1] === 216 ? 'image/jpeg' : signature.startsWith('\x89PNG\r\n\x1a\n') ? 'image/png' : signature.startsWith('RIFF') && signature.slice(8) === 'WEBP' ? 'image/webp' : '';
      if (!type) throw new Error(`${file.name}: PNG/JPEG/WebP only`);
      const bitmap = await createImageBitmap(file);
      const width = bitmap.width, height = bitmap.height; bitmap.close();
      if (width * height > 4194304 || width / height < .5 || width / height > 2) throw new Error(`${file.name}: image exceeds 4 MP or aspect ratio 1:2–2:1`);
      const url = await new Promise((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(`data:${type};base64,${reader.result.split(',')[1]}`);
        reader.onerror = () => reject(reader.error); reader.readAsDataURL(file);
      });
      additions.push({name: file.name, url, width, height}); // original bytes, no conversion
    }
    state.attachments.push(...additions); if (ctx === windowActive()) preview(); notice('');
  } catch (error) { notice(error.message); }
}
function preview() {
  layoutScrollUntil = performance.now() + 200;
  $('attachments').replaceChildren();
  state.attachments.forEach((file, index) => {
    const box = document.createElement('div'), image = document.createElement('img'), remove = document.createElement('button');
    image.src = file.url; image.alt = file.name; remove.type = 'button'; remove.textContent = 'Remove ' + file.name;
    remove.onclick = () => { state.attachments.splice(index, 1); preview(); };
    box.append(image, remove); $('attachments').appendChild(box);
  });
}
$('composer').onsubmit = event => { event.preventDefault(); send().catch(error => notice(error.message)); };
$('images').onchange = async event => { await attach([...event.target.files]); event.target.value = ''; };
$('images').closest('label').onkeydown = event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); if (!state.busy) $('images').click(); } };
async function attachFiles(files) {
  const ctx = state;
  const pdf = file => file.type === 'application/pdf' || /\.pdf$/i.test(file.name);
  for (const file of files.filter(pdf)) await addPDF(file, ctx);
  const images = files.filter(file => !pdf(file)); if (images.length) await attach(images, ctx);
}
$('composer').ondragover = event => event.preventDefault();
$('composer').ondrop = event => { event.preventDefault(); attachFiles([...event.dataTransfer.files]); };
$('input').onpaste = event => { const files = [...event.clipboardData.items].filter(item => item.kind === 'file').map(item => item.getAsFile()).filter(Boolean); if (files.length) { event.preventDefault(); attachFiles(files); } };
function resizeInput() { layoutScrollUntil = performance.now() + 200; $('input').style.height = 'auto'; $('input').style.height = Math.min(190, $('input').scrollHeight) + 'px'; }
$('input').oninput = resizeInput;
let composing = false;
$('input').addEventListener('compositionstart', () => { composing = true; });
$('input').addEventListener('compositionend', () => { composing = false; });
$('input').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && !composing && event.keyCode !== 229) { event.preventDefault(); if (!state.busy && !$('send').disabled) $('composer').requestSubmit(); } };
$('stop').onclick = async () => {
  const ctx = windowActive(), state = ctx;
  const {save, notice, setPhase, reconcile} = scoped(ctx);
  state.stop = true; setPhase('Stopping'); notice('Waiting for safe cancellation boundary…');
  const s = state.session;
  if (s?.pending) { s.pending.userStopRequested = true; try { await save(); } catch (_) { notice('Could not record Stop intent. Requesting safe cancellation…'); } }
  try {
    const id = s?.pending?.requestId || (await api(path(s.id))).active_request_id;
    if (id) await api(path(s.id) + '/cancel', 'POST', {request_id: id});
  } catch (error) { notice('Stop: ' + error.message + '; reconciling via GET'); }
  finally { state.controller?.abort(); if (!state.busy) await reconcile(true); }
};
$('newSession').onclick = () => createConversation().catch(error => notice(error.message));
$('sessions').onchange = () => select($('sessions').value).catch(error => notice(error.message));
$('reconnect').onclick = () => exclusive(async ctx => { await status(ctx); await reconcile(true, ctx); });
$('closeSession').onclick = async () => {
  const ctx = state, s = ctx.session;
  if (!s || ctx.busy || !confirm('Delete this conversation from this tab?')) return;
  conversations.delete(s.id); await ChatStore.remove('sessions', s.id);
  const next = conversations.keys().next().value;
  if (next) await select(next); else await createConversation();
  // Cleanup is an internal best-effort side effect, never correctness authority.
  if (s.native || s.creationUnknown) api(path(s.id), 'DELETE').catch(() => {});
};
$('discardPending').onclick = () => exclusive(async ctx => {
  const state = ctx;
  const {save, render, notice} = scoped(ctx);
  const s = state.session; if (!s?.pending) return;
  const rec = await api(path(s.id));
  if (!['empty', 'idle'].includes(rec.state) || rec.request_count !== s.pending.base) throw new Error('Runtime outcome is not unadmitted; reconcile rather than discard');
  if (!confirm('Discard only this UI request, which has no recorded committed turn? It will NOT be sent again.')) return;
  if (s.pending.addedUser) { s.messages.pop(); if (s.userPresentation) delete s.userPresentation[s.messages.length]; }
  s.pending = null; s.problem = ''; await save(); render(); notice('Unadmitted UI request discarded.');
});
$('reasoning').onchange = $('thinking').onchange = () => {
  const protocol = state.session?.protocol;
  if (protocol) { $('thinking').value = protocol.reasoning === 'none' ? 'off' : 'on'; if (protocol.reasoning !== 'none') $('reasoning').value = protocol.reasoning; }
  policyChanged(); controls();
};
$('toolsMode').onchange = () => { state.toolsMode = $('toolsMode').value; controls(); };
$('settingsButton').onclick = async () => { $('settingsDialog').showModal(); try { await menus(); } catch (error) { notice(error.message); } };
$('settingsClose').onclick = () => $('settingsDialog').close();
$('outputLimit').onchange = () => { const custom = $('outputLimit').value === 'custom'; $('maxTokens').hidden = !custom; if (!custom && $('outputLimit').value !== 'auto') $('maxTokens').value = $('outputLimit').value; policyChanged(); };
$('maxTokens').onchange = policyChanged;
$('resetGeneration').onclick = () => { try { const d = negotiate(); if (state.session?.protocol) d.reasoning_effort = state.session.protocol.reasoning; applyDefaults(d); policyChanged(); } catch (error) { notice(error.message); } };
$('skipTools').onclick = () => exclusive(async ctx => {
  const state = ctx;
  const {save, render, notice, reconcile, pendingCalls, generate, toolLoop} = scoped(ctx);
  await reconcile(); if (state.session.problem) throw new Error(state.session.problem);
  if (state.stop) { notice('Stopped before continuing.'); return; }
  if (!confirm('Do not execute/retry these tools. Return explicit unavailable results to the model?')) return;
  for (const call of pendingCalls()) state.session.messages.push({role: 'tool', tool_call_id: call.id, content: JSON.stringify({error: 'Client tool result unavailable; NOT executed/retried by recovery'})});
  state.session.effects = {count: state.session.count, state: 'completed'}; await save(); render();
  await generate(); await toolLoop();
});
(async () => {
  try {
    await status(); await createConversation();
    controls();
  } catch (error) { notice(error.message); }
})();
setInterval(() => status().catch(error => { state.runtime = {status: 'error', error: error.message}; controls(); notice('Cannot connect to runtime. Open Settings → Recovery to reconnect.'); }), 10000);
function nativeIds() { return [...conversations.values()].filter(ctx => ctx.session.native || ctx.session.creationUnknown).map(ctx => ctx.session.id); }
setInterval(() => { const ids = nativeIds(); if (ids.length) api('/api/keepalive', 'POST', {ids}).catch(() => {}); }, 30000);
window.addEventListener('pagehide', () => {
  const ids = nativeIds();
  if (ids.length) navigator.sendBeacon('/api/release', new Blob([JSON.stringify({ids})], {type: 'application/json'}));
  // Beacon success is not retirement authority. Server idle cleanup is the fallback.
});
