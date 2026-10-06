/* Thin stateful application. IndexedDB is UI history; runtime state is canonical. */
'use strict';
const $ = id => document.getElementById(id);
const state = {session: null, busy: false, stop: false, attachments: [], tools: [], runtime: null, live: null, controller: null};
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
function setPhase(value) { phase = value; controls(); }
const path = id => '/api/session/' + encodeURIComponent(id);
function notice(value) { $('notice').textContent = String(value || ''); }
async function api(url, method = 'GET', value) {
  const response = await fetch(url, {method, headers: {'content-type': 'application/json'}, body: value === undefined ? undefined : JSON.stringify(value)});
  const data = await response.json();
  if (!response.ok) throw Object.assign(new Error(data.error?.message || response.statusText), {status: response.status});
  return data;
}
async function save() { if (state.session) await ChatStore.put('sessions', state.session); }
function controls() {
  for (const id of ['send', 'newSession', 'closeSession', 'saveSession', 'restoreSession', 'sessions', 'saves', 'images']) $(id).disabled = state.busy;
  $('send').disabled ||= !state.session || !!state.session.problem || !!state.session.pending || state.session.effects?.state === 'reserved' || state.session.toolAwaitingResponse || pendingCalls().length > 0;
  const active = state.busy || state.session?.record?.state === 'busy';
  $('send').hidden = active; $('stop').hidden = !active;
  $('stop').disabled = !active || !!state.session?.pendingRestore || (state.stop && state.busy);
  $('status').textContent = state.stop && state.busy ? 'Stopping' : phase || (state.busy ? 'Recovering' : state.session?.record?.state === 'busy' ? 'Generating' : state.session?.problem || state.session?.toolPause === 'error' || state.session?.effects?.state === 'reserved' ? 'Error' : state.runtime?.status === 'ready' ? 'Ready' : ['error', 'unreachable', 'unavailable'].includes(state.runtime?.status) ? 'Error' : 'Recovering');
  const locked = !!state.session?.protocol;
  if (locked) { const effort = state.session.protocol.reasoning; $('thinking').value = effort === 'none' ? 'off' : 'on'; if (effort !== 'none') $('reasoning').value = ({minimal: 'low', medium: 'high', xhigh: 'high'}[effort] || effort); }
  $('thinking').disabled = locked || state.busy;
  $('reasoning').disabled = locked || state.busy || $('thinking').value === 'off';
  $('thinkLabel').textContent = $('thinking').value === 'off' ? 'Think Off' : 'Think ' + ({low: 50, high: 75, xhigh: 75, max: 100}[$('reasoning').value] || 75);
  $('thinkingLock').textContent = locked ? 'Locked for this conversation. Start a new chat to change.' : '';
  $('toolsLabel').textContent = 'Tools ' + {auto: 'Auto', ask: 'Ask', off: 'Off'}[$('toolsMode').value];
  $('discardPending').hidden = !state.session?.pending || !state.session?.problem;
  $('skipTools').hidden = !state.session || !pendingCalls().length || !!state.session.problem;
  $('diagnostics').textContent = JSON.stringify({runtime: state.runtime, session: state.session?.record, count: state.session?.count, pending: state.session?.pending, effects: state.session?.effects, persistence: state.session?.pendingRestore}, null, 2);
  termination();
}
async function menus() {
  $('sessions').replaceChildren();
  for (const session of state.session ? [state.session] : []) {
    const option = document.createElement('option'); option.value = session.id;
    option.textContent = session.title || session.id; option.selected = session.id === state.session?.id;
    $('sessions').appendChild(option);
  }
  if (!$('settingsDialog').open) return;
  const selected = $('saves').value;
  $('saves').replaceChildren();
  for (const item of await ChatStore.metadata('saves')) {
    const option = document.createElement('option'); option.value = item.id;
    option.textContent = `${item.title} · ${new Date(item.savedAt).toLocaleString()}`; option.selected = item.id === selected;
    $('saves').appendChild(option);
  }
}
function render() {
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
  controls(); followLatest();
}
function termination() {
  document.querySelectorAll('.termination').forEach(node => node.remove());
  const s = state.session; if (!s || state.busy) return;
  let label = '', actions = [];
  if (['unrecoverable', 'closed'].includes(s.record?.state)) { label = 'This conversation cannot safely continue. Start a new chat or restore a saved state.'; actions = [['New chat', () => $('newSession').click()], ['Recovery', () => $('settingsDialog').showModal()]]; }
  else if (s.record?.state === 'busy') { label = 'The response is still running. Wait for it to settle or Stop.'; actions = [['Reconcile', () => $('reconnect').click()], ['Stop', () => $('stop').click()]]; }
  else if (s.pendingRestore) { label = 'Restore is not yet confirmed. It will not be automatically repeated.'; actions = [['Reconcile', () => $('reconnect').click()], ['Details', () => $('settingsDialog').showModal()]]; }
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
async function pauseTools() { await exclusive(async () => { state.session.toolPause = 'stopped'; state.session.interrupted = true; await save(); }); }
async function continueResponse() { await exclusive(async () => { await reconcile(); if (state.stop) { notice('Stopped before continuing.'); return; } if (state.session.problem || state.session.pending || pendingCalls().length || state.session.effects?.state === 'reserved') throw new Error('Reconcile the current outcome first'); state.session.toolPause = ''; await generate(); await toolLoop(); }); }
async function continueTools() { await exclusive(async () => { await reconcile(); if (state.stop) { notice('Stopped before continuing tools.'); return; } const s = state.session; if (s.problem || s.pending || s.effects?.state === 'reserved') throw new Error('Reconcile the tool outcome first'); if (s.toolAwaitingResponse && !pendingCalls().length) await generate(); await toolLoop(true); }); }
async function select(id) {
  state.session = await ChatStore.get('sessions', id);
  localStorage.setItem('ds41f.selected', id);
  visible = 100; state.attachments = []; rendered.clear(); follow = true;
  const reasoning = state.session?.protocol?.reasoning;
  if (reasoning) { $('thinking').value = reasoning === 'none' ? 'off' : 'on'; $('reasoning').value = reasoning === 'none' ? 'max' : reasoning === 'xhigh' ? 'high' : reasoning; }
  preview(); render();
  state.busy = true; setPhase('Recovering');
  try { await reconcile(true); } finally { state.busy = false; phase = ''; controls(); }
  await menus();
}
async function status() {
  const result = await api('/api/status');
  state.tools = result.tools || []; state.runtime = result.runtime;
  controls();
}
function appendResults(result) {
  const s = state.session;
  for (const message of result.messages) {
    // Only current canonical batch; each result is appended once.
    if (!s.messages.some(m => m.role === 'tool' && m.tool_call_id === message.tool_call_id)) s.messages.push(message);
  }
  s.effects = {count: s.count, state: 'completed'};
  s.toolAwaitingResponse = true;
  s.capacityStop = result.budget_error || '';
  const failure = result.results?.find(item => item.error);
  s.toolPause = failure ? 'error' : ''; s.toolError = failure ? (typeof failure.error === 'string' ? failure.error : JSON.stringify(failure.error)) : '';
}
async function reconcile(wait = false) {
  const s = state.session;
  if (!s) return;
  try {
    let rec;
    for (let i = 0; ; i++) {
      rec = await api(path(s.id));
      if (!wait || rec.state !== 'busy' || i >= 600) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    $('sessionInfo').textContent = `${s.id} · ${rec.state} · turns ${rec.request_count} · frontier ${rec.diagnostics?.m8?.frontier ?? 0} · replay ${rec.diagnostics?.m8?.total_prompt_replay_count ?? 0} · repack ${rec.diagnostics?.m8?.total_full_cache_repack_count ?? 0}`;
    if (rec.state === 'busy') { s.problem = 'Runtime request is still active; wait or Stop with its active request identity.'; }
    else if (s.pendingRestore && rec.state === 'idle') {
      if (rec.diagnostics?.m8?.frontier !== s.pendingRestore.frontier) throw new Error('Restored application/runtime frontier mismatch');
      s.count = rec.request_count; s.pendingRestore = null; s.problem = '';
      notice('Native restore reconciled by its known session ID; no prompt replay.');
    }
    else if (s.pendingRestore) { s.problem = 'Restore has no confirmed idle publication yet. Keep this session ID and reconcile; never automatically repeat restore.'; }
    else if (rec.state === 'unrecoverable' || rec.state === 'closed') { s.problem = 'Protocol/state cannot safely continue. Close this native session; no automatic replay.' + (rec.last_error ? '\n' + rec.last_error : ''); }
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
    s.problem = error.status === 404 ? (s.pendingRestore ? 'Restore outcome not published at the known session ID. Reconcile later; no automatic restore retry.' : 'Runtime session is absent (closed or server restarted). Select a saved entry and Restore as a NEW native session.') : error.message;
    notice(s.problem); controls(); throw error;
  }
}
async function exclusive(action) {
  if (state.busy) return;
  const name = 'ds41f.session.' + (state.session?.id || 'lifecycle');
  await ChatPlatform.withLock(name, async () => {
    state.busy = true; state.stop = false; phase = 'Recovering'; controls();
    try {
      if (state.session) state.session = await ChatStore.get('sessions', state.session.id);
      await action();
    } catch (error) { notice(error.message); }
    finally { state.busy = false; phase = ''; state.controller = null; state.live = null; controls(); followLatest(); await menus(); }
  });
}
function settings() {
  const temperature = Number($('temperature').value);
  const max_tokens = $('outputLimit').value === 'auto' ? 'auto' : Number($('maxTokens').value), top_p = Number($('topP').value);
  if ((max_tokens !== 'auto' && (!Number.isInteger(max_tokens) || max_tokens < 1 || max_tokens > 4294967295)) || !Number.isFinite(temperature) || temperature < 0 || !Number.isFinite(top_p) || top_p < 0 || top_p > 1) throw new Error('Invalid temperature, output budget or top-p');
  return {temperature, top_p, max_tokens, reasoning_effort: $('thinking').value === 'off' ? 'none' : $('reasoning').value};
}
function generationRequest() {
  const s = state.session, options = settings();
  s.protocol ||= {tools: state.tools, reasoning: options.reasoning_effort};
  return {model: 'deepseek-v4.1-flash', messages: s.messages, stream: true, ...options,
    reasoning_effort: s.protocol.reasoning, tools: s.protocol.tools, tool_choice: 'auto'};
}
async function generate() {
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
  await save(); // frozen ordinary history durable BEFORE request/effect
  state.controller = new AbortController();
  s.visibleUnsettled = null;
  state.live = {role: 'assistant', content: '', reasoning_content: ''};
  liveNode = ChatRender.message($('messages'), state.live, [], true);
  setPhase('Generating');
  // Append to persistent text nodes while streaming; Markdown parses once at settlement.
  const paintTimer = setInterval(() => { if (state.live) { ChatRender.updateLive(liveNode, state.live); followLatest(); } }, 80);
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
function pendingCalls() {
  const s = state.session;
  const last = [...(s?.messages || [])].reverse().find(m => m.role === 'assistant');
  return (last?.tool_calls || []).filter(call => !s.messages.some(m => m.role === 'tool' && m.tool_call_id === call.id));
}
async function toolLoop(explicit = false) {
  const s = state.session;
  if (!pendingCalls().length) return;
  if (!explicit && $('toolsMode').value !== 'auto') { s.toolPause = 'approval'; await save(); return; }
  if (explicit && $('toolsMode').value === 'off') { notice('Tools are Off. Choose Ask or Auto to execute them.'); return; }
  const ceiling = 128; // runaway circuit breaker, not an ordinary UX round quota
  s.toolPause = ''; s.toolRounds = 0;
  for (let round = 0; round < ceiling && !state.stop; round++) {
    const calls = pendingCalls(); if (!calls.length || state.session.problem) return;
    if ($('toolsMode').value === 'off') { s.toolPause = 'approval'; await save(); return; }
    if (state.session.effects?.state === 'reserved') { notice('Tool reservation is uncertain; no automatic re-execution'); return; }
    setPhase('Using tool'); notice(calls.some(c => c.function.name === 'web_search') ? 'Searching web…' : 'Fetching URL…');
    state.session.effects = {count: state.session.count, state: 'reserved'}; await save();
    let result;
    try { result = await api('/api/tools', 'POST', {session_id: state.session.id, request_count: state.session.count, request: generationRequest()}); }
    catch (error) { s.toolPause = 'error'; s.toolError = error.message; await save(); throw error; }
    appendResults(result); s.toolRounds = round + 1; await save(); render(); notice('');
    if (result.budget_error) { s.capacityStop = result.budget_error; await save(); render(); return; }
    const failed = result.results?.find(item => item.error);
    if (failed) { s.toolPause = 'error'; s.toolError = typeof failed.error === 'string' ? failed.error : JSON.stringify(failed.error); await save(); return; }
    if (state.stop) { s.interrupted = true; s.interruptions ||= []; const index = s.messages.findLastIndex(m => m.role === 'assistant'); if (!s.interruptions.includes(index)) s.interruptions.push(index); s.interruptionKinds ||= {}; s.interruptionKinds[index] = 'user'; await save(); render(); return; }
    await generate();
    if ($('toolsMode').value === 'ask' && pendingCalls().length) { s.toolPause = 'approval'; await save(); return; }
  }
  if (pendingCalls().length && !state.stop) { s.toolPause = 'ceiling'; await save(); }
  if (state.stop) { s.interrupted = true; await save(); }
}
async function send() {
  const text = $('input').value.trim();
  if (!text && !state.attachments.length) return;
  await exclusive(async () => {
    settings(); await reconcile();
    if (state.stop) { notice('Stopped before sending. Your draft was kept.'); return; }
    const s = state.session;
    if (s.problem || s.pending || s.effects?.state === 'reserved') throw new Error(s.problem || 'Unresolved application request/effect');
    if (pendingCalls().length || s.toolAwaitingResponse) throw new Error('Continue the pending tool response before another user turn');
    const parts = [];
    if (text) parts.push({type: 'text', text});
    for (const file of state.attachments) parts.push({type: 'image_url', image_url: {url: file.url}});
    s.toolPause = ''; s.toolError = ''; s.interrupted = false;
    s.messages.push({role: 'user', content: state.attachments.length ? parts : text});
    s.title ||= text.slice(0, 60) || 'Image conversation';
    s.pending = {base: s.count, requestId: null, nonce: ChatPlatform.uuid(), addedUser: true};
    await save(); $('input').value = ''; resizeInput(); state.attachments = []; follow = true; preview(); render();
    await generate(); await toolLoop();
  });
}
async function attach(files) {
  if (state.busy) return;
  try {
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
    state.attachments.push(...additions); preview(); notice('');
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
$('composer').ondragover = event => event.preventDefault();
$('composer').ondrop = event => { event.preventDefault(); attach([...event.dataTransfer.files]); };
$('input').onpaste = event => { const files = [...event.clipboardData.items].filter(item => item.kind === 'file').map(item => item.getAsFile()).filter(Boolean); if (files.length) { event.preventDefault(); attach(files); } };
function resizeInput() { layoutScrollUntil = performance.now() + 200; $('input').style.height = 'auto'; $('input').style.height = Math.min(190, $('input').scrollHeight) + 'px'; }
$('input').oninput = resizeInput;
let composing = false;
$('input').addEventListener('compositionstart', () => { composing = true; });
$('input').addEventListener('compositionend', () => { composing = false; });
$('input').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && !composing && event.keyCode !== 229) { event.preventDefault(); if (!state.busy && !$('send').disabled) $('composer').requestSubmit(); } };
$('stop').onclick = async () => {
  state.stop = true; setPhase('Stopping'); notice('Waiting for safe cancellation boundary…');
  const s = state.session;
  if (s?.pending) { s.pending.userStopRequested = true; try { await save(); } catch (_) { notice('Could not record Stop intent. Requesting safe cancellation…'); } }
  try {
    const id = s?.pending?.requestId || (await api(path(s.id))).active_request_id;
    if (id) await api(path(s.id) + '/cancel', 'POST', {request_id: id});
  } catch (error) { notice('Stop: ' + error.message + '; reconciling via GET'); }
  finally { state.controller?.abort(); if (!state.busy) await exclusive(async () => { await reconcile(true); }); }
};
$('newSession').onclick = () => exclusive(async () => {
  if (state.session) {
    if (!confirm('Start a new chat?\n\nThe current native conversation will be closed.' + (state.session.pending || state.session.problem || state.session.effects?.state === 'reserved' ? '\nThe current outcome is unresolved. Closing retires it; it will not be retried.' : ''))) return;
    try { await api(path(state.session.id), 'DELETE'); } catch (error) { if (error.status !== 404) throw error; }
    await ChatStore.remove('sessions', state.session.id); state.session = null; localStorage.removeItem('ds41f.selected'); render();
  }
  const rec = await api('/api/session', 'POST', {});
  await ChatStore.put('sessions', {id: rec.id, title: '', messages: [], count: 0, frontier: 0, problem: ''});
  await select(rec.id);
});
$('sessions').onchange = () => select($('sessions').value).catch(error => notice(error.message));
$('reconnect').onclick = () => { if (!state.busy) return exclusive(async () => { await status(); await reconcile(true); }); };
$('closeSession').onclick = () => exclusive(async () => {
  if (!state.session || !confirm('Close the native runtime session? Saved artifacts remain available.')) return;
  try { await api(path(state.session.id), 'DELETE'); } catch (error) { if (error.status !== 404) throw error; }
  await ChatStore.remove('sessions', state.session.id); state.session = null; localStorage.removeItem('ds41f.selected'); render();
});
$('saveSession').onclick = () => exclusive(async () => {
  const rec = await reconcile(); const s = state.session;
  if (rec.state !== 'idle' || s.pending || s.problem || pendingCalls().length) throw new Error('Save requires a settled idle application/runtime frontier, with completed tool results');
  const result = rec.persisted_artifact?.frontier === s.frontier ? {artifact: rec.persisted_artifact} : await api(path(s.id) + '/persist', 'POST', {});
  if (result.artifact.frontier !== s.frontier) throw new Error('Native artifact frontier changed; application snapshot NOT paired. Reconcile before saving. Artifact: ' + result.artifact.path);
  await ChatStore.put('saves', {id: s.id, title: s.title || 'Conversation', messages: s.messages, interruptions: s.interruptions || [], interruptionKinds: s.interruptionKinds || {}, lateStops: s.lateStops || [], endings: s.endings || {}, finishReason: s.finishReason, terminationReason: s.terminationReason, capacity: s.capacity, capacityStop: s.capacityStop, interrupted: !!s.interrupted, toolAwaitingResponse: !!s.toolAwaitingResponse, toolPause: s.toolPause || '', toolError: s.toolError || '', toolRounds: s.toolRounds || 0, protocol: s.protocol, frontier: s.frontier, artifact: result.artifact.path, savedAt: Date.now()});
  notice('Saved native idle artifact and matching original-byte browser history. Later chat does not change this snapshot. Keep this browser storage: native artifacts do not contain historical image bytes.');
});
$('restoreSession').onclick = () => exclusive(async () => {
  if (state.session?.pendingRestore) throw new Error('An existing restore outcome is unresolved. Reconcile its known ID before another restore.');
  const snapshot = await ChatStore.get('saves', $('saves').value);
  if (!snapshot) throw new Error('Choose a saved session. Native artifacts alone do not contain ordinary image/history payloads.');
  if (state.session) {
    if (!confirm('Restore saved state?\n\nThe current native conversation will be closed. Unresolved outcomes will not be retried.')) return;
    try { await api(path(state.session.id), 'DELETE'); } catch (error) { if (error.status !== 404) throw error; }
    await ChatStore.remove('sessions', state.session.id); state.session = null; localStorage.removeItem('ds41f.selected');
  }
  const id = 'sess_' + ChatPlatform.uuid().replaceAll('-', '');
  const journal = {id, title: snapshot.title + ' (restored)', messages: snapshot.messages,
    interruptions: snapshot.interruptions || [], interruptionKinds: snapshot.interruptionKinds || {}, lateStops: snapshot.lateStops || [], endings: snapshot.endings || {}, finishReason: snapshot.finishReason, terminationReason: snapshot.terminationReason, capacity: snapshot.capacity, capacityStop: snapshot.capacityStop, interrupted: !!snapshot.interrupted, toolAwaitingResponse: !!snapshot.toolAwaitingResponse, toolPause: snapshot.toolPause || '', toolError: snapshot.toolError || '', toolRounds: snapshot.toolRounds || 0, protocol: snapshot.protocol, count: 0,
    frontier: snapshot.frontier, pendingRestore: {frontier: snapshot.frontier, artifact: snapshot.artifact},
    problem: 'Restore pending; outcome must be reconciled before continuation'};
  await ChatStore.put('sessions', journal); // known standard-OFF session ID BEFORE restore I/O
  state.session = journal; localStorage.setItem('ds41f.selected', id); render();
  const rec = await api('/api/session/restore', 'POST', {artifact_path: snapshot.artifact, id});
  if (rec.id !== id || rec.diagnostics?.m8?.frontier !== snapshot.frontier) throw new Error('Saved application/runtime frontier mismatch; do not continue');
  await reconcile(); notice('Fresh native restore; no prompt replay or historical image re-encoding.');
});
$('discardPending').onclick = () => exclusive(async () => {
  const s = state.session; if (!s?.pending) return;
  const rec = await api(path(s.id));
  if (!['empty', 'idle'].includes(rec.state) || rec.request_count !== s.pending.base) throw new Error('Runtime outcome is not unadmitted; reconcile rather than discard');
  if (!confirm('Discard only this UI request, which has no recorded committed turn? It will NOT be sent again.')) return;
  if (s.pending.addedUser) s.messages.pop();
  s.pending = null; s.problem = ''; await save(); render(); notice('Unadmitted UI request discarded.');
});
$('reasoning').onchange = $('thinking').onchange = () => {
  const protocol = state.session?.protocol;
  if (protocol) { $('thinking').value = protocol.reasoning === 'none' ? 'off' : 'on'; $('reasoning').value = protocol.reasoning === 'none' ? 'max' : protocol.reasoning === 'xhigh' ? 'high' : protocol.reasoning; }
  controls(); persistPreferences();
};
$('toolsMode').onchange = () => { controls(); persistPreferences(); };
$('settingsButton').onclick = async () => { $('settingsDialog').showModal(); try { await menus(); } catch (error) { notice(error.message); } };
$('settingsClose').onclick = () => $('settingsDialog').close();
$('outputLimit').onchange = () => { const custom = $('outputLimit').value === 'custom'; $('maxTokens').hidden = !custom; if (!custom && $('outputLimit').value !== 'auto') $('maxTokens').value = $('outputLimit').value; persistPreferences(); };
function persistPreferences() { const values = {}; for (const id of ['temperature', 'topP', 'outputLimit', 'maxTokens', 'thinking', 'reasoning', 'toolsMode']) values[id] = $(id).value; localStorage.setItem('ds41f.preferences', JSON.stringify(values)); }
for (const id of ['temperature', 'topP', 'maxTokens']) $(id).onchange = persistPreferences;
$('resetGeneration').onclick = () => { $('temperature').value = '0'; $('topP').value = '0'; $('outputLimit').value = 'custom'; $('maxTokens').value = '128'; $('maxTokens').hidden = false; persistPreferences(); };
$('skipTools').onclick = () => exclusive(async () => {
  await reconcile(); if (state.session.problem) throw new Error(state.session.problem);
  if (state.stop) { notice('Stopped before continuing.'); return; }
  if (!confirm('Do not execute/retry these tools. Return explicit unavailable results to the model?')) return;
  for (const call of pendingCalls()) state.session.messages.push({role: 'tool', tool_call_id: call.id, content: JSON.stringify({error: 'Client tool result unavailable; NOT executed/retried by recovery'})});
  state.session.effects = {count: state.session.count, state: 'completed'}; await save(); render();
  await generate(); await toolLoop();
});
(async () => {
  try {
    try { const values = JSON.parse(localStorage.getItem('ds41f.preferences') || '{}'); for (const [id, value] of Object.entries(values)) if ($(id)) $(id).value = value; $('maxTokens').hidden = $('outputLimit').value !== 'custom'; } catch (_) { /* Corrupt presentation preferences do not authorize runtime work. */ }
    await status(); await menus();
    const id = localStorage.getItem('ds41f.selected');
    if (id && await ChatStore.get('sessions', id)) await select(id);
    else { const rec = await api('/api/session', 'POST', {}); await ChatStore.put('sessions', {id: rec.id, title: '', messages: [], count: 0, frontier: 0, problem: ''}); await select(rec.id); }
    controls();
  } catch (error) { notice(error.message); }
})();
setInterval(() => status().catch(error => { state.runtime = {status: 'error', error: error.message}; controls(); notice('Cannot connect to runtime. Open Settings → Recovery to reconnect.'); }), 10000);
