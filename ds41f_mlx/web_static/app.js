/* Thin stateful application. IndexedDB is UI history; runtime state is canonical. */
'use strict';
const $ = id => document.getElementById(id);
const state = {session: null, busy: false, stop: false, attachments: [], tools: [], runtime: null, live: null, controller: null};
let visible = 100;
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
  $('send').disabled ||= !state.session || !!state.session.problem || !!state.session.pending || state.session.effects?.state === 'reserved';
  $('stop').disabled = !state.busy || !!state.session?.pendingRestore;
}
async function menus() {
  const sessions = await ChatStore.all('sessions');
  $('sessions').replaceChildren();
  for (const session of sessions) {
    const option = document.createElement('option'); option.value = session.id;
    option.textContent = session.title || session.id; option.selected = session.id === state.session?.id;
    $('sessions').appendChild(option);
  }
  $('saves').replaceChildren();
  for (const item of await ChatStore.all('saves')) {
    const option = document.createElement('option'); option.value = item.id;
    option.textContent = `${item.title} · ${new Date(item.savedAt).toLocaleString()} · frontier ${item.frontier}`;
    $('saves').appendChild(option);
  }
}
function render() {
  $('messages').replaceChildren();
  const messages = state.session?.messages || [];
  const start = Math.max(0, messages.length - visible);
  if (start) {
    const older = document.createElement('button'); older.textContent = `Show 100 earlier messages (${start} hidden)`;
    older.onclick = () => { visible += 100; render(); }; $('messages').appendChild(older);
  }
  for (let index = start; index < messages.length; index++) {
    const node = ChatRender.message($('messages'), messages[index]);
    if (state.session.interruptions?.includes(index)) {
      const label = document.createElement('div'); label.className = 'small';
      label.textContent = 'Interrupted · canonical committed partial response'; node.appendChild(label);
    }
  }
  if (state.live) ChatRender.message($('messages'), state.live);
  else if (state.session?.visibleUnsettled) {
    const node = ChatRender.message($('messages'), state.session.visibleUnsettled);
    const label = document.createElement('div'); label.className = 'error';
    label.textContent = 'Unsettled / unrecoverable visible output · display only, NOT tool execution permission';
    node.appendChild(label);
  }
  controls();
}
async function select(id) {
  state.session = await ChatStore.get('sessions', id);
  localStorage.setItem('ds41f.selected', id);
  visible = 100; state.attachments = [];
  $('reasoning').value = state.session?.protocol?.reasoning || 'none';
  preview(); render();
  await reconcile(); await menus();
}
async function status() {
  const result = await api('/api/status');
  state.tools = result.tools || []; state.runtime = result.runtime;
  $('status').textContent = `Runtime ${result.runtime.status}${result.runtime.error ? ': ' + result.runtime.error : ''} · tools: ${state.tools.map(x => x.function.name).join(', ')}`;
}
function appendResults(result) {
  const s = state.session;
  for (const message of result.messages) {
    // Only current canonical batch; each result is appended once.
    if (!s.messages.some(m => m.role === 'tool' && m.tool_call_id === message.tool_call_id)) s.messages.push(message);
  }
  s.effects = {count: s.count, state: 'completed'};
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
      s.messages.push(turn.response_json.choices[0].message);
      if (turn.cancelled) { s.interruptions ||= []; s.interruptions.push(s.messages.length - 1); }
      s.count = rec.request_count; s.pending = null; s.problem = ''; s.visibleUnsettled = null;
      s.interrupted = !!turn.cancelled;
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
  await navigator.locks.request(name, {ifAvailable: true}, async lock => {
    if (!lock) throw new Error('Another browser tab is using this conversation');
    state.busy = true; state.stop = false; controls();
    try {
      if (state.session) state.session = await ChatStore.get('sessions', state.session.id);
      await action();
    } catch (error) { notice(error.message); }
    finally { state.busy = false; state.controller = null; state.live = null; controls(); await menus(); }
  });
}
function settings() {
  const temperature = {precise: 0, balanced: .6, creative: .9}[$('preset').value];
  const max_tokens = Number($('maxTokens').value), top_p = Number($('topP').value);
  if (!Number.isInteger(max_tokens) || max_tokens < 1 || max_tokens > 4096 || !Number.isFinite(top_p) || top_p < 0 || top_p > 1) throw new Error('Invalid output budget or top-p');
  return {temperature, top_p, max_tokens, reasoning_effort: $('reasoning').value};
}
async function generate() {
  const s = state.session;
  s.protocol ||= {tools: state.tools, reasoning: $('reasoning').value};
  const request = {model: 'deepseek-v4.1-flash', messages: s.messages, stream: true, ...settings(),
    reasoning_effort: s.protocol.reasoning, tools: s.protocol.tools, tool_choice: 'auto'};
  // Recipe tool declarations are part of the historical prompt prefix. Keep
  // them stable; the checkbox controls client execution, not prefix rewriting.
  s.pending ||= {base: s.count, requestId: null, nonce: crypto.randomUUID(), addedUser: false};
  await save(); // frozen ordinary history durable BEFORE request/effect
  state.controller = new AbortController();
  s.visibleUnsettled = null;
  state.live = {role: 'assistant', content: '', reasoning_content: ''};
  let liveNode = ChatRender.message($('messages'), state.live), lastPaint = 0;
  const paint = () => {
    const expanded = [...liveNode.querySelectorAll('details')].map(node => node.open);
    const parent = document.createElement('div'); const next = ChatRender.message(parent, state.live);
    [...next.querySelectorAll('details')].forEach((node, index) => { node.open = expanded[index] || false; });
    liveNode.replaceWith(next); liveNode = next;
  };
  try {
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
      if (performance.now() - lastPaint > 50) { paint(); lastPaint = performance.now(); }
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
  const last = [...s.messages].reverse().find(m => m.role === 'assistant');
  return (last?.tool_calls || []).filter(call => !s.messages.some(m => m.role === 'tool' && m.tool_call_id === call.id));
}
async function toolLoop() {
  if (!$('toolsEnabled').checked) {
    if (pendingCalls().length) notice('Client tool execution paused. Enable execution and run requested tools, or return explicit unavailable results.');
    return;
  }
  for (let round = 0; round < 4 && !state.stop; round++) {
    const calls = pendingCalls(); if (!calls.length || state.session.problem) return;
    if (state.session.effects?.state === 'reserved') { notice('Tool reservation is uncertain; no automatic re-execution'); return; }
    notice('Calling: ' + calls.map(c => c.function.name).join(', '));
    state.session.effects = {count: state.session.count, state: 'reserved'}; await save();
    const result = await api('/api/tools', 'POST', {session_id: state.session.id, request_count: state.session.count});
    appendResults(result); await save(); render();
    for (const display of result.results) ChatRender.activity($('messages'), display);
    if (state.stop) return;
    await generate();
  }
  if (pendingCalls().length && !state.stop) notice('Client tool round limit reached; requested tools were not re-executed. Continue explicitly.');
}
async function send() {
  const text = $('input').value.trim();
  if (!text && !state.attachments.length) return;
  await exclusive(async () => {
    await reconcile();
    const s = state.session;
    if (s.problem || s.pending || s.effects?.state === 'reserved') throw new Error(s.problem || 'Unresolved application request/effect');
    if (pendingCalls().length) throw new Error('Resolve outstanding canonical tool calls before another user turn');
    const parts = [];
    if (text) parts.push({type: 'text', text});
    for (const file of state.attachments) parts.push({type: 'image_url', image_url: {url: file.url}});
    s.messages.push({role: 'user', content: state.attachments.length ? parts : text});
    s.title ||= text.slice(0, 60) || 'Image conversation';
    s.pending = {base: s.count, requestId: null, nonce: crypto.randomUUID(), addedUser: true};
    await save(); $('input').value = ''; state.attachments = []; preview(); render();
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
    state.attachments.push(...additions); preview(); notice('Original image bytes attached. Runtime validates animation/container/context before admission.');
  } catch (error) { notice(error.message); }
}
function preview() {
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
$('input').ondragover = event => event.preventDefault();
$('input').ondrop = event => { event.preventDefault(); attach([...event.dataTransfer.files]); };
$('stop').onclick = async () => {
  state.stop = true;
  const s = state.session;
  try {
    const id = s?.pending?.requestId || (await api(path(s.id))).active_request_id;
    if (id) await api(path(s.id) + '/cancel', 'POST', {request_id: id});
  } catch (error) { notice('Stop: ' + error.message + '; reconciling via GET'); }
  finally { state.controller?.abort(); }
};
$('newSession').onclick = () => exclusive(async () => {
  const rec = await api('/api/session', 'POST', {});
  await ChatStore.put('sessions', {id: rec.id, title: '', messages: [], count: 0, frontier: 0, problem: ''});
  await select(rec.id);
});
$('sessions').onchange = () => select($('sessions').value).catch(error => notice(error.message));
$('reconnect').onclick = () => { if (!state.busy) exclusive(async () => { await status(); await reconcile(true); }); };
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
  await ChatStore.put('saves', {id: s.id, title: s.title || s.id, messages: s.messages, interruptions: s.interruptions || [], protocol: s.protocol, frontier: s.frontier, artifact: result.artifact.path, savedAt: Date.now()});
  notice('Saved native idle artifact and matching original-byte browser history. Later chat does not change this snapshot. Keep this browser storage: native artifacts do not contain historical image bytes.');
});
$('restoreSession').onclick = () => exclusive(async () => {
  if (state.session?.pendingRestore) throw new Error('An existing restore outcome is unresolved. Reconcile its known ID before another restore.');
  const snapshot = await ChatStore.get('saves', $('saves').value);
  if (!snapshot) throw new Error('Choose a saved session. Native artifacts alone do not contain ordinary image/history payloads.');
  const id = 'sess_' + crypto.randomUUID().replaceAll('-', '');
  const journal = {id, title: snapshot.title + ' (restored)', messages: snapshot.messages,
    interruptions: snapshot.interruptions || [], protocol: snapshot.protocol, count: 0,
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
$('runTools').onclick = () => exclusive(async () => { await reconcile(); await toolLoop(); });
$('reasoning').onchange = () => {
  if (state.session?.protocol && $('reasoning').value !== state.session.protocol.reasoning) {
    $('reasoning').value = state.session.protocol.reasoning;
    notice('Reasoning protocol is fixed for this history. Choose the desired mode before the first turn of a new session.');
  }
};
$('skipTools').onclick = () => exclusive(async () => {
  await reconcile(); if (state.session.problem) throw new Error(state.session.problem);
  if (!confirm('Do not execute/retry these tools. Return explicit unavailable results to the model?')) return;
  for (const call of pendingCalls()) state.session.messages.push({role: 'tool', tool_call_id: call.id, content: JSON.stringify({error: 'Client tool result unavailable; NOT executed/retried by recovery'})});
  state.session.effects = {count: state.session.count, state: 'completed'}; await save(); render();
  await generate(); await toolLoop();
});
(async () => {
  try {
    await status(); await menus();
    const id = localStorage.getItem('ds41f.selected');
    if (id && await ChatStore.get('sessions', id)) await select(id);
    else notice('Create or select a conversation. Legacy localStorage transcripts are not automatically replayed.');
    controls();
  } catch (error) { notice(error.message); }
})();
setInterval(() => status().catch(error => notice(error.message)), 10000);
