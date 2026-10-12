/* Document previews are UI resources, never conversation image payloads. */
'use strict';
let selectedDocument = null;
function renderDocuments(ctx = state) {
  if (ctx !== windowActive()) return;
  const docs = ctx.session?.documents || [];
  $('documentCount').textContent = docs.length ? `(${docs.length})` : '';
  $('documents').replaceChildren();
  $('documentCards').replaceChildren();
  for (const doc of docs) {
    const row = document.createElement('div'), label = document.createElement('span'), open = document.createElement('button');
    row.className = 'documentContext';
    const pages = [...new Set(doc.readPages || [])].sort((a, b) => a - b);
    const ranges = [];
    for (let i = 0; i < pages.length; i++) {
      const start = pages[i]; let end = start;
      while (pages[i + 1] === end + 1) end = pages[++i];
      ranges.push(start === end ? String(start) : `${start}–${end}`);
    }
    label.textContent = `${doc.name || 'PDF'} · ${doc.page_count || '?'} pages\n${doc.state || 'unknown'} · ${(doc.pageStates || []).filter(p => p.observation_state === 'completed').length} derived observations\nReferenced pages: ${ranges.join(', ') || 'none'}`;
    open.type = 'button'; open.textContent = 'Open / source';
    open.onclick = () => {
      selectedDocument = doc; $('documentTitle').textContent = `${doc.name || 'PDF'} · source SHA256 ${doc.sha256 || doc.artifact_id}`;
      $('documentPageNumber').max = doc.page_count || 10000; $('documentPageNumber').value = 1;
      $('documentOriginal').href = `/api/documents/${doc.artifact_id}/original`;
      $('documentPage').replaceChildren(); $('documentDialog').showModal();
    };
    row.append(label, open);
    const pageDetails = document.createElement('details'), summary = document.createElement('summary');
    summary.textContent = 'Referenced page list'; pageDetails.appendChild(summary);
    const filter = document.createElement('input'); filter.placeholder = 'Pages: 12, 12-18'; filter.setAttribute('aria-label', 'Filter referenced pages by number or range'); pageDetails.appendChild(filter);
    const links = document.createElement('div'); pageDetails.appendChild(links);
    filter.oninput = () => {
      const terms = filter.value.split(',').map(t => t.trim());
      for (const link of links.children) {
        const number = Number(link.dataset.page);
        link.hidden = !!filter.value.trim() && !terms.some(term => {
          const match = /^(\d+)(?:\s*[-–]\s*(\d+))?$/.exec(term);
          return match && number >= Number(match[1]) && number <= Number(match[2] || match[1]);
        });
      }
    };
    for (const number of pages) {
      const pageLink = document.createElement('button'); pageLink.type = 'button'; pageLink.textContent = `p. ${number}`;
      pageLink.onclick = () => { open.onclick(); $('documentPageNumber').value = number; $('openDocumentPage').click(); };
      pageLink.dataset.page = number; links.appendChild(pageLink);
    }
    row.appendChild(pageDetails);
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Remove reference'; remove.dataset.action = 'remove-document'; remove.disabled = state.busy;
    remove.onclick = () => exclusive(async () => { state.session.documents = state.session.documents.filter(d => d.artifact_id !== doc.artifact_id); await save(); renderDocuments(); });
    row.appendChild(remove); $('documents').appendChild(row);
    const card = document.createElement('div'); card.className = 'documentCard';
    const show = document.createElement('button'); show.type = 'button';
    show.textContent = `📄 ${doc.name || 'PDF'} · ${doc.page_count || '?'} pages`;
    show.onclick = () => $('documentsDrawer').showModal();
    const detach = document.createElement('button'); detach.type = 'button'; detach.textContent = '×';
    detach.setAttribute('aria-label', `Remove reference to ${doc.name || 'PDF'}`);
    detach.dataset.action = 'remove-document'; detach.disabled = state.busy; detach.onclick = remove.onclick;
    card.append(show, detach); $('documentCards').appendChild(card);
  }
}
async function refreshDocuments(ctx = state) {
  const {notice, save} = scoped(ctx);
  const s = ctx.session; if (!s) return;
  for (const doc of s.documents || []) {
    try {
      const info = await api(`/api/documents/${doc.artifact_id}`);
      Object.assign(doc, {name: info.filename || doc.name, page_count: info.page_count, state: info.state,
        sha256: info.sha256, expires_at: info.expires_at, pageStates: info.pages || []});
      if (info.error) notice(info.error);
    } catch (error) { doc.state = error.status === 400 ? 'unavailable' : 'unconfirmed'; notice(error.message + '; no upload retry'); }
  }
  await save(); renderDocuments(ctx);
}
async function addPDF(file, owner = state) {
  if (!file || owner.busy || !owner.session) { notice('Create a chat before adding PDF', owner); return; }
  await exclusive(async ctx => {
    const state = ctx;
    const {save, notice} = scoped(ctx);
    if (file.size > 64 * 1024 * 1024) throw new Error('Local PDF upload ceiling is 64 MiB (public URL acquisition remains 8 MiB)');
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (new TextDecoder().decode(bytes.subarray(0, 5)) !== '%PDF-') throw new Error('Not PDF bytes');
    const idBytes = crypto.getRandomValues(new Uint8Array(32));
    const id = [...idBytes].map(b => b.toString(16).padStart(2, '0')).join('');
    const doc = {artifact_id: id, name: Array.from(file.name).slice(0, 200).join(''), state: 'upload-unconfirmed', announced: false};
    state.session.documents ||= []; state.session.documents.push(doc);
    // Known receipt identity is saved BEFORE upload. Never resend on failure.
    await save(); renderDocuments(ctx);
    let binary = ''; for (let i = 0; i < bytes.length; i += 32768) binary += String.fromCharCode(...bytes.subarray(i, i + 32768));
    try { await api('/api/documents', 'POST', {id, name: doc.name, data: btoa(binary)}); }
    catch (error) { notice('Upload unconfirmed: ' + error.message + '. Refresh observes receipt; no automatic retry.'); }
    await refreshDocuments(ctx);
    // Keep the conversation visible; details are opened explicitly from the card.
  }, owner);
}
$('pdfs').onchange = async event => { const file = event.target.files[0]; event.target.value = ''; await addPDF(file); };
$('pdfs').closest('label').onkeydown = event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); if (!state.busy) $('pdfs').click(); } };
$('refreshDocuments').onclick = () => exclusive(refreshDocuments);
window.addEventListener('ds41f-document-page', event => {
  const {artifact_id, page} = event.detail;
  const doc = (state.session?.documents || []).find(d => d.artifact_id === artifact_id);
  if (!doc) { notice('Document reference unavailable; refresh documents.'); return; }
  selectedDocument = doc; $('documentTitle').textContent = `${doc.name || 'PDF'} · source SHA256 ${doc.sha256 || artifact_id}`;
  $('documentPageNumber').max = doc.page_count || 10000; $('documentPageNumber').value = page;
  $('documentOriginal').href = `/api/documents/${artifact_id}/original`;
  if (!$('documentDialog').open) $('documentDialog').showModal();
  $('openDocumentPage').click();
});
$('documentsClose').onclick = () => $('documentsDrawer').close();
$('documentClose').onclick = () => $('documentDialog').close();
$('openDocumentPage').onclick = async () => {
  if (!selectedDocument) return;
  const ctx = state, doc = selectedDocument;
  const page = Number($('documentPageNumber').value), box = $('documentPage');
  box.textContent = 'Loading original-page raster…';
  try {
    const result = await api(`/api/documents/${doc.artifact_id}/pages/${page}`);
    if (ctx !== windowActive() || selectedDocument !== doc) return;
    box.replaceChildren();
    const label = document.createElement('p'); label.textContent = `Original PDF page ${page} raster (parser-rendered); not added to chat Vision history.`; box.appendChild(label);
    if (result.error) { label.textContent = result.error; return; }
    if (result.raster_error) label.textContent = 'Raster unavailable: ' + result.raster_error + '. Parser text shown below.';
    if (result.image) {
      const image = document.createElement('img'); image.src = result.image; image.alt = `PDF page ${page}`; image.style.maxWidth = '100%'; box.appendChild(image);
      if (!production()) {
        const confirm = document.createElement('button'); confirm.type = 'button'; confirm.textContent = 'Add this page to ordinary chat Vision (uses one of 4 history slots)';
        confirm.onclick = async () => {
          const binary = atob(result.image.split(',')[1]), bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
          await attach([new File([bytes], `${doc.name} page ${page} (parser raster)`, {type: 'image/png'})], ctx);
        };
        box.appendChild(confirm);
      }
    }
    const text = document.createElement('pre'); text.style.whiteSpace = 'pre-wrap'; text.textContent = 'Parser-derived text:\n' + (result.pages?.[0]?.text || '[No text layer]'); box.appendChild(text);
    const observation = document.createElement('pre'); observation.style.whiteSpace = 'pre-wrap';
    observation.textContent = 'Model-derived observation — NOT original source:\n' + (result.observation?.observation || result.observation?.state || '[Not interpreted]'); box.appendChild(observation);
  } catch (error) { box.textContent = error.message; }
};
