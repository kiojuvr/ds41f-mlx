/* All model/provider data goes through text nodes. No HTML parsing or network images. */
(() => {
  'use strict';
  const text = (parent, value) => parent.appendChild(document.createTextNode(String(value ?? '')));
  const element = (parent, tag, cls) => { const node = document.createElement(tag); if (cls) node.className = cls; parent.appendChild(node); return node; };
  function sourceURL(value) {
    try { const url = new URL(String(value)); return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null; } catch (_) { return null; }
  }
  function copyButton(parent, value, label = 'Copy') {
    const button = element(parent, 'button'); button.type = 'button'; text(button, label);
    button.onclick = async () => { try { await navigator.clipboard.writeText(value); button.textContent = 'Copied'; } catch (_) { button.textContent = 'Copy unavailable'; } setTimeout(() => { button.textContent = label; }, 1800); };
  }
  function inline(parent, value, depth = 0) {
    // Bounded scans: pathological/unclosed markup remains literal. Raw HTML is text.
    if (depth > 6 || value.length > 20000) { text(parent, value); return; }
    let pos = 0, plain = '';
    const flush = () => { if (plain) text(parent, plain); plain = ''; };
    while (pos < value.length) {
      if (value[pos] === '\\' && pos + 1 < value.length) { plain += value[pos + 1]; pos += 2; continue; }
      if (value[pos] === '[') {
        const end = value.indexOf('](', pos + 1), close = end < 0 ? -1 : value.indexOf(')', end + 2);
        if (end >= 0 && close >= 0) {
          const url = sourceURL(value.slice(end + 2, close));
          if (url) { flush(); const a = element(parent, 'a'); a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; inline(a, value.slice(pos + 1, end), depth + 1); pos = close + 1; continue; }
        }
      }
      const delimiter = value.startsWith('**', pos) ? '**' : ['*', '_', '`'].includes(value[pos]) ? value[pos] : null;
      if (delimiter && !value.startsWith('``', pos)) {
        const end = value.indexOf(delimiter, pos + delimiter.length);
        if (end > pos + delimiter.length) { flush(); const node = element(parent, delimiter === '`' ? 'code' : delimiter === '**' ? 'strong' : 'em'); const body = value.slice(pos + delimiter.length, end); if (delimiter === '`') text(node, body); else inline(node, body, depth + 1); pos = end + delimiter.length; continue; }
      }
      plain += value[pos++];
    }
    flush();
  }
  function prose(parent, value, depth = 0) {
    if (depth >= 16) { text(element(parent, 'div', 'prose'), value); return; }
    const lines = String(value ?? '').split('\n');
    const cells = line => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(x => x.trim());
    const tableRule = line => { const parts = cells(line); return parts.length >= 2 && parts.every(cell => /^:?-{3,}:?$/.test(cell)); };
    const special = line => /^(#{1,6}\s|\s*```|>\s?|\s*[-*+]\s|\s*\d+\.\s|\s*(?:---+|\*\*\*+)\s*$)/.test(line);
    for (let i = 0; i < lines.length;) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      const fence = /^\s*```([A-Za-z0-9_+.-]*)\s*$/.exec(line);
      if (fence) {
        const body = []; i++; while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]); if (i < lines.length) i++;
        const box = element(parent, 'div', 'codeblock'), bar = element(box, 'div', 'codeHeader'); text(element(bar, 'span'), fence[1] || 'text'); copyButton(bar, body.join('\n')); text(element(element(box, 'pre'), 'code'), body.join('\n')); continue;
      }
      const heading = /^(#{1,6})\s+(.*)$/.exec(line);
      if (heading) { inline(element(parent, 'h' + heading[1].length), heading[2]); i++; continue; }
      if (/^\s*(?:---+|\*\*\*+)\s*$/.test(line)) { element(parent, 'hr'); i++; continue; }
      if (/^>/.test(line)) { const body = []; while (i < lines.length && /^>/.test(lines[i])) body.push(lines[i++].replace(/^>\s?/, '')); prose(element(parent, 'blockquote'), body.join('\n'), depth + 1); continue; }
      const list = /^\s*([-*+]|\d+\.)\s+(.*)$/.exec(line);
      if (list) { const ordered = /\d/.test(list[1]), node = element(parent, ordered ? 'ol' : 'ul'); if (ordered) node.start = parseInt(list[1], 10); while (i < lines.length) { const item = /^\s*([-*+]|\d+\.)\s+(.*)$/.exec(lines[i]); if (!item || /\d/.test(item[1]) !== ordered) break; inline(element(node, 'li'), item[2]); i++; } continue; }
      if (i + 1 < lines.length && line.includes('|') && tableRule(lines[i + 1])) {
        const table = element(parent, 'table'), head = element(element(table, 'thead'), 'tr'); cells(line).forEach(cell => inline(element(head, 'th'), cell)); const body = element(table, 'tbody'); i += 2;
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { const row = element(body, 'tr'); cells(lines[i++]).forEach(cell => inline(element(row, 'td'), cell)); } continue;
      }
      const body = [line]; i++; while (i < lines.length && lines[i].trim() && !special(lines[i]) && !(i + 1 < lines.length && lines[i].includes('|') && lines[i + 1].includes('---'))) body.push(lines[i++]); inline(element(parent, 'p'), body.join('\n'));
    }
  }
  function content(parent, value, markdown = true) {
    if (!Array.isArray(value)) { if (markdown) prose(parent, value); else text(element(parent, 'div', 'prose'), value); return; }
    for (const part of value) {
      if (part.type === 'text') content(parent, part.text, markdown);
      else if (part.type === 'image_url') {
        const url = part.image_url?.url;
        if (typeof url === 'string' && /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(url)) { const image = element(parent, 'img', 'attachment'); image.src = url; image.alt = 'Conversation image'; image.loading = 'lazy'; }
        else text(element(parent, 'div', 'error'), 'Unsupported image source');
      }
    }
  }
  function sources(parent, results) {
    if (!Array.isArray(results)) return;
    const list = element(parent, 'ol', 'sources');
    for (const result of results.slice(0, 5)) { if (!result || typeof result !== 'object') continue; const row = element(list, 'li'), url = sourceURL(result.url), label = result.title || result.url || 'Source'; if (url) { const link = element(row, 'a'); link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; text(link, label); } else text(row, label); }
  }
  function toolResult(parent, message) {
    try { const result = JSON.parse(message.content); if (result.error) text(element(parent, 'div', 'error'), '⚠ Tool failed: ' + (typeof result.error === 'string' ? result.error : JSON.stringify(result.error))); sources(parent, result.results || (result.url ? [result] : [])); if (!result.error) text(element(parent, 'div', 'small'), '✓ Result received'); } catch (_) { text(element(parent, 'div', 'small'), 'Result received'); }
  }
  function message(parent, message, results = [], live = false) {
    const role = ['user', 'assistant', 'tool', 'system', 'error'].includes(message.role) ? message.role : 'system';
    const node = element(parent, 'article', `msg ${role}`); text(element(node, 'div', 'role'), role === 'user' ? 'You' : role === 'assistant' ? 'DeepSeek' : role === 'tool' ? 'Tool activity' : role);
    if (message.reasoning_content || live) { const details = element(node, 'details', 'reasoning'); text(element(details, 'summary'), 'Reasoning'); if (live) { details.hidden = !message.reasoning_content; text(element(details, 'div', 'prose liveReasoning'), message.reasoning_content); } else prose(details, message.reasoning_content); }
    if (role === 'tool') { const details = element(node, 'details'); text(element(details, 'summary'), 'Tool result'); toolResult(details, message); }
    else {
      if (message.tool_calls?.length) { const details = element(node, 'details', 'toolcall'); text(element(details, 'summary'), results.length ? `Used ${results.length} tools${results.length < message.tool_calls.length ? ' · ' + (message.tool_calls.length - results.length) + ' requested' : ''}` : `Requested ${message.tool_calls.length} tools`);
        for (const call of message.tool_calls) { const row = element(details, 'div', 'toolrow'), result = results.find(m => m.tool_call_id === call.id); text(row, `${result ? '✓' : '●'} ${call.function?.name === 'web_search' ? 'Search web' : call.function?.name === 'fetch_url' ? 'Fetch URL' : call.function?.name || 'Tool'} `); try { const args = JSON.parse(call.function?.arguments || '{}'); const url = sourceURL(args.url); if (url) { const a = element(row, 'a'); a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; text(a, url); } else text(row, args.query || ''); } catch (_) { text(row, call.function?.arguments || ''); } if (result) toolResult(row, result); }
      }
      const body = element(node, 'div', 'responseBody');
      if (live) { element(body, 'div', 'liveBlocks'); text(element(body, 'div', 'prose liveContent'), message.content); node._stream = {consumed: 0, scanned: 0, searched: 0, fence: false, shown: String(message.content || '').length}; }
      else content(body, message.content, role === 'assistant');
      if (role === 'assistant' && !live && message.content) copyButton(node, String(message.content), 'Copy response');
    }
    return node;
  }
  function updateLive(node, message) {
    const stream = node._stream, value = message.content;
    let end, boundary = stream.consumed;
    // Only scan newly completed lines. Finalize paragraphs/fences once, keeping the unfinished tail literal.
    while ((end = value.indexOf('\n', stream.searched)) >= 0) {
      const line = value.slice(stream.scanned, end);
      if (stream.fence ? /^\s*```\s*$/.test(line) : /^\s*```[A-Za-z0-9_+.-]*\s*$/.test(line)) stream.fence = !stream.fence;
      stream.scanned = stream.searched = end + 1;
      if (!stream.fence && !line.trim()) boundary = end + 1;
    }
    stream.searched = value.length;
    const tail = node.querySelector('.liveContent').firstChild;
    if (boundary > stream.consumed) { prose(node.querySelector('.liveBlocks'), value.slice(stream.consumed, boundary)); stream.consumed = boundary; tail.data = value.slice(boundary); }
    else tail.appendData(value.slice(stream.shown));
    stream.shown = value.length;
    node.querySelector('.reasoning').hidden = !message.reasoning_content;
    const reasoning = node.querySelector('.liveReasoning').firstChild;
    reasoning.appendData(message.reasoning_content.slice(reasoning.length));
  }
  function activity(parent, result) { const node = element(parent, 'article', 'msg tool'); text(element(node, 'div', 'role'), 'Tool activity'); if (result.error) text(element(node, 'div', 'error'), result.error); sources(node, result.results); return node; }
  window.ChatRender = {message, activity, sourceURL, prose, updateLive};
})();
