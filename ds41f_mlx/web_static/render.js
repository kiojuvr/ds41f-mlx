/* Untrusted conversation and provider data never enter an HTML parser. */
(() => {
  'use strict';
  const text = (parent, value) => parent.appendChild(document.createTextNode(String(value ?? '')));
  const element = (parent, tag, cls) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    parent.appendChild(node);
    return node;
  };
  function sourceURL(value) {
    try {
      const url = new URL(String(value));
      return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null;
    } catch (_) { return null; }
  }
  function prose(parent, value) {
    // Linear fenced-code rendering; no HTML parser or backtracking over long
    // unbroken model output. Markdown constructs other than fences stay text.
    let code = false, lines = [];
    const flush = () => {
      if (!lines.length) return;
      const node = code ? element(element(parent, 'pre'), 'code') : element(parent, 'div', 'prose');
      text(node, lines.join('\n')); lines = [];
    };
    for (const line of String(value ?? '').split('\n')) {
      if ((code && line.trim() === '```') || (!code && line.length < 128 && /^```[A-Za-z0-9_+.-]*\s*$/.test(line))) {
        flush(); code = !code;
      } else lines.push(line);
    }
    flush();
  }
  function content(parent, value) {
    if (!Array.isArray(value)) { prose(parent, value); return; }
    for (const part of value) {
      if (part.type === 'text') prose(parent, part.text);
      else if (part.type === 'image_url') {
        const url = part.image_url?.url;
        // Do not make arbitrary network requests for transcript images.
        if (typeof url === 'string' && /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(url)) {
          const image = element(parent, 'img', 'attachment');
          image.src = url;
          image.alt = 'Conversation image';
          image.loading = 'lazy';
        } else text(element(parent, 'div', 'error'), 'Unsupported image source');
      }
    }
  }
  function message(parent, message) {
    const role = ['user', 'assistant', 'tool', 'system', 'error'].includes(message.role) ? message.role : 'system';
    const node = element(parent, 'article', `msg ${role}`);
    text(element(node, 'div', 'role'), role);
    if (message.reasoning_content) {
      const details = element(node, 'details', 'reasoning');
      text(element(details, 'summary'), 'Reasoning');
      prose(details, message.reasoning_content);
    }
    if (role === 'tool') {
      const details = element(node, 'details');
      text(element(details, 'summary'), `Tool result · ${message.tool_call_id || ''}`);
      content(details, message.content);
      // Sources remain usable after transcript rendering/reload.
      try {
        const result = JSON.parse(message.content);
        sources(node, result.results || (result.url ? [result] : []));
        if (result.error) text(element(node, 'div', 'error'), result.error);
      } catch (_) { /* Non-JSON tool results are ordinary text. */ }
    } else content(node, message.content);
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        const details = element(node, 'details', 'toolcall');
        text(element(details, 'summary'), `Requested ${call.function?.name || 'tool'} · ${call.id || ''}`);
        prose(details, call.function?.arguments || '');
      }
    }
    return node;
  }
  function sources(parent, results) {
    if (!Array.isArray(results)) return;
    const list = element(parent, 'ol', 'sources');
    for (const result of results.slice(0, 5)) {
      if (!result || typeof result !== 'object') continue;
      const row = element(list, 'li');
      const url = sourceURL(result.url);
      const label = result.title || result.url || 'Source';
      if (url) {
        const link = element(row, 'a');
        link.href = url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        text(link, label);
      } else text(row, label);
      if (result.snippet) text(element(row, 'div'), result.snippet);
    }
  }
  function activity(parent, result) {
    const node = element(parent, 'article', 'msg tool');
    text(element(node, 'div', 'role'), `Tool activity · ${result.tool || ''} · ${result.id || ''}`);
    if (result.error) text(element(node, 'div', 'error'), result.error);
    if (result.results) sources(node, result.results);
    else prose(node, JSON.stringify(result, null, 2));
    return node;
  }
  window.ChatRender = {message, activity, sourceURL};
})();
