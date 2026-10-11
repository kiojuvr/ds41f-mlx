"""Wide-character terminal editor and restrained text Chat presentation."""
from __future__ import annotations

import json
import unicodedata

from .operator_dashboard import Line


def cells(char):
    if unicodedata.combining(char) or unicodedata.category(char) in ('Mn', 'Me', 'Cf'):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1


def wrap(text, width):
    """Cell-aware wrapping; terminal control characters are never emitted."""
    lines, row, used = [], '', 0
    for char in str(text):
        if char == '\n':
            lines.append(row)
            row, used = '', 0
            continue
        if char == '\t':
            char = ' '
        if unicodedata.category(char) in ('Cc', 'Cs'):
            char = '�'
        size = cells(char)
        if used + size > width and row:
            lines.append(row)
            row, used = '', 0
        row += char
        used += size
    return lines + [row]


class Composer:
    def __init__(self):
        self.text = ''
        self.cursor = 0
        self.escape = ''
        self.pasting = False
        self.paste_cr = False

    def insert(self, text):
        self.text = self.text[:self.cursor] + text + self.text[self.cursor:]
        self.cursor += len(text)

    def decode(self, key):
        """Consume bracketed-paste delimiters across polling ticks.

        Paste bytes are text, never actions, including embedded control keys.
        Plain Enter ALWAYS inserts newline, so unbracketed multiline paste also
        cannot send. Send uses Ctrl-G, independent of IME Enter/Shift-Enter.
        """
        if not isinstance(key, str):
            return None if self.pasting else key
        if self.escape or key == '\x1b':
            self.escape += key
            if self.escape == '\x1b[200~':
                self.pasting = True
                self.paste_cr = False
                self.escape = ''
            elif self.escape == '\x1b[201~':
                self.pasting = False
                self.escape = ''
            elif not any(s.startswith(self.escape) for s in ('\x1b[200~', '\x1b[201~')):
                if self.pasting:
                    self.insert(self.escape.replace('\x1b', ''))
                self.escape = ''
            return None
        if self.pasting:
            if key in ('\r', '\n'):
                if key != '\n' or not self.paste_cr:
                    self.insert('\n')
            elif key == '\t' or key.isprintable():
                self.insert(key)
            self.paste_cr = key == '\r'
            return None
        return key

    def edit(self, key, curses, width=80):
        if key in ('\r', '\n', curses.KEY_ENTER):
            self.insert('\n')
        elif key in ('\x7f', '\b', curses.KEY_BACKSPACE):
            if self.cursor:
                self.text = self.text[:self.cursor-1] + self.text[self.cursor:]
                self.cursor -= 1
        elif key == curses.KEY_DC:
            self.text = self.text[:self.cursor] + self.text[self.cursor+1:]
        elif key == curses.KEY_LEFT:
            self.cursor = max(0, self.cursor-1)
        elif key == curses.KEY_RIGHT:
            self.cursor = min(len(self.text), self.cursor+1)
        elif key in (curses.KEY_UP, curses.KEY_DOWN):
            positions = []
            row = col = 0
            for char in self.text + '\0':
                if char not in ('\n', '\0') and col+cells(char) > width:
                    row, col = row+1, 0
                positions.append((row, col))
                if char == '\n':
                    row, col = row+1, 0
                else:
                    col += cells(char)
            row, col = positions[self.cursor]
            target = row + (-1 if key == curses.KEY_UP else 1)
            candidates = [(abs(c-col), i) for i, (r, c) in enumerate(positions) if r == target]
            if candidates:
                self.cursor = min(candidates)[1]
        elif key in ('\x01', curses.KEY_HOME):
            self.cursor = self.text.rfind('\n', 0, self.cursor) + 1
        elif key == curses.KEY_END:
            end = self.text.find('\n', self.cursor)
            self.cursor = len(self.text) if end < 0 else end
        elif key == '\x15':
            start = self.text.rfind('\n', 0, self.cursor) + 1
            self.text = self.text[:start] + self.text[self.cursor:]
            self.cursor = start
        elif isinstance(key, str) and (key.isprintable() or key == '\t'):
            self.insert(key)


class ChatView:
    def __init__(self):
        self.editor = Composer()
        self.top = None  # None follows latest; integer is a physical-line anchor
        self.body_height = 1
        self.total = 0
        self.cache_key = None
        self.history = []
        self.settings_open = False
        self.setting_index = 0
        self.setting_editor = Composer()

    @property
    def active_editor(self):
        return self.setting_editor if self.settings_open else self.editor

    def load_setting(self, app):
        from .chat_capability import PARAMETERS
        name = ('reasoning_effort',) + PARAMETERS
        value = app.settings.values[name[self.setting_index]]
        self.setting_editor.text = json.dumps(value) if not isinstance(value, str) else value
        self.setting_editor.cursor = len(self.setting_editor.text)

    def setting_key(self, key, app, curses, busy):
        from .chat_capability import PARAMETERS
        names = ('reasoning_effort',) + PARAMETERS
        if key == '\x0f':
            if app.settings is None:
                app.notice = 'Ctrl-N: negotiate runtime before settings'
                return True
            self.settings_open = not self.settings_open
            if self.settings_open:
                self.load_setting(app)
            return True
        if not self.settings_open:
            return False
        if key in ('\t', curses.KEY_BTAB):
            self.setting_index = (self.setting_index + (-1 if key == curses.KEY_BTAB else 1)) % len(names)
            self.load_setting(app)
        elif key == '\x07':
            try:
                if busy or app.pending or app.calls() or app.awaiting_response:
                    raise ValueError('Settle/continue the current workflow before changing settings')
                name = names[self.setting_index]
                text = self.setting_editor.text.strip()
                if name == 'reasoning_effort':
                    value = text
                elif name == 'seed' and text == 'null':
                    value = None
                elif app.settings.contract['parameters'][name].get('type') == 'integer':
                    value = int(text)
                else:
                    value = float(text)
                app.settings.change(name, value)
                app.notice = 'Setting applied to subsequent user turns'
            except (ValueError, TypeError) as exc:
                app.notice = str(exc)
        elif key not in ('\x0e', '\x17', '\x12', '\x05', '\x14', '\x11', '\x03'):
            self.setting_editor.edit(key, curses)
        else:
            return False
        return True

    def render_settings(self, app, width, height):
        from .chat_capability import PARAMETERS
        lines = [Line('Generation · runtime generation-policy-v1', bold=True)]
        names = ('reasoning_effort',) + PARAMETERS
        for i, name in enumerate(names):
            spec = (app.settings.contract['reasoning'] if name == 'reasoning_effort'
                    else app.settings.contract['parameters'][name])
            lock = ' · SESSION LOCKED; Ctrl-N new chat' if name == 'reasoning_effort' and app.settings.reasoning is not None else ''
            for row in wrap(f'{">" if i == self.setting_index else " "} {name}={app.settings.values[name]} · default={app.settings.defaults[name]} · {json.dumps(spec)}{lock}', max(2, width-2)):
                lines.append(Line(row, tone='cyan' if i == self.setting_index else 'dim'))
        # Keep editor/actions visible even on short terminals.
        footer = [Line(app.notice, tone='yellow'), Line(names[self.setting_index] + ': ' + self.setting_editor.text),
                  Line('Tab/Shift-Tab Select · ^G Apply · ^O Chat · Enter Newline · seed: null', tone='dim')]
        return (lines[:max(0, height-3)] + footer)[:height], None

    def scroll(self, amount):
        current = max(0, self.total-self.body_height) if self.top is None else self.top
        self.top = max(0, min(max(0, self.total-self.body_height), current+amount))

    def render(self, app, width, height, runtime, server, busy):
        if self.settings_open and app.settings:
            return self.render_settings(app, width, height)
        width = max(2, width-2)
        def message_lines(messages):
            result = []
            for message in messages:
                result.append(Line(message['role'], bold=True))
                if message.get('reasoning_content'):
                    result += [Line(s, tone='dim') for s in wrap('reasoning: ' + message['reasoning_content'], width)]
                content = message.get('content') or ''
                result += [Line(s) for s in wrap(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False), width)]
                for call in message.get('tool_calls') or []:
                    result += [Line(s, tone='dim') for s in wrap('tool ' + call['function']['name'] + ': ' + call['function']['arguments'], width)]
                result.append(Line(''))
            return result
        key = (app.id, id(app.messages), len(app.messages), width)
        if key != self.cache_key:
            self.history = message_lines(app.messages)
            self.cache_key = key
        body = list(self.history)
        if app.live:
            body += message_lines([dict(app.live, role='assistant [UNSETTLED — not canonical]')])
        if not body:
            body = [Line('Text conversation · in-memory only', tone='dim')]
        draft = wrap(self.editor.text, width)
        before = wrap(self.editor.text[:self.editor.cursor], width)
        cursor_row, cursor_col = len(before)-1, sum(cells(c) for c in before[-1])
        # At an exact wrap boundary the cursor starts the next physical line.
        if cursor_col >= width:
            cursor_row += 1
            cursor_col = 0
            if cursor_row >= len(draft):
                draft.append('')
        help_lines = [Line(s, tone='dim') for text in (
            '^G Send  Enter Newline  ^C Stop  ^T Dashboard  ^Q Quit',
            '^O Generation ^N New ^W Close ^R Reconcile ^E Continue PgUp/Dn Scroll ^L Latest F2 Start/F3 Stop Server')
            for s in wrap(text, width)]
        input_height = max(1, min(max(1, height//4), max(2, len(draft)), height-5-len(help_lines)))
        first = max(0, cursor_row-input_height+1)
        input_lines = draft[first:first+input_height]
        self.body_height = max(0, height-len(input_lines)-4-len(help_lines))
        self.total = len(body)
        if self.top is not None:
            self.top = min(self.top, max(0, len(body)-self.body_height))
        top = max(0, len(body)-self.body_height) if self.top is None else self.top
        defaults = app.settings.values if app.settings else {}
        capability = ((app.runtime or {}).get('capabilities') or {}).get('stateful_chat', 'standard-off')
        lines = [Line(f'Chat · runtime {runtime} · server {server} · {"working" if busy else "idle"} · {"latest" if self.top is None else "scroll paused"}', bold=True),
                 Line(f'{app.id or "no session"} · {(app.record or {}).get("state", "—")} · turns {app.count} · {capability} · T={defaults.get("temperature", "—")} p={defaults.get("top_p", "—")} k={defaults.get("top_k", "—")} min_p={defaults.get("min_p", "—")} reasoning={defaults.get("reasoning_effort", "—")} output={defaults.get("max_tokens", "—")}', tone='dim'),
                 Line(app.problem or app.notice, tone='yellow' if app.problem else 'dim')]
        lines += body[top:top+self.body_height]
        lines += [Line('')] * max(0, self.body_height-len(body[top:top+self.body_height]))
        lines.append(Line('Composer' + (' · pasting' if self.editor.pasting else ''), tone='dim'))
        input_row = len(lines)
        lines += [Line(s) for s in input_lines]
        lines += help_lines
        return lines[:height], (input_row+cursor_row-first, 1+cursor_col)
