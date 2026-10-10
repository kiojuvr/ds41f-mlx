"""Content-free layouts for the operator terminal (no runtime ownership)."""
from dataclasses import dataclass
import re
from time import monotonic

from .operator_control import RUNTIME_PORT


# Display vocabulary only: observed phases remain the execution authority.
PHASE_LABELS = [('RECEIVED', 'RECEIVED'), ('QUEUED', 'QUEUED'),
                ('ENCODING', 'ENCODE'), ('CACHE_LOOKUP', 'LOOKUP'),
                ('RESTORING', 'RESTORE'), ('SUFFIX_APPEND', 'APPEND'),
                ('CHECKPOINT_CAPTURE', 'CHECKPOINT'), ('P5_HANDOFF', 'P5'),
                ('DECODING', 'DECODE'), ('SETTLING', 'SETTLE'),
                ('DONE', 'DONE'), ('ERROR', 'ERROR')]


@dataclass(frozen=True)
class Line:
    text: str = ''
    tone: str = ''
    bold: bool = False
    values: tuple[tuple[int, int], ...] = ()
    accents: tuple[tuple[int, int, str], ...] = ()


@dataclass(frozen=True)
class Viewport:
    width: int
    height: int

    @property
    def layout(self):
        return 'wide' if self.width >= 100 else 'medium' if self.width >= 70 else 'compact'


def number(value, unit=''):
    if value is None:
        return '—'
    return (f'{value:,}' if isinstance(value, int) else f'{value:,.1f}') + unit


def elapsed(now, start):
    return '—' if start is None else f'{max(0, now-start):.2f} s'


def state_tone(state):
    if state in ('FAILED', 'CONFLICT'):
        return 'red'
    if state in ('LOADING', 'STARTING', 'STOPPING', 'QUEUED'):
        return 'yellow'
    return 'cyan' if state == 'BUSY' else 'green' if state == 'READY' else 'dim'


class Dashboard:
    """A projection of known scalar fields, never arbitrary request objects."""
    def __init__(self, snapshot, now=None, conflict=False):
        self.s = snapshot or {}
        self.now = monotonic() if now is None else now
        self.r = self.s.get('current_request') or {}
        self.last = self.s.get('last_request') or {}
        self.m = (self.r or self.last).get('metrics', {})
        self.stats = self.last.get('metrics', {})
        self.state = self.s.get('state', 'CONFLICT' if conflict else 'STOPPED')
        if self.state == 'READY' and self.r:
            self.state = 'BUSY'
        prompt, cached = self.m.get('prompt_tokens'), self.m.get('cached_tokens')
        self.new = None if prompt is None or cached is None else prompt-cached
        self.work = None if self.new is None else max(0, self.new-1)
        decode_s, append_s = self.m.get('decode_s'), self.m.get('suffix_append_s')
        decode = self.m.get('generated')
        drafted, accepted = self.stats.get('proposed_drafts'), self.stats.get('accepted_drafts')
        ttft = self.m.get('total_ttft_s')
        self.performance = [
            ('Decode', number(decode/decode_s if decode is not None and decode_s else None, ' tok/s')),
            ('Prefill', number(self.work/append_s if self.work and append_s else None, ' tok/s')),
            ('TTFT', number(None if ttft is None else ttft*1000, ' ms')),
            ('MTP accept (last)', number(100*accepted/drafted if drafted and accepted is not None else None, '%'))]
        self.context = [
            ('Prompt / limit', f'{number(prompt)} / {number((self.s.get("endpoint") or {}).get("context_tokens"))}'),
            ('Cached', number(cached)), ('New', number(self.new)),
            ('Cache hit', number(100*cached/prompt if prompt and cached is not None else None, '%'))]

    def health(self, compact=False):
        aligned = self.stats.get('aligned')
        replay, repack = self.m.get('prompt_replay'), self.m.get('full_cache_repack')
        label = '✓ ALIGNED' if aligned is True else 'NOT ALIGNED' if aligned is False else 'alignment —'
        text = f'{label} · replay {number(replay)} · repack {number(repack)}'
        if not compact:
            text += f' · frontier {number(self.stats.get("canonical_frontier"))}'
        warnings = [label] if aligned is False else []
        warnings += [f'{name} {number(value)}' for name, value in (('replay', replay), ('repack', repack)) if value]
        fields = [('replay', replay), ('repack', repack)]
        if not compact:
            fields.append(('frontier', self.stats.get('canonical_frontier')))
        return Line(text, values=tuple((text.index(name+' ')+len(name)+1, len(number(value))) for name, value in fields),
                    accents=tuple((text.index(warning), len(warning), 'red') for warning in warnings))

    def now_lines(self, width):
        phase = self.r.get('current_phase')
        rows = [Line('NOW', 'dim')]
        if phase:
            label = phase.replace('_', ' ')
            duration = elapsed(self.now, self.r.get('phase_started_mono'))
            rows += [Line(pair(label, duration, width), bold=True,
                          accents=((0, len(label), 'red' if phase == 'ERROR' else state_tone('QUEUED' if phase == 'QUEUED' else 'BUSY')),)),
                     request_line(self.now, self.r, self.s, width)]
        elif self.state == 'READY':
            rows += [Line('IDLE', bold=True), Line('ready for request', 'dim')]
            if self.last:
                rows.append(Line(f'last request {elapsed(self.now, self.last.get("finished_mono"))} ago', 'dim'))
                if self.last.get('finish_reason') == 'tool_calls':
                    rows.append(Line('WAITING FOR CLIENT', 'dim'))
        else:
            label = 'LOADING MODEL' if self.state == 'LOADING' else self.state
            rows.append(Line(pair(label, '—' if self.state == 'LOADING' else '', width), bold=True,
                             accents=((0, len(label), state_tone(self.state)),)))
            if self.state == 'LOADING':
                rows.append(Line('waiting for runtime READY', 'dim'))
            elif self.state == 'CONFLICT':
                rows.append(Line('API :8000 occupied without ds41f control', 'red'))
        queued = self.s.get('queued_requests')
        if queued:
            starts = [r.get('queue_entered_mono') for r in self.s.get('requests', []) if r.get('current_phase') == 'QUEUED']
            oldest = min(starts) if starts and all(t is not None for t in starts) else None
            rows.append(Line(f'QUEUE  {queued} waiting · oldest {elapsed(self.now, oldest)}', bold=True,
                             accents=((0, 5, 'yellow'),)))
        return rows

    def rail_lines(self, width):
        # Token layout, not prose wrapping; never invent past or future phases.
        durations = self.r.get('completed_phase_durations', {})
        phase = self.r.get('current_phase')
        rows, text, accents = [], '', []
        for key, label in PHASE_LABELS:
            if phase != key and key.lower()+'_ms' not in durations:
                continue
            token = f'[ {label} ]' if phase == key else label
            separator = ' ─ ' if text else ''
            if text and len(text)+len(separator)+len(token) > width:
                rows.append(Line(text, 'dim', accents=tuple(accents)))
                text, accents, separator = '', [], ''
            start = len(text)+len(separator)
            text += separator+token
            if phase == key:
                tone = 'red' if key == 'ERROR' else 'yellow' if key == 'QUEUED' else 'cyan'
                accents.append((start, len(token), tone))
        if text:
            rows.append(Line(text, 'dim', accents=tuple(accents)))
        return rows

    def rail(self):
        return next(iter(self.rail_lines(1000)), Line())

    def timing_lines(self, width):
        durations = (self.r or self.last).get('completed_phase_durations', {})
        values = [(label.lower(), number(durations[key.lower()+'_ms'], ' ms'))
                  for key, label in PHASE_LABELS if key.lower()+'_ms' in durations]
        if not values:
            return []
        count = max(1, min(4, (width+4)//26))
        cell = (width-4*(count-1))//count
        rows = [Line('REQUEST TIMINGS · '+('current' if self.r else 'last'), 'dim')]
        for start in range(0, len(values), count):
            rows.append(columns([metric_rows('', [(label, value)], cell)[1]
                                 for label, value in values[start:start+count]], cell, 4))
        return rows

    def diagnostics(self):
        endpoint = self.s.get('endpoint') or {}
        rows = [Line('RUNTIME', 'dim')]
        values = [('state', self.state), ('endpoint', f'{endpoint.get("host", "—")}:{endpoint.get("port", "—")}'),
                  ('control', f'127.0.0.1:{RUNTIME_PORT}'), ('active', number(self.s.get('active_requests'))),
                  ('queued', number(self.s.get('queued_requests'))), ('lifecycle elapsed', '— (not published)')]
        rows += [Line(f'{k:<24} {v}') for k, v in values]
        rows += [Line(), Line('STATE INTEGRITY · last settled alignment/frontier', 'dim'), self.health()]
        # The projection exposes combined alignment, not individual offset arrays.
        rows += [Line(), Line('CURRENT REQUEST TIMINGS' if self.r else 'LAST REQUEST TIMINGS', 'dim')]
        timings = [('recipe convert', 'recipe_convert_s'), ('recipe render', 'recipe_render_s'),
                   ('recipe encode', 'recipe_encode_s'), ('request prepare', 'request_prepare_s'),
                   ('cache lookup', 'cache_lookup_s'), ('paired restore', 'paired_restore_s'),
                   ('checkpoint', 'prompt_checkpoint_capture_s'), ('suffix append', 'suffix_append_s'),
                   ('P5 handoff', 'p5_handoff_s'), ('first decode', 'first_decode_call_s'),
                   ('first native decode', 'first_native_decode_s'), ('TTFT', 'total_ttft_s')]
        rows += [Line(f'{label:<24} {number(None if self.m.get(key) is None else self.m[key]*1000, " ms")}') for label, key in timings]
        rows += [Line(f'new prefill tokens       {number(self.work)} (excludes P5)'), Line(), Line('OBSERVED PHASE DURATIONS', 'dim')]
        # Phase keys and cache keys originate in the content-free projection.
        rows += [Line(f'{k:<28} {number(v, " ms")}') for k, v in (self.r or self.last).get('completed_phase_durations', {}).items() if isinstance(v, (int, float))]
        rows += [Line(), Line('QUEUE', 'dim')]
        rows += [Line(f'{r.get("queue_reason", "—")} · age {elapsed(self.now, r.get("queue_entered_mono"))}') for r in self.s.get('requests', []) if r.get('current_phase') == 'QUEUED'] or [Line('—')]
        rows += [Line(), Line('CACHE · retained scalar summary', 'dim')]
        rows += [Line(f'{k:<28} {number(v)}') for k, v in (self.s.get('cache') or {}).items() if isinstance(v, (int, float, bool))] or [Line('—')]
        return rows


def pair(label, value, width):
    return label + ' '*max(1, width-len(label)-len(value)) + value


def request_line(now, request, snapshot, width):
    left = f'{"req" if width < 68 else "request"} {elapsed(now, request.get("request_started_mono"))}'
    right = (f'{"a" if width < 68 else "active "}{number(snapshot.get("active_requests"))} · '
             f'{"q" if width < 68 else "queued "}{number(snapshot.get("queued_requests"))}')
    text = pair(left, right, width)
    return Line(text, values=numeric_spans(text))


def columns(lines, cell, gap):
    text, values, accents = '', [], []
    for line in lines:
        offset = len(text)
        values.extend((start+offset, length) for start, length in line.values)
        accents.extend((start+offset, length, tone) for start, length, tone in line.accents)
        text += line.text.ljust(cell)+' '*gap
    return Line(text.rstrip(), lines[0].tone if lines else '', values=tuple(values), accents=tuple(accents))


def numeric_spans(text, offset=0):
    return tuple((offset+match.start(), len(match.group())) for match in re.finditer(r'\d[\d,.]*|—', text))


def metric_rows(title, values, width):
    rows = [Line(title, 'dim')]
    for label, value in values:
        text = pair(label, value, width)
        rows.append(Line(text, values=numeric_spans(value, len(text)-len(value))))
    return rows


def render_screen(snapshot, viewport, *, now=None, mode='dashboard', conflict=False,
                  chat=None, chat_conflict=False, message=''):
    """Return unwrapped rows; dashboard fits the viewport, details scroll separately."""
    d = Dashboard(snapshot, now, conflict)
    width = max(1, min(112, viewport.width-2))
    compact = viewport.layout == 'compact'
    title = 'ds41f' if compact else 'ds41f / Diagnostics' if mode == 'diagnostics' else 'ds41f ─ DeepSeek-V4.1-Flash'
    header_text = pair(title, d.state, width)
    header = [Line(header_text, bold=True,
                   accents=((len(header_text)-len(d.state), len(d.state), state_tone(d.state)),))]
    endpoint = d.s.get('endpoint') or {}
    chat_state = (chat or {}).get('state', 'CONFLICT' if chat_conflict else 'STOPPED')
    if not compact:
        header.append(Line(f'{endpoint.get("profile", "MTP serving v1")} · API :{endpoint.get("port", "—")} · Chat {chat_state} :{((chat or {}).get("endpoint") or {}).get("port", 8080)}', 'dim'))
    if mode == 'diagnostics':
        return header + [Line()] + d.diagnostics()
    if mode == 'help':
        return header + [Line(), Line('KEYS', 'dim')] + [Line(s) for s in (
            'S  Start runtime', 'X  Graceful Stop', 'R  Restart runtime', 'C  Chat Start',
            'V  Chat Stop', 'D  Toggle Diagnostics', '?  Toggle help', '↑↓ Scroll Diagnostics / help',
            'Q  Quit TUI; runtime remains running', '', 'Metrics: current request, otherwise last.',
            'MTP / alignment / frontier: last settled request.', '— means not published or not yet measured.')]
    now_rows = d.now_lines(width)
    metric_width = min(width, 56)  # keep stacked labels near their values
    if compact:
        perf = metric_rows('PERFORMANCE · '+('current' if d.r else 'last'),
                           list(zip(('DEC', 'PRE', 'TTFT', 'MTP · last'), (v for _, v in d.performance))), metric_width)
        ctx = metric_rows('CONTEXT', list(zip(('CTX / limit', 'Cached', 'New', 'Cache hit'),
                                            (v for _, v in d.context))), metric_width)
    elif viewport.layout == 'wide':
        cell = (width-6)//2
        left = metric_rows('PERFORMANCE · '+('current' if d.r else 'last'), d.performance, cell)
        right = metric_rows('CONTEXT', d.context, cell)
        perf = [columns([a, b], cell, 6) for a, b in zip(left, right)]
        ctx = []
    else:
        perf = metric_rows('PERFORMANCE · '+('current' if d.r else 'last'), d.performance, metric_width)
        ctx = metric_rows('CONTEXT', d.context, metric_width)
    rail = d.rail_lines(width) if d.r else []
    timing = d.timing_lines(width)
    health = [d.health(compact)]
    notice = [Line(message, 'red' if 'failed' in message.lower() or 'CONFLICT' in message else '')] if message else []
    budget = max(0, viewport.height-1)

    def compose():
        sections = [header+notice, now_rows, rail, perf, ctx, timing, health]
        allowance = max(0, budget-sum(len(section) for section in sections))
        gaps, rules = set(), set()
        # Spend available whitespace at reading boundaries, not at the bottom.
        for index in (1, 3, 4, 5, 6, 2):
            if allowance and sections[index]:
                gaps.add(index)
                allowance -= 1
        for index in (1, 6):
            if allowance:
                rules.add(index)
                allowance -= 1
        rows = []
        for index, section in enumerate(sections):
            if index in rules:
                rows.append(Line('─'*width, 'dim'))
            if index in gaps:
                rows.append(Line())
            rows.extend(section)
        return rows

    rows = compose()
    # On physically short terminals, details remain accessible with D.
    if len(rows) > budget:
        timing = []
        rows = compose()
    if len(rows) > budget:
        rail = []
        rows = compose()
    if len(rows) > budget:
        rows = [header[0]] + notice + now_rows[1:3] + [d.health(True)]
        if d.s.get('queued_requests'):
            rows.append(now_rows[-1])
        rows += metric_rows('', d.performance[:3], metric_width)[1:]
        for text in (f'CTX {d.context[0][1]} · hit {d.context[3][1]}',
                     f'cached {d.context[1][1]} · new {d.context[2][1]}',
                     f'MTP · last {d.performance[3][1]}'):
            rows.append(Line(text, values=numeric_spans(text)))
    return rows[:budget]


def command_strip(mode, width):
    if width < 45:
        return 'S Start X Stop R Restart Q Quit D/?'
    if width < 70:
        return 'S Start X Stop R Restart Q Quit C/V Chat D/?'
    return 'S Start  X Stop  R Restart  C Chat  V Stop Chat  D Details  ? Help  Q Quit' + ('  ↑↓ scroll' if mode != 'dashboard' else '')
