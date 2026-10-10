"""Stdlib TUI: read-only projection consumer and canonical launcher client."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from time import monotonic
from urllib.request import Request, urlopen
from urllib.error import URLError

from .operator_control import RUNTIME_PORT, CHAT_PORT, tui_guard


def call(port, path='/ds41f/status', method='GET'):
    req = Request(f'http://127.0.0.1:{port}{path}', method=method,
                  data=b'' if method == 'POST' else None)
    with urlopen(req, timeout=1) as response:
        return json.load(response)


def launch(chat=False, endpoint=None):
    args = ['ds41f_mlx.web'] if chat else ['ds41f_mlx.ops', 'start', '--profile', 'mtp-serving-v1']
    if endpoint:
        args += ['--host', endpoint['host'], '--port', str(endpoint['port'])]
    return subprocess.Popen([sys.executable, '-m', *args], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def number(value, unit=''):
    if value is None:
        return '—'
    return (f'{value:,}' if isinstance(value, int) else f'{value:,.1f}') + unit


def render(snapshot, now=None):
    now = monotonic() if now is None else now
    if snapshot is None:
        return 'SERVER STOPPED\nNo ds41f runtime control listener.', '', '', ''
    r, last = snapshot['current_request'], snapshot['last_request']
    state, queue, active = snapshot['state'], snapshot['queued_requests'], snapshot['active_requests']
    if r:
        phase = r['current_phase']
        now_text = (f"{phase.replace('_', ' ')}   {now-r['phase_started_mono']:.2f} s\n"
                    f"request {now-r['request_started_mono']:.2f} s   active {active}   queued {queue}")
        waiting = [item for item in snapshot['requests'] if item['current_phase'] == 'QUEUED']
        if waiting:
            age = max(now-item['queue_entered_mono'] for item in waiting)
            reasons = ', '.join(sorted({item.get('queue_reason', 'scheduler_worker') for item in waiting}))
            now_text += f'\nQUEUED  oldest {age:.2f} s — {reasons}; waiting behind active request'
    elif state == 'READY':
        now_text = 'SERVER IDLE — no request waiting inside ds41f'
        if last:
            now_text += f"\nlast request {now-last['finished_mono']:.1f} s ago"
            if last['finish_reason'] == 'tool_calls':
                now_text += '\nWAITING FOR CLIENT — last finish: tool_calls; no active request'
    else:
        now_text = state + ' — runtime lifecycle'
    m = (r or last or {}).get('metrics', {})
    label = 'current' if r else 'last'
    prompt, cached = m.get('prompt_tokens'), m.get('cached_tokens')
    new = None if prompt is None or cached is None else prompt-cached
    hit = None if not prompt or cached is None else 100*cached/prompt
    decode = m.get('generated', 0)/m['decode_s'] if m.get('decode_s', 0) else None
    # Only genuine new prefill work; excludes terminal P5 holdout.
    prefill_work = max(0, new-1) if new is not None else None
    prefill = prefill_work/m['suffix_append_s'] if m.get('suffix_append_s', 0) and prefill_work else None
    ttft = m.get('total_ttft_s')
    stats = (last or {}).get('metrics', {})
    drafted = stats.get('proposed_drafts', 0)
    accept = 100*stats.get('accepted_drafts', 0)/drafted if drafted else None
    performance = (f'DECODE {number(decode, " tok/s")}   PREFILL {number(prefill, " tok/s")} ({label})\n'
                   f'TTFT {number(None if ttft is None else ttft*1000, " ms")}   MTP ACCEPT {number(accept, "%")} (last settled)')
    capacity = (snapshot.get('endpoint') or {}).get('context_tokens')
    context = (f'CONTEXT / PROMPT {number(prompt)} / {number(capacity)}   CACHE HIT {number(hit, "%")} ({label})\n'
               f'CACHED {number(cached)} / {number(prompt)} tokens   NEW {number(new)} tokens\n'
               f'NEW PREFILL {number(prefill_work)} tokens (excludes terminal P5 token)')
    durations = (r or last or {}).get('completed_phase_durations', {})
    timings = '  |  '.join(f'{key.removesuffix("_ms").upper()} {value:.0f}ms' for key, value in durations.items())
    detailed = '  |  '.join(f'{key.removesuffix("_s")} {m[key]*1000:.0f}ms' for key in
        ('recipe_convert_s', 'recipe_render_s', 'recipe_encode_s', 'cache_lookup_s', 'paired_restore_s',
         'prompt_checkpoint_capture_s', 'suffix_append_s', 'p5_handoff_s', 'first_decode_call_s') if key in m)
    aligned = stats.get('aligned')
    diagnostic = (f"REPLAY {m.get('prompt_replay', '—')}   REPACK {m.get('full_cache_repack', '—')}   "
                  f"FRONTIER {stats.get('canonical_frontier', '—')}   "
                  f"{'ALIGNED' if aligned else 'NOT ALIGNED' if aligned is False else '—'} (last settled)\n"
                  f'{timings}\n{detailed}\nCACHE {json.dumps(snapshot.get("cache"), ensure_ascii=False)}')
    return now_text, performance, context, diagnostic


class Operator:
    def __init__(self):
        self.snapshot = self.chat_snapshot = None
        self.operation = False
        self.children = {}
        self.message = ''
        self.offset = 0
        self.conflicts = set()

    async def poll(self):
        async def get(port):
            try:
                return await asyncio.to_thread(call, port)
            except (OSError, URLError, ValueError):
                return None
        self.snapshot, self.chat_snapshot = await asyncio.gather(get(RUNTIME_PORT), get(CHAT_PORT))
        def occupied(port):
            import socket
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.1):
                    return True
            except OSError:
                return False
        for control_port, snapshot, service_port in ((RUNTIME_PORT, self.snapshot, 8000),
                                                     (CHAT_PORT, self.chat_snapshot, 8080)):
            if snapshot is None and await asyncio.to_thread(occupied, service_port):
                self.conflicts.add(control_port)
            else:
                self.conflicts.discard(control_port)
        for port, child in list(self.children.items()):
            if child.poll() is not None:
                if child.returncode:
                    self.message = f'FAILED/CONFLICT: launcher exited {child.returncode}; use headless CLI for startup logs'
                del self.children[port]

    async def lifecycle(self, action, chat=False):
        if self.operation:
            return
        self.operation = True
        port, name = (CHAT_PORT, 'Chat') if chat else (RUNTIME_PORT, 'LLM')
        try:
            try:
                status = await asyncio.to_thread(call, port)
            except (OSError, URLError):
                status = None
            endpoint = (status or {}).get('endpoint') or {}
            if action in ('stop', 'restart') and status:
                await asyncio.to_thread(call, port, '/ds41f/control/shutdown', 'POST')
                self.message = name+' graceful shutdown requested'
                for _ in range(1200):
                    await asyncio.sleep(.1)
                    try:
                        await asyncio.to_thread(call, port)
                    except (OSError, URLError):
                        break
                else:
                    raise RuntimeError('shutdown still in progress; Start not attempted')
                status = None
            if action in ('start', 'restart'):
                if status:
                    self.message = name+' already running'
                else:
                    # Protect against legacy/unrelated process on the service port.
                    import socket
                    service_port = endpoint.get('port', 8080 if chat else 8000)
                    try:
                        with socket.create_connection(('127.0.0.1', service_port), timeout=.2):
                            raise RuntimeError('CONFLICT: service port occupied without ds41f control; no model load attempted')
                    except ConnectionRefusedError:
                        pass
                    child = launch(chat, endpoint if action == 'restart' else None)
                    self.children[port] = child
                    self.message = name+' canonical launcher started'
            await self.poll()
        except Exception as exc:
            self.message = name+f' control failed: {exc}'
        finally:
            self.operation = False

    async def run(self, screen, curses):
        screen.nodelay(True)
        screen.keypad(True)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        poll_task = None
        action_task = None
        last_poll = -1.
        try:
            while True:
                now = monotonic()
                if now-last_poll >= .4 and (poll_task is None or poll_task.done()):
                    if poll_task is not None:
                        poll_task.result()
                    poll_task = asyncio.create_task(self.poll())
                    last_poll = now
                height, width = screen.getmaxyx()
                state = (self.snapshot or {}).get('state', 'CONFLICT' if RUNTIME_PORT in self.conflicts else 'STOPPED')
                if state == 'READY' and self.snapshot.get('current_request'):
                    state = 'BUSY'
                lines = [f'ds41f — DeepSeek-V4.1-Flash   {state}', '']
                panels = render(self.snapshot)
                if self.snapshot is None and RUNTIME_PORT in self.conflicts:
                    panels = ('CONFLICT — service :8000 occupied without ds41f control; no startup attempted', *panels[1:])
                for title, value in zip(('NOW', 'PERFORMANCE', 'CONTEXT / CACHE', 'RUNTIME DIAGNOSTICS'), panels):
                    lines.extend([f'── {title} ──', *value.splitlines(), ''])
                chat = self.chat_snapshot['state'] if self.chat_snapshot else 'CONFLICT' if CHAT_PORT in self.conflicts else 'STOPPED'
                lines.extend([f'CHAT {chat} — ds41f_mlx.web :8080', self.message])
                wrapped = [piece for line in lines for piece in (textwrap.wrap(line, max(1, width-2)) or [''])]
                self.offset = max(0, min(self.offset, len(wrapped)-max(1, height-2)))
                screen.erase()
                for row, line in enumerate(wrapped[self.offset:self.offset+max(0, height-2)]):
                    try:
                        screen.addnstr(row, 1, line, max(0, width-2), curses.A_BOLD if 'NOW' in line or row+self.offset in (0, 3) else 0)
                    except curses.error:
                        pass
                try:
                    screen.addnstr(max(0, height-1), 0,
                        '[S] Start [X] Stop [R] Restart [C] Chat Start [V] Chat Stop [Q] Quit ↑↓ scroll', max(0, width-1))
                except curses.error:
                    pass
                screen.refresh()
                key = screen.getch()
                if key in (ord('q'), ord('Q')):
                    break
                if key == curses.KEY_UP:
                    self.offset -= 1
                elif key == curses.KEY_DOWN:
                    self.offset += 1
                commands = {'s': ('start', False), 'x': ('stop', False), 'r': ('restart', False),
                            'c': ('start', True), 'v': ('stop', True)}
                command = commands.get(chr(key).lower()) if 0 <= key < 256 else None
                if command and (action_task is None or action_task.done()):
                    action_task = asyncio.create_task(self.lifecycle(*command))
                await asyncio.sleep(.1)
        finally:
            # Cancel UI work only: never signal the detached runtime on quit.
            for task in (poll_task, action_task):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(t for t in (poll_task, action_task) if t is not None), return_exceptions=True)


def main():
    try:
        import curses
    except ImportError:
        print('TUI requires Python curses support and a terminal; headless subcommands remain available.', file=sys.stderr)
        return 2
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print('ds41f TUI requires a terminal; use ds41f start/inspect/accept for headless operation.', file=sys.stderr)
        return 2
    try:
        guard = tui_guard()
    except OSError:
        print('CONFLICT: another ds41f TUI is running', file=sys.stderr)
        return 2
    try:
        curses.wrapper(lambda screen: asyncio.run(Operator().run(screen, curses)))
    finally:
        guard.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
