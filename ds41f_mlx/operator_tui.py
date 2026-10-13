"""Operator console: services, scalar telemetry and logs; never a chat frontend."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from time import monotonic
from urllib.request import Request, urlopen
from urllib.error import URLError

from .operator_control import RUNTIME_PORT, CHAT_PORT, tui_guard
from .operator_dashboard import Viewport, render_screen, command_strip


def call(port, path='/ds41f/status', method='GET'):
    req = Request(f'http://127.0.0.1:{port}{path}', method=method,
                  data=b'' if method == 'POST' else None)
    with urlopen(req, timeout=1) as response:
        return json.load(response)


def runtime_python():
    """Application tooling may control an already provisioned model environment."""
    from pathlib import Path
    configured = os.environ.get('DS41F_OPERATOR_RUNTIME_PYTHON')
    if configured:
        return configured
    if Path(sys.prefix).name == '.venv':
        repository = Path(sys.prefix).parent
        installed = repository.with_name(repository.name+'_mtp-venv')/'bin/python'
        if installed.is_file():
            return str(installed)
    return sys.executable


def launch(chat=False, endpoint=None, *, default_reasoning_effort=None):
    args = ['ds41f_mlx.web'] if chat else ['ds41f_mlx.ops', 'start', '--profile', 'mtp-serving-v1']
    if chat and endpoint is None:
        bind = os.environ.get('DS41F_CHAT_BIND', 'lan')
        if bind not in ('lan', 'loopback'):
            raise ValueError('DS41F_CHAT_BIND must be lan or loopback')
        endpoint = dict(host='0.0.0.0' if bind == 'lan' else '127.0.0.1',
                        port=int(os.environ.get('DS41F_WEB_PORT', '8080')), allow_private_lan=bind == 'lan')
    if endpoint:
        if chat and endpoint.get('scheme') == 'https':
            raise ValueError('TLS Web Chat restart requires manual certificate/key options; refusing HTTP downgrade')
        args += ['--host', endpoint['host'], '--port', str(endpoint['port'])]
        if chat:
            if endpoint.get('allow_private_lan', endpoint['host'] not in ('127.0.0.1', 'localhost', '::1')):
                args.append('--allow-private-lan')
            if endpoint.get('runtime_url'):
                args += ['--runtime-url', endpoint['runtime_url']]
    if not chat and default_reasoning_effort is not None:
        args += ['--default-reasoning-effort', default_reasoning_effort]
    environment = os.environ.copy()
    if not chat:
        for key in list(environment):
            if (key in ('DS41F_CHAT_BIND', 'DS41F_RUNTIME_URL', 'DS41F_OPERATOR_RUNTIME_PYTHON')
                    or key.startswith(('DS41F_WEB_', 'DS41F_DOCUMENT_', 'DS41F_EXA_', 'DS41F_PARALLEL_'))):
                environment.pop(key, None)
    interpreter = sys.executable if chat else runtime_python()
    return subprocess.Popen([interpreter, *([] if chat else ['-I']), '-m', *args],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True, env=environment)


def render(snapshot, now=None):
    return tuple(line.text for line in render_screen(snapshot, Viewport(100, 40), now=now))


def terminal_styles(curses):
    styles = {'': 0, 'dim': curses.A_DIM}
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            background = -1
        except curses.error:
            background = curses.COLOR_BLACK
        for pair, name in enumerate(('green', 'cyan', 'yellow', 'red'), 1):
            curses.init_pair(pair, getattr(curses, 'COLOR_'+name.upper()), background)
            styles[name] = curses.color_pair(pair)
    return styles


class Operator:
    def __init__(self):
        self.snapshot = self.chat_snapshot = None
        self.operation = False
        self.children = {}
        self.message = ''
        self.offset = 0
        self.mode = 'dashboard'
        from .operator_settings import load_reasoning
        self.default_reasoning_effort = load_reasoning()
        self.web_bind = os.environ.get('DS41F_CHAT_BIND', 'lan')
        self.web_port = int(os.environ.get('DS41F_WEB_PORT', '8080'))
        self.conflicts = set()
        from collections import deque
        from .runtime_logging import CAPACITY
        self.logs = deque(maxlen=CAPACITY)
        self.log_runtime = self.log_cursor = self.log_end = None
        self.log_gap = False
        self.log_lock = asyncio.Lock()

    async def poll_logs(self):
        async with self.log_lock:
            await self._poll_logs()

    async def _poll_logs(self):
        try:
            path = '/ds41f/logs' + (f'?after={self.log_cursor}' if self.log_cursor is not None else '')
            batch = await asyncio.to_thread(call, RUNTIME_PORT, path)
            if batch['runtime_id'] != self.log_runtime:
                self.logs.clear()
                self.log_runtime = batch['runtime_id']
                self.log_end = None
                self.log_gap = False
                if self.log_cursor is not None:
                    self.log_cursor = None
                    return await self._poll_logs()
            self.log_gap = self.log_gap or batch['gap']
            for record in batch['records']:
                if not self.logs or record['sequence'] > self.logs[-1]['sequence']:
                    self.logs.append(record)
            self.log_cursor = batch['next_sequence']
        except (OSError, URLError, ValueError, KeyError):
            pass

    async def poll(self):
        async def get(port):
            try:
                return await asyncio.to_thread(call, port)
            except (OSError, URLError, ValueError):
                return None
        self.snapshot, self.chat_snapshot = await asyncio.gather(get(RUNTIME_PORT), get(CHAT_PORT))
        await self.poll_logs()
        def occupied(port):
            import socket
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.1):
                    return True
            except OSError:
                return False
        for control_port, snapshot, service_port in ((RUNTIME_PORT, self.snapshot, 8000),
                                                     (CHAT_PORT, self.chat_snapshot, self.web_port)):
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
        port, name = (CHAT_PORT, 'Web Chat') if chat else (RUNTIME_PORT, 'LLM Runtime')
        self.message = f'{name}: {action} requested'
        try:
            try:
                status = await asyncio.to_thread(call, port)
            except (OSError, URLError):
                status = None
            if action == 'toggle':
                action = 'stop' if status else 'start'
            endpoint = (status or {}).get('endpoint') or {}
            if action == 'stop' and status:
                await asyncio.to_thread(call, port, '/ds41f/control/shutdown', 'POST')
                self.message = name+' graceful shutdown requested'
                for _ in range(1200):
                    await asyncio.sleep(.1)
                    await self.poll_logs()
                    try:
                        await asyncio.to_thread(call, port)
                    except (OSError, URLError):
                        break
                else:
                    raise RuntimeError('shutdown still in progress')
            if action == 'start':
                if status:
                    self.message = name+' already running'
                else:
                    import socket
                    service_port = self.web_port if chat else 8000
                    try:
                        with socket.create_connection(('127.0.0.1', service_port), timeout=.2):
                            raise RuntimeError('CONFLICT: service port occupied without ds41f control; no launch attempted')
                    except ConnectionRefusedError:
                        pass
                    if chat:
                        if self.web_bind not in ('lan', 'loopback'):
                            raise ValueError('Web Chat bind must be lan or loopback')
                        endpoint = dict(host='0.0.0.0' if self.web_bind == 'lan' else '127.0.0.1',
                                        port=self.web_port, allow_private_lan=self.web_bind == 'lan')
                    self.children[port] = (launch(True, endpoint) if chat else
                                           launch(default_reasoning_effort=self.default_reasoning_effort))
                    self.message = name+' canonical launcher started'
            await self.poll()
        except Exception as exc:
            self.message = name+f' control failed: {exc}'
        finally:
            self.operation = False

    async def open_browser(self):
        endpoint = (self.chat_snapshot or {}).get('endpoint') or {}
        if not endpoint:
            self.message = 'Web Chat is stopped; W: Start Web Chat'
            return
        url = endpoint.get('local_url') or f'http://127.0.0.1:{endpoint.get("port", 8080)}'
        self.message = f'Opening Web Chat: {url}'
        try:
            args = (['/usr/bin/open', url] if sys.platform == 'darwin' else
                    [sys.executable, '-m', 'webbrowser', '-t', url])
            process = await asyncio.create_subprocess_exec(
                *args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                code = await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()  # Launch helper only; never the browser.
                await process.wait()
                raise RuntimeError('browser launch helper timed out')
            if code:
                raise RuntimeError(f'browser launch helper exited {code}')
            self.message = f'Web Chat browser open requested: {url}'
        except Exception as exc:
            self.message = f'Browser launch failed: {exc}; open manually: {url}'

    async def run(self, screen, curses):
        curses.raw()  # No Ctrl-key workflows, including accidental SIGINT.
        curses.set_escdelay(25)
        screen.nodelay(True)
        screen.keypad(True)
        styles = terminal_styles(curses)
        poll_task = action_task = browser_task = None
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
                lines = render_screen(self.snapshot, Viewport(width, height), now=now,
                    mode=self.mode, conflict=RUNTIME_PORT in self.conflicts,
                    chat=self.chat_snapshot, chat_conflict=CHAT_PORT in self.conflicts,
                    message=self.message, logs=list(self.logs), log_end=self.log_end,
                    log_gap=self.log_gap, settings={'bind': self.web_bind, 'port': self.web_port,
                                                  'runtime_python': runtime_python(),
                                                  'default_reasoning_effort': self.default_reasoning_effort})
                if self.mode == 'dashboard':
                    self.offset = 0
                    visible = lines
                else:
                    self.offset = max(0, min(self.offset, max(0, len(lines)-height+1)))
                    visible = lines[self.offset:self.offset+height-1]
                screen.erase()
                for row, line in enumerate(visible[:max(0, height-1)]):
                    try:
                        screen.addnstr(row, 1, line.text, max(0, width-2),
                                       styles.get(line.tone, 0) | (curses.A_BOLD if line.bold else 0))
                        for start, length in line.values:
                            if start < width-2:
                                screen.addnstr(row, 1+start, line.text[start:start+length],
                                               min(length, width-2-start), curses.A_BOLD)
                        for start, length, tone in line.accents:
                            if start < width-2:
                                screen.addnstr(row, 1+start, line.text[start:start+length],
                                               min(length, width-2-start), styles.get(tone, 0) | (0 if tone == 'dim' else curses.A_BOLD))
                    except curses.error:
                        pass
                try:
                    screen.addnstr(max(0, height-1), 0,
                        command_strip(self.mode, width, self.snapshot, self.chat_snapshot), max(0, width-1), curses.A_DIM)
                    curses.curs_set(0)
                except curses.error:
                    pass
                screen.refresh()
                try:
                    key = screen.get_wch()
                except curses.error:
                    key = None
                if key in ('q', 'Q'):
                    break
                if key in ('r', 'R', 'w', 'W'):
                    if action_task is None or action_task.done():
                        action_task = asyncio.create_task(self.lifecycle('toggle', key in ('w', 'W')))
                    else:
                        self.message = 'Service control already in progress'
                elif key in ('o', 'O'):
                    if browser_task is None or browser_task.done():
                        browser_task = asyncio.create_task(self.open_browser())
                elif key in ('s', 'S', 'd', 'D', '?'):
                    target = 'settings' if key in ('s', 'S') else 'diagnostics' if key in ('d', 'D') else 'help'
                    self.mode = 'dashboard' if self.mode == target else target
                    self.offset = 0
                elif self.mode == 'settings':
                    if key in ('t', 'T'):
                        from .config import REASONING_EFFORTS
                        from .operator_settings import save_reasoning
                        value = REASONING_EFFORTS[(REASONING_EFFORTS.index(self.default_reasoning_effort) + 1) % len(REASONING_EFFORTS)]
                        try:
                            save_reasoning(value)
                            self.default_reasoning_effort = value
                            self.message = 'Default Reasoning saved; applies on next Runtime start'
                        except OSError as exc:
                            self.message = f'Could not save Runtime setting: {exc}'
                    elif key in ('l', 'L'):
                        self.web_bind = 'loopback' if self.web_bind == 'lan' else 'lan'
                    elif key in ('+', '-'):
                        self.web_port = max(1024, min(65535, self.web_port + (1 if key == '+' else -1)))
                elif self.mode == 'dashboard':
                    page = max(1, sum(1 for line in lines if ' #' in line.text)-1)
                    if key in (curses.KEY_UP, curses.KEY_PPAGE) and self.logs:
                        current = self.log_end if self.log_end is not None else self.logs[-1]['sequence']
                        self.log_end = max(self.logs[0]['sequence'], current-(page if key == curses.KEY_PPAGE else 1))
                    elif key in (curses.KEY_DOWN, curses.KEY_NPAGE) and self.logs and self.log_end is not None:
                        self.log_end += page if key == curses.KEY_NPAGE else 1
                        if self.log_end >= self.logs[-1]['sequence']:
                            self.log_end = None
                    elif key in ('l', 'L', curses.KEY_END):
                        self.log_end = None
                elif key in (curses.KEY_UP, curses.KEY_PPAGE):
                    self.offset -= 1 if key == curses.KEY_UP else max(1, height-3)
                elif key in (curses.KEY_DOWN, curses.KEY_NPAGE):
                    self.offset += 1 if key == curses.KEY_DOWN else max(1, height-3)
                await asyncio.sleep(.04)
        finally:
            if browser_task is not None:
                await browser_task  # Bounded launch helper; browser is not owned.
            for task in (poll_task, action_task):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(t for t in (poll_task, action_task) if t is not None), return_exceptions=True)


def main():
    try:
        import curses
    except ImportError:
        print('TUI requires Python curses support and a terminal.', file=sys.stderr)
        return 2
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print('ds41f TUI requires a terminal; use headless subcommands instead.', file=sys.stderr)
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
