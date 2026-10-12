"""Stdlib TUI: read-only projection consumer and canonical launcher client."""
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


def launch(chat=False, endpoint=None):
    args = ['ds41f_mlx.web'] if chat else ['ds41f_mlx.ops', 'start', '--profile', 'mtp-serving-v1']
    if chat and endpoint is None:
        bind = os.environ.get('DS41F_CHAT_BIND', 'lan')
        if bind not in ('lan', 'loopback'):
            raise ValueError('DS41F_CHAT_BIND must be lan or loopback')
        endpoint = dict(host='0.0.0.0' if bind == 'lan' else '127.0.0.1',
                        port=int(os.environ.get('DS41F_WEB_PORT', '8080')), allow_private_lan=bind == 'lan')
    if endpoint:
        if chat and endpoint.get('scheme') == 'https':
            raise ValueError('TLS Chat restart requires manual certificate/key options; refusing HTTP downgrade')
        args += ['--host', endpoint['host'], '--port', str(endpoint['port'])]
        if chat:
            if endpoint.get('allow_private_lan', endpoint['host'] not in ('127.0.0.1', 'localhost', '::1')):
                args.append('--allow-private-lan')
            if endpoint.get('runtime_url'):
                args += ['--runtime-url', endpoint['runtime_url']]
    return subprocess.Popen([sys.executable, '-m', *args], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def render(snapshot, now=None):
    """Text projection for headless checks; terminal uses explicit viewport layouts."""
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
        self.conflicts = set()
        from collections import deque
        from .runtime_logging import CAPACITY
        self.logs = deque(maxlen=CAPACITY)
        self.log_runtime = None
        self.log_cursor = None
        self.log_end = None  # sequence anchor; None follows latest
        self.log_gap = False
        self.log_lock = asyncio.Lock()
        from .chat_application import ChatApplication
        from .operator_chat import ChatView
        self.conversation = ChatApplication()
        self.chat_view = ChatView()
        self.chat_task = self.stop_task = None

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
            pass  # keep the final tail visible after runtime exits

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
            if chat and action == 'restart' and endpoint.get('scheme') == 'https':
                raise ValueError('TLS Chat restart requires manual certificate/key options; server kept running')
            if chat and action in ('stop', 'restart') and self.conversation.id:
                # Retire our application session before taking its backend away.
                await self.conversation.stop()
                if self.chat_task and not self.chat_task.done():
                    await asyncio.shield(self.chat_task)
                await self.conversation.close()
            if action in ('stop', 'restart') and status:
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
                    raise RuntimeError('shutdown still in progress; Start not attempted')
                status = None
            if action in ('start', 'restart'):
                if status:
                    self.message = name+' already running'
                else:
                    # Protect against legacy/unrelated process on the service port.
                    import socket
                    service_port = endpoint.get('port', int(os.environ.get('DS41F_WEB_PORT', '8080')) if chat else 8000)
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
            if chat and self.mode == 'chat':
                self.conversation.notice = self.message
            self.operation = False

    async def new_chat(self):
        if not self.conversation.id:
            endpoint = (self.chat_snapshot or {}).get('endpoint') or {}
            port = endpoint.get('port', 8080)
            self.conversation.http.base_url = f'http://127.0.0.1:{port}'
        await self.conversation.new()

    async def chat_action(self, action):
        try:
            await action()
        except Exception as exc:
            self.conversation.notice = str(exc)

    def begin_chat_action(self, action):
        if self.chat_task is None or self.chat_task.done():
            self.chat_task = asyncio.create_task(self.chat_action(action))

    async def run(self, screen, curses):
        # raw preserves Ctrl-C as application Stop, not process SIGINT.
        curses.raw()
        curses.set_escdelay(25)  # unknown ESC/paste sequences cannot stall polling
        sys.stdout.write('\x1b[?2004h')
        sys.stdout.flush()
        screen.nodelay(True)
        screen.keypad(True)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        styles = terminal_styles(curses)
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
                lines = render_screen(self.snapshot, Viewport(width, height), now=now,
                    mode=self.mode, conflict=RUNTIME_PORT in self.conflicts,
                    chat=self.chat_snapshot, chat_conflict=CHAT_PORT in self.conflicts,
                    message=self.message, logs=list(self.logs), log_end=self.log_end,
                    log_gap=self.log_gap)
                chat_cursor = None
                if self.mode == 'chat':
                    lines, chat_cursor = self.chat_view.render(self.conversation, width, height,
                        (self.snapshot or {}).get('state', 'STOPPED'),
                        (self.chat_snapshot or {}).get('state', 'STOPPED — F2 Start'),
                        self.chat_task is not None and not self.chat_task.done())
                # Main dashboard never scrolls. Secondary views retain the header.
                header_count = 1 if width < 70 else 2
                if self.mode in ('dashboard', 'chat'):
                    self.offset = 0
                    visible = lines
                else:
                    body_height = max(0, height-1-header_count)
                    self.offset = max(0, min(self.offset, len(lines)-header_count-body_height))
                    visible = lines[:header_count] + lines[header_count+self.offset:header_count+self.offset+body_height]
                screen.erase()
                for row, line in enumerate(visible[:height if self.mode == 'chat' else max(0, height-1)]):
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
                                               min(length, width-2-start), styles.get(tone, 0) | curses.A_BOLD)
                    except curses.error:
                        pass
                try:
                    if self.mode != 'chat':
                        screen.addnstr(max(0, height-1), 0, command_strip(self.mode, width), max(0, width-1), curses.A_DIM)
                    curses.curs_set(1 if chat_cursor else 0)
                    if chat_cursor:
                        screen.move(min(height-1, chat_cursor[0]), min(width-1, chat_cursor[1]))
                except curses.error:
                    pass
                screen.refresh()
                # Wide-character input: IME commits UTF-8 text, not getch bytes.
                # Drain paste/edit events without blocking polling or streams.
                keys = []
                for _ in range(4096):
                    try:
                        key = screen.get_wch()
                    except curses.error:
                        break
                    key = self.chat_view.active_editor.decode(key)
                    if key is not None:
                        keys.append(key)
                if self.mode == 'chat':
                    for key in keys:
                        if self.chat_view.setting_key(key, self.conversation, curses,
                                self.chat_task is not None and not self.chat_task.done()):
                            continue
                        if key == '\x11':
                            return
                        if key == '\x14':
                            self.mode = 'dashboard'
                        elif key == '\x0c':
                            self.chat_view.top = None
                        elif key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
                            self.chat_view.scroll((-1 if key == curses.KEY_PPAGE else 1)*self.chat_view.body_height)
                        elif key == '\x03':
                            if self.stop_task is None or self.stop_task.done():
                                self.stop_task = asyncio.create_task(self.chat_action(self.conversation.stop))
                        elif key in ('\x0e', '\x17', '\x12', '\x05', '\x07'):
                            actions = {'\x0e': self.new_chat, '\x17': self.conversation.close,
                                       '\x12': lambda: self.conversation.reconcile(wait=True),
                                       '\x05': self.conversation.resume}
                            if key == '\x07':
                                text = self.chat_view.editor.text.strip()
                                if text and (self.chat_task is None or self.chat_task.done()):
                                    async def send(text=text):
                                        def submitted():
                                            self.chat_view.editor.text = ''
                                            self.chat_view.editor.cursor = 0
                                        await self.conversation.send(text, on_submitted=submitted)
                                    self.chat_view.top = None
                                    self.begin_chat_action(send)
                            else:
                                self.begin_chat_action(actions[key])
                        elif key in (curses.KEY_F2, curses.KEY_F3):
                            if action_task is None or action_task.done():
                                action_task = asyncio.create_task(self.lifecycle('start' if key == curses.KEY_F2 else 'stop', True))
                        else:
                            self.chat_view.editor.edit(key, curses, max(2, width-2))
                    await asyncio.sleep(.04)
                    continue
                # Dashboard command path stays single-key, never interprets paste.
                key = next((key for key in keys if key != curses.KEY_RESIZE), -1)
                if isinstance(key, str):
                    key = ord(key)
                if key in (ord('c'), ord('C')):
                    self.mode = 'chat'
                    continue
                if key in (ord('o'), ord('O')):
                    endpoint = (self.chat_snapshot or {}).get('endpoint') or {}
                    if endpoint:
                        import webbrowser
                        webbrowser.open(endpoint.get('local_url') or f'http://127.0.0.1:{endpoint.get("port", 8080)}')
                    else:
                        self.message = 'Chat Server is stopped; use C then F2 to start'
                    continue
                if key in (ord('q'), ord('Q')):
                    break
                if key in (ord('d'), ord('D'), ord('?')):
                    target = 'help' if key == ord('?') else 'diagnostics'
                    self.mode = 'dashboard' if self.mode == target else target
                    self.offset = 0
                if self.mode == 'dashboard':
                    page = max(1, sum(1 for line in lines if ' #' in line.text)-1)
                    if key in (curses.KEY_UP, curses.KEY_PPAGE) and self.logs:
                        current = self.log_end if self.log_end is not None else self.logs[-1]['sequence']
                        self.log_end = max(self.logs[0]['sequence'], current-(page if key == curses.KEY_PPAGE else 1))
                    elif key in (curses.KEY_DOWN, curses.KEY_NPAGE) and self.logs and self.log_end is not None:
                        self.log_end += page if key == curses.KEY_NPAGE else 1
                        if self.log_end >= self.logs[-1]['sequence']:
                            self.log_end = None
                    elif key in (ord('l'), ord('L'), curses.KEY_END):
                        self.log_end = None
                if self.mode != 'dashboard' and key == curses.KEY_UP:
                    self.offset -= 1
                elif self.mode != 'dashboard' and key == curses.KEY_DOWN:
                    self.offset += 1
                commands = {'s': ('start', False), 'x': ('stop', False), 'r': ('restart', False),
                            'v': ('stop', True)}
                command = commands.get(chr(key).lower()) if 0 <= key < 256 else None
                if command and (action_task is None or action_task.done()):
                    action_task = asyncio.create_task(self.lifecycle(*command))
                await asyncio.sleep(.1)
        finally:
            sys.stdout.write('\x1b[?2004l')
            sys.stdout.flush()
            # Retire only this frontend's native session, never either service.
            if self.conversation.id:
                try:
                    await asyncio.wait_for(self.conversation.stop(), 10)
                    if self.chat_task and not self.chat_task.done():
                        await asyncio.wait_for(asyncio.shield(self.chat_task), 65)
                except Exception as exc:
                    print(f'Chat interruption could not be confirmed: {exc}; attempting session retirement', file=sys.stderr)
                # Even if Stop/reconciliation failed, attempt authoritative DELETE.
                try:
                    await asyncio.wait_for(self.conversation.shutdown(), 65)
                except Exception as exc:
                    print(f'Chat session {self.conversation.id}: retirement unconfirmed: {exc}; no retry', file=sys.stderr)
            else:
                await self.conversation.http.aclose()
            # Cancel UI work only: never signal the detached runtime on quit.
            for task in (poll_task, action_task, self.chat_task, self.stop_task):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(t for t in (poll_task, action_task, self.chat_task, self.stop_task) if t is not None), return_exceptions=True)


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
