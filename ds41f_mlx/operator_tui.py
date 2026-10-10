"""Stdlib TUI: read-only projection consumer and canonical launcher client."""
from __future__ import annotations

import asyncio
import json
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
    if endpoint:
        args += ['--host', endpoint['host'], '--port', str(endpoint['port'])]
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
                    message=self.message)
                # Main dashboard never scrolls. Secondary views retain the header.
                header_count = 1 if width < 70 else 2
                if self.mode == 'dashboard':
                    self.offset = 0
                    visible = lines
                else:
                    body_height = max(0, height-1-header_count)
                    self.offset = max(0, min(self.offset, len(lines)-header_count-body_height))
                    visible = lines[:header_count] + lines[header_count+self.offset:header_count+self.offset+body_height]
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
                                               min(length, width-2-start), styles.get(tone, 0) | curses.A_BOLD)
                    except curses.error:
                        pass
                try:
                    screen.addnstr(max(0, height-1), 0, command_strip(self.mode, width), max(0, width-1), curses.A_DIM)
                except curses.error:
                    pass
                screen.refresh()
                key = screen.getch()
                if key in (ord('q'), ord('Q')):
                    break
                if key in (ord('d'), ord('D'), ord('?')):
                    target = 'help' if key == ord('?') else 'diagnostics'
                    self.mode = 'dashboard' if self.mode == target else target
                    self.offset = 0
                if self.mode != 'dashboard' and key == curses.KEY_UP:
                    self.offset -= 1
                elif self.mode != 'dashboard' and key == curses.KEY_DOWN:
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
