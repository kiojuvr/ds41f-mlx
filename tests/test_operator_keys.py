"""Actual curses input: operator keys only; no conversation workflow exists."""
import os
import subprocess
import sys
import time


def test_plain_mnemonic_operator_keys_and_no_old_chat_routing(tmp_path):
    import pty, fcntl, struct, termios, select
    journal = tmp_path/'keys'
    source = '''
import asyncio, curses
from pathlib import Path
from ds41f_mlx.operator_tui import Operator
op = Operator()
def record(value):
    with Path(%r).open('a') as f: f.write(value + '\\n')
async def poll(): pass
op.poll = poll
async def lifecycle(action): record('LLM Runtime:'+action)
op.lifecycle = lifecycle
curses.wrapper(lambda screen: asyncio.run(op.run(screen, curses)))
record('exit:'+op.mode)
assert not any(hasattr(op, name) for name in ('conversation', 'chat_view', 'open_browser', 'chat_snapshot', 'web_port'))
''' % str(journal)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 32, 110, 0, 0))
    def control():
        os.setsid(); fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
    child = subprocess.Popen([sys.executable, '-c', source], stdin=slave, stdout=slave,
        stderr=slave, preexec_fn=control, env={**os.environ, 'TERM': 'xterm-256color', 'DS41F_CHAT_BIND': 'lan'})
    output = bytearray()
    def pump(seconds=.2):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            if select.select([master], [], [], .02)[0]: output.extend(os.read(master, 65536))
    try:
        pump(.5)
        for key in (b'c', b'\x0e', b'\x11', b'\x1bOQ', b'\x1bOR', b'\x06', b'\x04'):
            os.write(master, key); pump(.08)
        assert child.poll() is None and not journal.exists()
        for key in (b'r', b'w', b'o', b's', b'l', b's', b'q'):
            os.write(master, key); pump()
        child.wait(timeout=5)
        assert child.returncode == 0
        assert journal.read_text().splitlines() == ['LLM Runtime:toggle', 'exit:dashboard']
        assert b'TUI SETTINGS' in output and b'Composer' not in output
    finally:
        if child.poll() is None: child.kill(); child.wait()
        os.close(master); os.close(slave)
