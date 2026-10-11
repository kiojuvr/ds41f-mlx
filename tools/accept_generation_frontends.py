"""Actual curses/Chrome driver. Requires READY production runtime + Chat Server.

HTTP is observation only; every model/tool/cancel operation originates in the UI.
No fake responses, POST retries or runtime policy overrides. Collector deps only:
pyte (TUI), playwright + installed Chrome (Browser).
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.request import urlopen, Request


def api(path):
    with urlopen('http://127.0.0.1:8080' + path, timeout=5) as response:
        return json.load(response)


class Evidence:
    def __init__(self, out):
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.records = {}
        for _ in range(360):
            try: self.contract = api('/api/status')['runtime']
            except OSError: self.contract = {}
            if self.contract.get('status') == 'ready': break
            time.sleep(1)
        else: raise TimeoutError('Production runtime not READY; no frontend requests issued')
        self.execution_path = Path(__file__).resolve().parents[1] / 'artifacts/generation-policy/frontend-execution.jsonl'
        self.trace_start = len(self.all_executions())

    def all_executions(self):
        return [json.loads(line) for line in self.execution_path.read_text().splitlines()] if self.execution_path.exists() else []

    def executions(self):
        return self.all_executions()[self.trace_start:]

    def tools(self, base, expected):
        records = self.executions()[base:]
        assert any(r.get('tool_calls') for r in records)
        ids = [c['id'] for r in records for c in r.get('tool_calls', [])]
        assert len(ids) == len(set(ids)), 'Duplicate canonical tool call identity'
        for r in records:
            self.policy({'last_turn': r}, **expected)

    def observe(self, sid):
        rec = api('/api/session/' + sid)
        if rec['state'] == 'idle' and rec['request_count']:
            diag = rec['diagnostics']['m8']
            assert diag['all_cache_offsets_equal_frontier']
            assert diag['total_prompt_replay_count'] == diag['total_full_cache_repack_count'] == 0
            assert set(diag['cache_offsets_all'] + diag['dspark_offsets']) == {diag['frontier']}
            self.records[sid + ':' + str(rec['request_count'])] = rec
            (self.out / 'settlements.json').write_text(json.dumps(self.records, indent=2))
        return rec

    def policy(self, rec, **expected):
        actual = rec['last_turn']['generation_policy']  # execution trace, not UI/wire echo
        for name, value in (expected or self.contract['generation_policy']['defaults']).items():
            assert (actual[name] is None if value is None else math.isclose(actual[name], value, abs_tol=1e-6)), (name, actual, expected)

    def finish(self, frontend):
        executions = self.executions()
        assert executions, 'Run the pass-through runtime settlement observer'
        for r in executions:
            assert r['prompt_replay'] == r['full_cache_repack'] == 0
            assert set(r['target_offsets'] + r['dspark_offsets']) == {r['canonical_frontier']}
        (self.out / 'receipt.json').write_text(json.dumps(dict(frontend=frontend, status='REAL_MODEL_PASS', runtime=self.contract,
            settlements=self.records, executions=executions, physical_ime='not exercised by automation'), indent=2))


def active_decode():
    with urlopen('http://127.0.0.1:18441/ds41f/status', timeout=3) as response:
        current = json.load(response).get('current_request') or {}
    return current.get('current_phase') == 'DECODING' and current.get('metrics', {}).get('generated', 0) >= 20


def restart_chat(args, evidence):
    # Infrastructure fault injection only; never a model/tool POST or resend.
    with urlopen(Request('http://127.0.0.1:18443/ds41f/control/shutdown', method='POST', data=b''), timeout=5):
        pass
    end = time.monotonic() + 60
    while time.monotonic() < end:
        try:
            with urlopen('http://127.0.0.1:18443/ds41f/status', timeout=2): pass
        except OSError: break
        time.sleep(.1)
    else: raise TimeoutError('Chat Server did not retire')
    subprocess.Popen([args.python, '-I', '-m', 'ds41f_mlx.web'], cwd='/tmp', stdin=subprocess.DEVNULL,
        stdout=(evidence.out / 'chat-reconnect.log').open('w'), stderr=subprocess.STDOUT, start_new_session=True)
    end = time.monotonic() + 30
    while time.monotonic() < end:
        try:
            if api('/api/status')['runtime']['status'] == 'ready': return
        except OSError: pass
        time.sleep(.1)
    raise TimeoutError('Chat Server reconnect')


def tui(args, evidence):
    import fcntl
    import pty
    import select
    import struct
    import termios
    import pyte
    fd, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 40, 160, 0, 0))
    env = os.environ | {'TERM': 'xterm-256color', 'LANG': 'en_US.UTF-8'}
    child = subprocess.Popen([args.python, '-I', '-m', 'ds41f_mlx.ops'], cwd='/tmp', env=env,
        stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    class TerminalScreen(pyte.Screen):
        # ncurses uses ECMA-48 SU/SD; pyte 0.8 omits them. Preserve raw ANSI
        # untouched and implement only the terminal's actual scroll-region ops.
        def scroll_up(self, count=1):
            row = self.cursor.y
            self.cursor.y = self.margins.bottom if self.margins else self.lines-1
            for _ in range(count or 1): self.index()
            self.cursor.y = row
        def scroll_down(self, count=1):
            row = self.cursor.y
            self.cursor.y = self.margins.top if self.margins else 0
            for _ in range(count or 1): self.reverse_index()
            self.cursor.y = row
    class TerminalStream(pyte.ByteStream):
        csi = dict(pyte.ByteStream.csi, S='scroll_up', T='scroll_down')
    screen = TerminalScreen(160, 40); stream = TerminalStream(screen); raw = bytearray()
    def pump(seconds=.2):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if select.select([fd], [], [], .02)[0]:
                data = os.read(fd, 65536); raw.extend(data); stream.feed(data)
    def key(value):
        os.write(fd, value); pump()
    def text(): return '\n'.join(screen.display)
    def shot(name):
        pump(); (evidence.out / (name + '.txt')).write_text(text())
        (evidence.out / 'terminal.ansi').write_bytes(raw)
    def wait(predicate, seconds=240):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            pump(.1)
            try: result = predicate()
            except (OSError, KeyError, ValueError): result = None
            if result: return result
        shot('failure'); raise TimeoutError(text())
    def sid(): return re.search(r'tui-[0-9a-f-]{36}', text()).group()
    def settle(session, count, name):
        rec = wait(lambda: (r if r['state'] == 'idle' and r['request_count'] >= count else None)
                   if (r := evidence.observe(session)) else None)
        wait(lambda: 'idle' in text().splitlines()[0])
        shot(name); return rec
    def send(value):
        key(b'\x1b[200~' + value.encode() + b'\x1b[201~'); key(b'\x07')
    index = 0
    def setting(target, value):
        nonlocal index
        key(b'\x0f'); key(b'\t' * ((target-index) % 9)); index = target
        key(b'\x15' + value.encode()); key(b'\x07'); shot('setting-' + str(target)); key(b'\x0f')
    try:
        pump(1); key(b'c'); wait(lambda: api('/api/status')['runtime'].get('status') == 'ready', 360)
        key(b'\x0e'); wait(lambda: re.search(r'tui-[0-9a-f-]{36}', text()))
        session = sid()
        send('Say hello in one short sentence.'); rec = settle(session, 1, 'default'); evidence.policy(rec)
        setting(1, '.65'); setting(2, '.8')
        send('Say hello again, in Japanese.'); rec = settle(session, 2, 'temperature-top-p'); evidence.policy(rec, temperature=.65, top_p=.8)
        setting(3, '40'); setting(4, '.03')
        # Exercise advanced controls, not only their presence.
        setting(5, '1.03'); setting(6, '.1'); setting(7, '.1'); setting(8, '123')
        setting(8, '9' * 400); assert 'finite integer' in text()
        expected = dict(temperature=.65, top_p=.8, top_k=40, min_p=.03, repetition_penalty=1.03, presence_penalty=.1, frequency_penalty=.1, seed=123)
        send('What is two plus two? Answer in one short sentence.'); rec = settle(session, 3, 'all-sampling'); evidence.policy(rec, **expected)
        key(b'\x14'); key(b'c'); shot('mode-return')
        restart_chat(args, evidence); key(b'\x12'); wait(lambda: 'idle' in text().splitlines()[0]); shot('server-reconnected')
        assert evidence.observe(session)['request_count'] == 3
        send('Remember the preceding answer. Reply with the number only.'); rec = settle(session, 4, 'multi-turn'); evidence.policy(rec, **expected)
        key(b'\x0e'); wait(lambda: sid() != session); session = sid(); setting(0, 'low')
        send('A box contains 3 red and 5 blue balls. Draw two without replacement. Think carefully about the probability they have the same color, then give the reduced fraction.'); rec = settle(session, 1, 'reasoning-on'); evidence.policy(rec, **expected)
        assert rec['last_turn']['response_json']['choices'][0]['message'].get('reasoning_content')
        setting(0, 'high'); assert 'session-owned' in text() or 'new chat' in text()
        shot('reasoning-locked')
        base = len(evidence.executions())
        send('Use fetch_url exactly once to fetch https://example.com/ and report its page title.'); rec = settle(session, 3, 'tools'); evidence.policy(rec, **expected)
        evidence.tools(base, expected)
        # Cancel a public text response, not an unreconstructible thinking prefix.
        key(b'\x0e'); wait(lambda: sid() != session); session = sid(); setting(0, 'none')
        send('Use only general knowledge; do not call any tools, research or browse. Give a substantial five-section explanation with Python examples of csv.DictReader and Decimal validation: UTF-8 headers, quoting, missing columns, blanks and invalid numbers.')
        wait(lambda: 'Streaming (unsettled)' in text() and active_decode())
        shot('streaming'); key(b'\x03'); rec = settle(session, 1, 'cancel-reconcile')
        assert rec['last_turn']['cancelled']; evidence.policy(rec, **expected)
        key(b'\x12'); shot('explicit-reconcile')
        send('After Stop, answer OK only.'); rec = settle(session, 2, 'after-cancel'); evidence.policy(rec, **expected)
        setting(1, '.75'); expected['temperature'] = .75
        send('What was the last answer? Be brief.'); rec = settle(session, 3, 'subsequent-policy'); evidence.policy(rec, **expected)
        send('Now say goodbye briefly.'); rec = settle(session, 4, 'subsequent-multi-turn'); evidence.policy(rec, **expected)
        key(b'\x11'); child.wait(timeout=90); assert child.returncode == 0
        evidence.finish('actual curses PTY')
    finally:
        if child.poll() is None:
            key(b'\x11')
            try: child.wait(timeout=90)
            except subprocess.TimeoutExpired: child.terminate()
        os.close(fd)


def browser(args, evidence):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw, ExitStack() as cleanup:
        browser = pw.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 1000}); errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('dialog', lambda dialog: dialog.accept())
        def observe_wire(request):
            if '/api/' in request.url:
                with (evidence.out / 'wire.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(method=request.method, url=request.url, body=request.post_data)) + '\n')
        page.on('request', observe_wire)
        def retire_ui():
            if page.is_closed(): return
            try:
                if page.evaluate('() => !!state.session'):
                    if page.locator('#stop').is_visible(): page.locator('#stop').click()
                    page.wait_for_function('() => !state.busy', timeout=120000)
                    if not page.locator('#settingsDialog').evaluate('(node) => node.open'): page.locator('#settingsButton').click()
                    page.locator('#closeSession').click()
                    page.wait_for_function('() => !state.session && !state.busy')
            except Exception as error:
                print('UI retirement unconfirmed; known identity is in wire.jsonl; no retry:', error)
        cleanup.callback(retire_ui)
        page.goto('http://127.0.0.1:8080')
        page.wait_for_function('() => state.session && !state.busy && state.runtime.status === "ready"')
        for name, spec in evidence.contract['generation_policy']['parameters'].items():
            node = page.locator('#policy_' + name)
            if 'maximum' in spec: assert float(node.get_attribute('max')) == spec['maximum']
            if 'minimum' in spec: assert float(node.get_attribute('min')) == spec['minimum']
            if 'exclusiveMinimum' in spec: assert str(spec['exclusiveMinimum']) in node.get_attribute('title')
        def sid(): return page.evaluate('state.session.id')
        def shot(name): page.screenshot(path=str(evidence.out / (name + '.png')))
        def settle(session, count, name):
            end = time.monotonic() + 300
            while time.monotonic() < end:
                rec = evidence.observe(session)
                if rec['state'] == 'idle' and rec['request_count'] >= count:
                    page.wait_for_function('() => !state.busy'); shot(name); return rec
                page.wait_for_timeout(100)
            raise TimeoutError(page.locator('#notice').inner_text())
        def send(value): page.locator('#input').fill(value); page.locator('#send').click()
        def setting(name, value):
            page.locator('#settingsButton').click()
            if name in ('seed', 'repetition_penalty', 'presence_penalty', 'frequency_penalty'):
                page.locator('#samplingControls details').evaluate('(node) => node.open = true')
            page.locator('#policy_' + name).fill(value); page.locator('#policy_' + name).dispatch_event('change')
            page.locator('#settingsClose').click()
        session = sid(); send('Say hello briefly.'); rec = settle(session, 1, 'default'); evidence.policy(rec)
        expected = dict(temperature=.65, top_p=.8, top_k=40, min_p=.03, repetition_penalty=1.03, presence_penalty=.1, frequency_penalty=.1, seed=123)
        for name, value in expected.items(): setting(name, str(value))
        send('What is two plus two? Reply briefly.'); rec = settle(session, 2, 'all-sampling'); evidence.policy(rec, **expected)
        page.locator('#settingsButton').click(); page.locator('#outputLimit').select_option('custom'); page.locator('#maxTokens').fill('24'); page.locator('#maxTokens').dispatch_event('change'); page.locator('#settingsClose').click()
        send('Write integers 1 through 1000, one per line without explanation.'); rec = settle(session, 3, 'length-boundary'); assert rec['last_turn']['finish_reason'] == 'length'; evidence.policy(rec, **expected)
        setting('temperature', '.75')
        assert 'Continuation snapshot:' in page.locator('#effectivePolicy').inner_text()
        page.get_by_role('button', name='Continue response', exact=True).click(); rec = settle(session, 4, 'explicit-continuation'); evidence.policy(rec, **expected)
        # The original workflow retains .65; a subsequent user turn owns .75.
        page.locator('#settingsButton').click(); page.locator('#outputLimit').select_option('auto'); page.locator('#settingsClose').click()
        send('Answer OK only.'); rec = settle(session, 5, 'after-explicit-continuation'); evidence.policy(rec, **(expected | {'temperature': .75}))
        setting('temperature', '.65')
        page.locator('#newSession').click(); page.wait_for_function('(old) => state.session && state.session.id !== old && !state.busy', arg=session)
        session = sid()
        page.locator('#thinkLabel').click(); page.locator('#thinking').select_option('on'); page.locator('#reasoning').select_option('low'); page.locator('#thinkLabel').click()
        send('A box contains 3 red and 5 blue balls. Draw two without replacement. Think carefully about the probability they have the same color, then give the reduced fraction.'); rec = settle(session, 1, 'reasoning-on'); evidence.policy(rec, **expected)
        assert rec['last_turn']['response_json']['choices'][0]['message'].get('reasoning_content')
        page.locator('#thinkLabel').click()
        assert page.locator('#thinking').is_disabled() and page.locator('#reasoning').is_disabled()
        assert 'new chat' in page.locator('#thinkingLock').inner_text()
        shot('reasoning-locked'); page.locator('#thinkLabel').click()
        base = len(evidence.executions())
        send('Use fetch_url exactly once to fetch https://example.com/ and report its page title.'); rec = settle(session, 3, 'tools'); evidence.policy(rec, **expected)
        evidence.tools(base, expected)
        page.locator('#newSession').click(); page.wait_for_function('(old) => state.session && state.session.id !== old && !state.busy', arg=session)
        session = sid(); page.locator('#thinkLabel').click(); page.locator('#thinking').select_option('off'); page.locator('#thinkLabel').click()
        send('Use only general knowledge; do not call any tools, research or browse. Give a substantial five-section explanation with Python examples of csv.DictReader and Decimal validation: UTF-8 headers, quoting, missing columns, blanks and invalid numbers.')
        page.wait_for_function('() => state.live && state.live.content.length > 80'); shot('streaming')
        page.locator('#stop').click(); rec = settle(session, 1, 'cancel-reconcile'); assert rec['last_turn']['cancelled']; evidence.policy(rec, **expected)
        restart_chat(args, evidence)
        page.locator('#settingsButton').click(); page.locator('#reconnect').click(); page.wait_for_function('() => !state.busy'); page.locator('#settingsClose').click(); shot('server-reconnected')
        assert evidence.observe(session)['request_count'] == 1
        send('After Stop, answer OK only.'); rec = settle(session, 2, 'after-cancel'); evidence.policy(rec, **expected)
        setting('temperature', '.75'); expected['temperature'] = .75
        send('Say goodbye briefly.'); rec = settle(session, 3, 'subsequent-policy'); evidence.policy(rec, **expected)
        assert not errors, errors
        page.locator('#settingsButton').click(); page.locator('#closeSession').click(); page.wait_for_function('() => !state.session && !state.busy')
        evidence.finish('actual Chrome Browser Chat'); browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frontend', choices=['tui', 'browser'])
    parser.add_argument('--python', default='/Volumes/SDXC-512/ds41f-mlx_mtp-venv/bin/python')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); evidence = Evidence(args.output)
    globals()[args.frontend](args, evidence)
    print('REAL_MODEL_PASS', args.frontend, flush=True)
