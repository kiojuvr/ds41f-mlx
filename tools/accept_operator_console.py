"""Current-source operator/Web product acceptance, not a core performance campaign.

Run from checkout with .venv. --phase ui validates source UI before installation;
--phase full additionally exercises the admitted logical-retention package.
Real Chrome, real PTY, official runtime; never substitutes model execution.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import termios
import time

import httpx
import pyte
from playwright.async_api import async_playwright

from ds41f_mlx.operator_control import RUNTIME_PORT, CHAT_PORT
from ds41f_mlx.operator_tui import call

ROOT = Path(__file__).resolve().parents[1]


class TerminalScreen(pyte.Screen):
    # ncurses uses standard SU/SD within scroll margins; pyte 0.8 omits them.
    # This models the terminal protocol, never repairs application output.
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


class Terminal:
    def __init__(self):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 42, 132, 0, 0))
        def control():
            os.setsid(); fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
        self.child = subprocess.Popen([sys.executable, '-m', 'ds41f_mlx.ops'],
            stdin=slave, stdout=slave, stderr=slave, cwd=ROOT, preexec_fn=control,
            env={**os.environ, 'TERM': 'xterm-256color'})
        os.close(slave)
        self.screen = TerminalScreen(132, 42); self.stream = TerminalStream(self.screen)
        self.output = bytearray()
        self.reader = asyncio.create_task(self.drain())

    async def drain(self):
        # Like a real terminal, continuously consume output while Browser actions
        # await the model. PTY backpressure is not a TUI responsiveness test.
        while self.child.poll() is None:
            await self.pump(.1)

    async def pump(self, seconds=.3):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            if select.select([self.master], [], [], 0)[0]:
                try: data = os.read(self.master, 65536)
                except OSError: break
                self.output.extend(data); self.stream.feed(data)
            await asyncio.sleep(.02)

    async def key(self, key):
        os.write(self.master, key.encode()); await self.pump()

    def text(self): return '\n'.join(self.screen.display)

    async def close(self):
        if self.child.poll() is None:
            await self.key('q')
            await asyncio.to_thread(self.child.wait, timeout=15)
        self.reader.cancel()
        await asyncio.gather(self.reader, return_exceptions=True)
        os.close(self.master)
        assert self.child.returncode == 0, self.output.decode(errors='replace')


async def status(port):
    try: return await asyncio.to_thread(call, port)
    except (OSError, ValueError): return None


async def wait_service(ui, port, ready):
    for _ in range(1500):
        await ui.pump(.2)
        value = await status(port)
        if (value is not None and value.get('state') == 'READY') if ready else value is None:
            return value
    raise TimeoutError(f'service {port}: {ui.text()}')


async def wait_page(page, expression, timeout=180000):
    # Direct DevTools evaluation, not Playwright's eval-based polling helper;
    # keep the application's strict CSP enabled during acceptance.
    until = time.monotonic()+timeout/1000
    while time.monotonic() < until:
        if await page.evaluate(expression): return
        await asyncio.sleep(.05)
    raise TimeoutError(expression+' '+str(await page.evaluate('({busy:state.busy,notice:state.notice,session:state.session})')))


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {'phase': args.phase, 'checks': {}, 'runtime_python': os.environ.get('DS41F_OPERATOR_RUNTIME_PYTHON')}
    def check(name, value=True):
        assert value, name
        receipt['checks'][name] = value
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))
        print('PASS', name, flush=True)
    if not args.reuse and (await status(RUNTIME_PORT) or await status(CHAT_PORT)):
        raise RuntimeError('Acceptance needs stopped services; do not take over another operator workload')
    ui = Terminal()
    try:
        await ui.pump(.6)
        check('TUI current source starts', 'LLM Runtime' in ui.text() and 'Web Chat' in ui.text())
        await ui.key('s'); check('S Settings', 'TUI SETTINGS' in ui.text()); await ui.key('s')
        if not await status(RUNTIME_PORT):
            await ui.key('r')
        await wait_service(ui, RUNTIME_PORT, True); check('R Start Runtime' if not args.reuse else 'Runtime rediscovered (start checked in prior receipt)')
        if not await status(CHAT_PORT):
            await ui.key('w')
        web = await wait_service(ui, CHAT_PORT, True); check('W Start Web Chat' if not args.reuse else 'Web Chat rediscovered (start checked in prior receipt)')
        await ui.pump(1)  # Control READY may precede the operator's next snapshot.
        url = web['endpoint']['local_url']
        await ui.key('o')
        for _ in range(70):
            await ui.pump(.2)
            if 'browser open requested' in ui.text(): break
        (args.output/'browser-open.txt').write_text(ui.text())
        check('O macOS open nonblocking', 'browser open requested' in ui.text())
        await ui.key('s'); check('TUI remains responsive after O', 'TUI SETTINGS' in ui.text()); await ui.key('s')
        screen = ui.text(); (args.output/'dashboard.txt').write_text(screen)
        (args.output/'dashboard.ansi').write_bytes(ui.output)
        check('obsolete TUI Chat shortcuts absent', all(k not in screen for k in ('F2', 'F3', 'Composer', '^N', '^G')))
        await ui.close(); check('Q leaves services running', bool(await status(RUNTIME_PORT)) and bool(await status(CHAT_PORT)))
        ui = Terminal(); await ui.pump(.6)
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
            context = await browser.new_context(viewport={'width': 1280, 'height': 1000})
            page = await context.new_page(); errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            await page.goto(url)
            await wait_page(page, 'state.session && state.runtime?.status === "ready"')
            a = await page.evaluate('state.session.id')
            await page.fill('#input', 'draft A')
            await page.click('#newSession')
            b = await page.evaluate('state.session.id')
            check('New adds independent conversation', a != b and await page.locator('#sessions option').count() == 2)
            await page.select_option('#sessions', a)
            check('Switch preserves draft A', await page.input_value('#input') == 'draft A')
            await page.select_option('#sessions', b)
            check('Switch B has independent draft', await page.input_value('#input') == '')
            await page.screenshot(path=str(args.output/'web.png'), full_page=True)
            # Independently exercise awaited attachment ownership on another tab;
            # no native state is allocated for these UI-only conversations.
            import base64
            from tools.qualify_pdf_documents import text_pdf
            fixture = args.output/'source.pdf'; fixture.write_bytes(text_pdf(1))
            upload = await context.new_page()
            upload.on('pageerror', lambda error: errors.append(str(error)))
            async def delayed(route):
                await asyncio.sleep(1); await route.continue_()
            await upload.route('**/api/documents', delayed)
            await upload.goto(url); await wait_page(upload, '!!state.session')
            await upload.evaluate("encoded => { const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0)); attachFiles([new File([bytes],'one.pdf',{type:'application/pdf'}),new File([bytes],'two.pdf',{type:'application/pdf'})]); }", base64.b64encode(fixture.read_bytes()).decode())
            await wait_page(upload, 'state.busy && state.session.documents?.length===1')
            await upload.click('#newSession')
            await wait_page(upload, '[...conversations.values()].every(c=>!c.busy)')
            files = await upload.evaluate('[...conversations.values()].map(c=>({documents:c.session.documents||[],native:c.session.native}))')
            check('Switch during file batch preserves original owner', len(files[0]['documents']) == 2 and all(d['state'] == 'retained' for d in files[0]['documents']) and not files[1]['documents'] and not any(c['native'] for c in files))
            receipt['attachment_switch'] = files
            await upload.close()
            if args.phase == 'full':
                # A abandoned tab whose cleanup beacon is deliberately lost.
                orphan = await context.new_page(); await orphan.goto(url)
                await wait_page(orphan, 'state.session && state.runtime?.status === "ready"')
                await orphan.evaluate('ensureRuntime(state)')
                orphan_id = await orphan.evaluate('state.session.id')
                await orphan.evaluate('navigator.sendBeacon = () => false')
                await orphan.close(); orphan_start = time.monotonic()
                # Two conversations in the SAME tab submit without awaiting the
                # other's delivery. Scheduler, not a browser-wide mutex, queues.
                await page.select_option('#sessions', a)
                await page.evaluate("$('outputLimit').value='custom'; $('maxTokens').value=128; policyChanged(); $('input').value='Remember the word ALPHA for this conversation. Write 20 short numbered facts about the moon.'; send(); void 0;")
                await wait_page(page, 'state.busy && state.session.pending?.requestId')
                await page.select_option('#sessions', b)
                await page.evaluate("$('outputLimit').value='custom'; $('maxTokens').value=64; policyChanged(); $('input').value='Remember the word BETA for this conversation. Answer briefly: what is two plus two?'; send(); void 0;")
                saw_queue = False; max_active = 0; samples = []
                async with httpx.AsyncClient(trust_env=False) as observer:
                    for _ in range(1800):
                        h = (await observer.get('http://127.0.0.1:8000/health')).json()
                        max_active = max(max_active, h.get('active_requests', 0))
                        if h.get('queued_requests'):
                            saw_queue = True
                            if len(samples) < 8: samples.append({k: h[k] for k in ('active_requests', 'queued_requests')})
                        if await page.evaluate('[...conversations.values()].every(c => !c.busy)'): break
                        await asyncio.sleep(.1)
                receipt['queue_samples'] = samples; receipt['max_native_active'] = max_active
                receipt['concurrent_final_state'] = await page.evaluate('[...conversations.values()].map(c=>({busy:c.busy,notice:c.notice,count:c.session.count,problem:c.session.problem}))')
                check('simultaneous Browser conversations queue', saw_queue)
                check('one native-active execution lane', max_active == 1)
                records = await page.evaluate('[...conversations.values()].map(c=>({id:c.session.id,count:c.session.count,messages:c.session.messages,problem:c.session.problem}))')
                check('independent canonical histories', all(r['count'] == 1 and not r['problem'] and len(r['messages']) == 2 for r in records))
                receipt['conversations'] = records
                async with httpx.AsyncClient(trust_env=False) as client:
                    outcomes = [(await client.get(url+'/api/session/'+sid)).json() for sid in (a,b)]
                    check('replay/repack zero', all(r['diagnostics']['m8']['total_prompt_replay_count'] == 0 and r['diagnostics']['m8']['total_full_cache_repack_count'] == 0 for r in outcomes))
                    receipt['outcomes'] = outcomes
                await page.select_option('#sessions', a)
                check('A-B-A retains A history', await page.evaluate('state.session.messages[0].content.includes("ALPHA") && !state.session.messages[0].content.includes("BETA")'))
                await page.evaluate("$('input').value='Which word did I ask you to remember? Reply with the word only.'; send(); void 0;")
                await wait_page(page, '!state.busy && state.session.count===2')
                check('A continuation after B uses retained pair', await page.evaluate('state.session.messages.at(-1).content.includes("ALPHA")'))
                # Stop remains conversation-scoped even when delivery is queued.
                await page.select_option('#sessions', b)
                await page.evaluate("$('maxTokens').value=512; policyChanged(); $('input').value='Write 500 numbered facts about oceans.'; send(); void 0;")
                await wait_page(page, 'state.busy && state.session.pending?.requestId && state.live?.content.length > 80')
                await page.click('#stop')
                await wait_page(page, '!state.busy')
                check('cancellation reconciles without resend', await page.evaluate('!state.session.pending && !state.session.problem'))
                check('no automatic archive / restore UI', await page.locator('#saveSession, #restoreSession').count() == 0)
                # Actual application tool/settlement loop, no external network effect.
                from tools.qualify_pdf_documents import text_pdf
                fixture = args.output/'source.pdf'; fixture.write_bytes(text_pdf(1))
                await page.click('#newSession')
                await page.set_input_files('#pdfs', str(fixture))
                await wait_page(page, '!state.busy && state.session.documents?.[0]?.state === "retained"')
                await page.evaluate("$('outputLimit').value='custom'; $('maxTokens').value=256; policyChanged(); $('input').value='Use document_pdf to read page 1 of the attached retained PDF. Report the inventory and cite the page.'; send(); void 0;")
                await wait_page(page, '!state.busy && state.session.count >= 2')
                check('real retained PDF tool loop settles', await page.evaluate('!state.session.problem && state.session.messages.some(m=>m.role==="tool") && state.session.messages.at(-1).content.includes("17")'))
                receipt['tool_conversation'] = await page.evaluate('({messages:state.session.messages,count:state.session.count,problem:state.session.problem})')
                # Keep the live tab alive while verifying the lost beacon lease.
                async with httpx.AsyncClient(trust_env=False) as client:
                    # Do not GET the orphan while waiting: GET itself renews the
                    # runtime lease. The live tab keeps renewing its own chats.
                    await asyncio.sleep(max(0, 635-(time.monotonic()-orphan_start)))
                    response = await client.get('http://127.0.0.1:8000/v1/sessions/'+orphan_id)
                    check('lost browser cleanup retires orphan', response.status_code == 404)
                    final = [(await client.get(url+'/api/session/'+sid)).json() for sid in await page.evaluate('nativeIds()')]
                    check('all final paired outcomes aligned with replay/repack zero', all(r['diagnostics']['m8']['all_cache_offsets_equal_frontier'] and r['diagnostics']['m8']['total_prompt_replay_count'] == 0 and r['diagnostics']['m8']['total_full_cache_repack_count'] == 0 for r in final))
                    receipt['final_outcomes'] = final
            check('no browser JavaScript errors', not errors)
            receipt['browser_errors'] = errors
            await context.close(); await browser.close()
        await ui.key('w'); await wait_service(ui, CHAT_PORT, False); check('W Stop Web Chat')
        await ui.key('r'); await wait_service(ui, RUNTIME_PORT, False); check('R Stop Runtime')
        await ui.close(); check('Q Quit TUI')
    finally:
        if ui.child.poll() is None:
            await ui.close()
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['ui', 'full'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true', help='resume an owned interrupted acceptance, never take over a user workload')
    asyncio.run(run(parser.parse_args()))
