"""Focused real Chrome + ordinary MTP application Vision acceptance (not qualification)."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import uvicorn
from playwright.async_api import async_playwright

from tools.accept_operator_console import wait_page
from tools.qualify_pdf_documents import text_pdf


async def browser_run(url, cfg, backend, output, receipt):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
        page = await browser.new_page()
        errors = []
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        await page.goto(url)
        await wait_page(page, 'state.session && state.runtime?.status === "ready"')
        assert not await page.locator('#images').is_disabled()
        status = await page.evaluate('api("/api/status")')
        assert status['runtime']['profile'] == 'mtp-serving-v1'
        assert status['documents']['vision_available'] and status['documents']['vision_profile'] == 'mtp-serving-v1'
        receipt['capabilities'] = status['runtime']['capabilities']
        async def send(text):
            await page.fill('#input', text)
            await page.evaluate('send()')
            await wait_page(page, '!state.busy', timeout=1800000)
            result = await page.evaluate('({problem:state.session.problem, messages:state.session.messages})')
            assert not result['problem'], result
            return result['messages']
        async def settings(maximum, tools):
            await page.click('#settingsButton')
            await page.select_option('#outputLimit', 'custom')
            await page.fill('#maxTokens', str(maximum))
            await page.click('#settingsClose')
            await page.click('#toolsLabel')
            await page.select_option('#toolsMode', tools)
            await page.click('#toolsLabel')
        await settings(128, 'off')
        text = await send('Say hello in one short sentence.')
        assert text[-1]['role'] == 'assistant' and text[-1]['content']
        receipt['text'] = text[-1]['content']
        image = cfg.checkpoint_path/'inference/examples/images/corn.jpeg'
        await page.set_input_files('#images', str(image))
        await wait_page(page, 'state.attachments.length === 1')
        assert await page.locator('#attachments img').count() == 1
        messages = await send('Identify the food in this image in one short sentence.')
        assert 'corn' in messages[-1]['content'].lower(), messages[-1]
        trace = backend.serving_traces[-1]
        assert trace['vision_encoded_count'] == 1 and sum(trace['mtp_stats']['depth_drafted']) > 0
        receipt['image'] = dict(sha256=hashlib.sha256(image.read_bytes()).hexdigest(), answer=messages[-1]['content'])
        receipt['image_trace'] = project(trace)
        # A new PDF conversation has no image history. The model's tool call must
        # cause one stateless Vision execution and only derived text in main chat.
        await page.click('#newSession')
        await settings(512, 'auto')
        fixture = output.parent/'page.pdf'; fixture.write_bytes(text_pdf(1))
        await page.set_input_files('#pdfs', str(fixture.resolve()))
        await wait_page(page, '!state.busy && state.session.documents?.[0]?.state === "retained"')
        start = len(backend.serving_traces)
        messages = await send('Use document_pdf action observe on page 1 of the attached PDF (not read). Then report its inventory number, citing the page. Do not retry.')
        tools = [m for m in messages if m['role'] == 'tool']
        assert tools and any('model-derived-non-authoritative' in m['content'] and 'completed' in m['content'] for m in tools), tools
        assert not any(p.get('type') == 'image_url' for m in messages if isinstance(m.get('content'), list) for p in m['content'])
        assert not any('data:image/' in m.get('content', '') for m in messages if isinstance(m.get('content'), str))
        traces = list(backend.serving_traces)[start:]
        vision = [t for t in traces if t.get('vision_encoded_count') == 1]
        assert len(vision) == 1 and sum(vision[0]['mtp_stats']['depth_drafted']) > 0
        receipt['pdf_trace'] = project(vision[0])
        receipt['pdf_tool_result'] = [m['content'] for m in tools]
        receipt['pdf_answer'] = messages[-1]['content']
        assert '17' in messages[-1]['content'], messages[-1]
        assert await page.evaluate('state.attachments.length') == 0
        assert not errors, errors
        await browser.close()


def project(trace):
    assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
    assert set(trace['target_offsets'] + trace['dspark_offsets']) == {trace['canonical_frontier']}
    return {k: trace[k] for k in ('vision_encoded_count', 'mtp_stats', 'target_offsets', 'dspark_offsets', 'canonical_frontier', 'prompt_replay', 'full_cache_repack')}


def run(output):
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    from ds41f_mlx.web import create_app as web_app
    cfg = config(port=18073, profile='mtp-serving-v1')
    identity = json.loads(subprocess.check_output([sys.executable, '-I', '-c',
        'import json; from ds41f_mlx.mtp_identity import config, inspect; print(json.dumps(inspect(config(profile="mtp-serving-v1"))))'], text=True))
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    # This probe has no separate Vision endpoint, no OFF process, no retry.
    os.environ.pop('DS41F_DOCUMENT_VISION_URL', None)
    os.environ['DS41F_WEB_ARTIFACT_DB'] = str(output.parent/'artifacts.db')
    runtime_url = f'http://{cfg.host}:{cfg.port}'
    runtime_app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
    async def load_model():
        await backend._call(backend.load)  # same eager startup as ordinary serve
    runtime_app.router.add_event_handler('startup', load_model)
    apps = [(runtime_app, cfg.port), (web_app(runtime_base_url=runtime_url), 18074)]
    servers = [uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning')) for app, port in apps]
    threads = [threading.Thread(target=server.run) for server in servers]
    root = Path(__file__).resolve().parents[1]
    sources = ['ds41f_mlx/' + name for name in (
        'web.py', 'web_documents.py', 'chat_capability.py', 'serving/server.py',
        'web_static/app.js', 'web_static/documents.js', 'web_static/policy.js')]
    receipt = dict(status='RUNNING', scope='current-source application delta; admitted installed assets/dependencies, not release requalification',
                   dependency_identity=identity['identity_sha256'],
                   source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})
    try:
        for thread in threads: thread.start()
        for server, thread in zip(servers, threads):
            while not server.started and thread.is_alive(): time.sleep(.1)
            assert server.started
        asyncio.run(browser_run('http://127.0.0.1:18074', cfg, backend, output, receipt))
        receipt['status'] = 'PASS'
    except BaseException as exc:
        receipt.update(status='FAIL', error=str(exc))
        raise
    finally:
        output.write_text(json.dumps(receipt, indent=2, default=str)+'\n')
        for server, thread in reversed(list(zip(servers, threads))):
            server.should_exit = True
            thread.join(timeout=1800)
        backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/web-mtp-vision/receipt.json'))
    run(parser.parse_args().output)
