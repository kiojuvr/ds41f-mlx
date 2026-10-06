#!/usr/bin/env python3
"""Real Chrome + Web surface operational collector (no frontend framework).

Launch Chrome headless with a dedicated profile / remote debugging port. --phase
initial saves an idle artifact; after a fresh ds41f server/model restart, --phase
restored reconnects the same browser history and restores via the UI's saved menu.
Uses an installed Chrome and Python websockets only as qualification tooling.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Browser:
    async def connect(self, url):
        self.ws = await websockets.connect(url, max_size=8*1024*1024)
        self.serial = 0
    async def command(self, method, params=None):
        self.serial += 1
        identity = self.serial
        await self.ws.send(json.dumps({'id':identity, 'method':method, 'params':params or {}}))
        while True:
            response = json.loads(await self.ws.recv())
            if response.get('id') == identity:
                if 'error' in response: raise RuntimeError(response['error'])
                return response.get('result', {})
    async def evaluate(self, expression):
        response = await self.command('Runtime.evaluate', {'expression':expression, 'awaitPromise':True, 'returnByValue':True})
        if 'exceptionDetails' in response: raise RuntimeError(response['exceptionDetails'])
        return response.get('result', {}).get('value')
    async def wait(self, expression, timeout=1800):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = await self.evaluate(expression)
                if result: return result
            except RuntimeError: pass
            await asyncio.sleep(.1)
        raise TimeoutError(expression)
    async def reload(self):
        # Page.reload returns before navigation; never act on the dying JS world.
        await self.evaluate('window.__qualificationReloadMarker=true')
        await self.command('Page.reload')
        await self.wait('window.__qualificationReloadMarker === undefined && typeof state !== "undefined" && state.session && state.tools.length')


async def exercise(args, receipt):
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    async with httpx.AsyncClient(timeout=1800) as client:
        pages = (await client.get(args.chrome + '/json')).json()
        page = next(p for p in pages if p['type'] == 'page')
        browser = Browser(); await browser.connect(page['webSocketDebuggerUrl'])
        async def record(label):
            await browser.wait('!state.busy')  # onclick may start exclusive without returning its promise
            # Do not retain full image bytes/another application transcript.
            summary = await browser.evaluate("""(async()=>{
              const s=state.session;
              const rec=s?await api('/api/session/'+encodeURIComponent(s.id)):null;
              const m8=rec?.diagnostics?.m8;
              return {id:s?.id,count:s?.count,problem:s?.problem,pending:s?.pending,interrupted:s?.interrupted,
                messageCount:s?.messages.length,roles:s?.messages.map(m=>m.role),notice:$('notice').textContent,
                images:(s?.messages||[]).flatMap(m=>Array.isArray(m.content)?m.content:[]).filter(p=>p.type==='image_url').length,
                toolMessages:(s?.messages||[]).filter(m=>m.role==='tool').map(m=>({id:m.tool_call_id,content:m.content})),
                state:rec?.state,frontier:m8?.frontier,offsets:m8?.cache_offsets_all,replay:m8?.total_prompt_replay_count,
                repack:m8?.total_full_cache_repack_count,imageEncoded:rec?.diagnostics?.last_image_encoded_count,
                cancelled:rec?.last_turn?.cancelled,generated:rec?.last_turn?.generated_tokens?.length,
                reconstruction:rec?.last_turn?.reconstruction,active:rec?.active_request_id,artifact:rec?.persisted_artifact};
            })()""")
            receipt.setdefault('steps', {})[label] = summary
            args.output.write_text(json.dumps(receipt, indent=2))
            assert summary['state'] == 'idle' and not summary['problem'] and not summary.get('pending'), summary
            assert summary['replay'] == summary['repack'] == 0
            assert len(summary['offsets']) == 40 and all(x == summary['frontier'] for x in summary['offsets'])
            return summary
        async def send(text):
            await browser.evaluate('$('+json.dumps('input')+').value='+json.dumps(text))
            await browser.evaluate('send()')
            summary = await browser.evaluate('({problem:state.session.problem,pending:state.session.pending,notice:$("notice").textContent})')
            assert not summary['problem'] and not summary.get('pending'), summary
        async def add_image(name):
            dom = await browser.command('DOM.getDocument')
            node = await browser.command('DOM.querySelector', {'nodeId':dom['root']['nodeId'],'selector':'#images'})
            await browser.command('DOM.setFileInputFiles', {'nodeId':node['nodeId'],'files':[str(cfg.checkpoint_path/'inference/examples/images'/name)]})
            await browser.wait('state.attachments.length === 1')
            assert await browser.evaluate('document.querySelectorAll("#attachments img").length') == 1
        try:
            await browser.command('Page.navigate', {'url': args.web})
            await browser.wait('typeof state !== "undefined" && state.tools.length > 0')
            if args.phase == 'initial':
                await browser.evaluate('$("newSession").onclick()')
                await browser.wait('state.session && !state.busy')
                await browser.evaluate('$("maxTokens").value="128"; $("toolsEnabled").checked=false; $("input").value="List numbers 1 to 100, one per line."; window.browserJob=send(); true')
                started = time.monotonic()
                live = await browser.wait('state.live?.content?.length > 10 && state.busy ? ({content:state.live.content,requestId:state.session.pending?.requestId}) : false')
                receipt['live_observation'] = {'elapsed':time.monotonic()-started, 'observed':live,
                    'runtime':(await client.get(args.web+'/api/session/'+(await browser.evaluate('state.session.id')))).json()['state']}
                assert receipt['live_observation']['runtime'] == 'busy'
                await browser.evaluate('window.browserJob')
                await record('streaming_text')
                await send('What was the first number? Answer in a short phrase.'); await record('multi_turn')
                await browser.evaluate('$("toolsEnabled").checked=true')
                await send('Use web_search to find the official Python documentation. Give one source link and a short description.')
                searched = await record('web_search')
                search_results = [json.loads(x['content']) for x in searched['toolMessages'] if 'results' in json.loads(x['content'])]
                assert any(any(r.get('url', '').startswith(('http://', 'https://')) for r in x.get('results', [])) for x in search_results), searched
                await send('Use fetch_url to retrieve https://docs.python.org/3/ and briefly state what page it is.')
                fetched = await record('fetch_url')
                assert any(json.loads(x['content']).get('url', '').startswith('https://docs.python.org/') and json.loads(x['content']).get('excerpt') for x in fetched['toolMessages']), fetched
                ids = [x['id'] for x in fetched['toolMessages']]; assert len(ids) == len(set(ids))
                await browser.evaluate('$("toolsEnabled").checked=false; $("maxTokens").value="64"')
                await add_image('corn.jpeg'); await send('Identify this food briefly.')
                vision = await record('vision'); assert vision['images'] == vision['imageEncoded'] == 1
                await send('What color is the first food? Answer briefly.')
                continuation = await record('image_text'); assert continuation['images'] == 1 and continuation['imageEncoded'] == 0
                await add_image('carrots.jpeg'); await send('Contrast this second food with the first briefly.')
                additional = await record('second_image'); assert additional['images'] == 2 and additional['imageEncoded'] == 1
                await browser.evaluate('$("maxTokens").value="512"; $("input").value="Describe both foods in 20 long paragraphs."; window.browserJob=send(); true')
                await browser.wait('state.live?.content?.length > 30 && state.session.pending?.requestId')
                await browser.evaluate('$("stop").onclick()'); await browser.evaluate('window.browserJob')
                stopped = await record('active_stop'); assert stopped['cancelled'] and stopped['generated'] < 512 and stopped['reconstruction']['representable']
                await browser.reload()
                await browser.wait('state.session.record?.state === "idle"')
                reload = await record('browser_reload'); assert reload['count'] == stopped['count'] and reload['images'] == 2
                await browser.evaluate('$("saveSession").onclick()')
                saved = [await browser.evaluate('ChatStore.get("saves", state.session.id)')]
                # Snapshot's image SHA256, not the raw payload, is evidence.
                receipt['saved'] = {k:v for k,v in saved[-1].items() if k != 'messages'}
                receipt['saved']['image_sha256'] = [hashlib.sha256(__import__('base64').b64decode(p['image_url']['url'].split(',')[1])).hexdigest()
                     for m in saved[-1]['messages'] if isinstance(m.get('content'), list) for p in m['content'] if p.get('type') == 'image_url']
                await record('persist')
            elif args.phase == 'restored':
                await browser.wait('state.session?.problem?.includes("absent")')
                receipt['restart_recovery_notice'] = await browser.evaluate('$("notice").textContent')
                previous_id = await browser.evaluate('state.session.id')
                initial = json.loads((ROOT/'artifacts/web-application/initial.json').read_text())
                await browser.evaluate('$("saves").value='+json.dumps(initial['saved']['id']))
                await browser.evaluate('window.restoreJob=$("restoreSession").onclick(); true')
                known_id = await browser.wait('state.session?.pendingRestore ? state.session.id : false')
                assert known_id != previous_id
                for _ in range(200):
                    response = await client.get(args.web + '/api/session/' + known_id)
                    if response.status_code == 200 and response.json()['state'] == 'busy': break
                    await asyncio.sleep(.1)
                else: raise AssertionError('restore did not reserve its known native ID')
                receipt['restore_journal_id'] = known_id
                await browser.reload()  # lose restore response, NOT the native authority
                await browser.wait('typeof state !== "undefined" && state.session?.pendingRestore && state.tools.length')
                for _ in range(1800):
                    response = await client.get(args.web + '/api/session/' + known_id)
                    if response.status_code == 200 and response.json()['state'] == 'idle': break
                    await asyncio.sleep(1)
                else: raise AssertionError('native restore did not settle')
                await browser.evaluate('$("reconnect").onclick()')
                await browser.wait('state.session && !state.session.problem && !state.session.pendingRestore && !state.busy', timeout=1800)
                assert await browser.evaluate('state.session.id') == known_id
                restored = await record('fresh_restore')
                assert restored['images'] == 2
                await send('Which of the two foods is orange? Answer briefly.')
                continuation = await record('restored_continuation')
                assert continuation['imageEncoded'] == 0 and continuation['images'] == 2
            elif args.phase == 'observed-restore':
                # Explicit observer continuation after a collector-only failure,
                # before its first generation. Never repeat restore I/O.
                prior = json.loads(args.resume_receipt.read_text())
                assert prior['error'] == "KeyError('pending')" and prior['sources'] == receipt['sources']
                await browser.wait('state.session?.record?.state === "idle" && !state.session.problem')
                assert await browser.evaluate('state.session.id') == prior['restore_journal_id']
                restored = await record('fresh_restore')
                assert restored['count'] == 0 and restored['frontier'] == prior['steps']['fresh_restore']['frontier'] and restored['images'] == 2
                receipt['restore_journal_id'] = prior['restore_journal_id']
                receipt['collector_resume_from'] = str(args.resume_receipt)
                await send('Which of the two foods is orange? Answer briefly.')
                continuation = await record('restored_continuation')
                assert continuation['imageEncoded'] == 0 and continuation['images'] == 2
            else:
                await browser.wait('state.session?.record?.state === "idle" && !state.session.problem')
                await browser.evaluate('$("toolsEnabled").checked=false; $("maxTokens").value="32"; $("preset").value="balanced"')
                receipt['soak_count_start'] = await browser.evaluate('state.session.count')
                for index in range(8):
                    await send(f'Without tools, name the color of the {"first" if index%2==0 else "second"} food in one word.')
                    sample = await record('soak_' + str(index))
                    assert sample['imageEncoded'] == 0
                await browser.evaluate('$("preset").value="precise"; $("maxTokens").value="512"; $("input").value="Describe both foods in 20 long paragraphs without tools."; window.browserJob=send(); true')
                await browser.wait('state.live?.content?.length > 15 && state.session.pending?.requestId')
                await browser.reload()  # actual browser/proxy socket disconnect
                await browser.wait('typeof state !== "undefined" && state.session && state.tools.length')
                await browser.evaluate('$("reconnect").onclick()')
                interrupted = await record('active_browser_reload')
                assert interrupted['cancelled'] and interrupted['generated'] < 512
                await browser.evaluate('$("toolsEnabled").checked=true; $("maxTokens").value="96"; window.originalApi=api; api=async(...args)=>{const result=await originalApi(...args);if(args[0]==="/api/tools")throw new Error("QUALIFICATION: result lost after effect");return result}')
                await send('Use fetch_url to read https://example.com and report its title briefly.')
                lost = await record('tool_response_loss')
                assert await browser.evaluate('state.session.effects.state') == 'reserved'
                await browser.reload()
                await browser.wait('state.session?.effects?.state === "completed" && state.session.record?.state === "idle"')
                recovered = await record('tool_result_observed_without_execution')
                assert recovered['count'] == lost['count'] and len(recovered['toolMessages']) == len(lost['toolMessages']) + 1
                await browser.evaluate('$("toolsEnabled").checked=false; $("maxTokens").value="16"')
                await send('Without tools, reply only OK.')
                await record('recovered_tool_continuation')
                receipt['dom_bound'] = await browser.evaluate('(()=>{const old=state.session.messages;state.session.messages=Array.from({length:2000},()=>({role:"user",content:"UI-only bounded rendering test"}));render();const count=document.querySelectorAll("#messages .msg").length;state.session.messages=old;render();return count})()')
                assert receipt['dom_bound'] == 100
                await record('soak_final')
        finally:
            await browser.ws.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['initial', 'restored', 'observed-restore', 'soak'], required=True)
    parser.add_argument('--resume-receipt', type=Path)
    parser.add_argument('--web', default='http://127.0.0.1:8080')
    parser.add_argument('--chrome', default='http://127.0.0.1:9222')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or ROOT/f'artifacts/web-application/{args.phase}.json'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = ['ds41f_mlx/web.py', 'ds41f_mlx/web_client.py', 'ds41f_mlx/web_tools.py', 'ds41f_mlx/web_static/app.js', 'ds41f_mlx/web_static/store.js', 'ds41f_mlx/web_static/render.js', 'ds41f_mlx/web_static/index.html', 'ds41f_mlx/web_static/style.css']
    receipt = {'status':'RUNNING','phase':args.phase,'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    args.output.write_text(json.dumps(receipt, indent=2))
    try:
        asyncio.run(exercise(args, receipt)); receipt['status'] = 'PASS'
    except BaseException as exc:
        receipt.update(status='FAIL',error=repr(exc)); raise
    finally: args.output.write_text(json.dumps(receipt, indent=2))

if __name__ == '__main__': main()
