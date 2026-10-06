#!/usr/bin/env python3
"""Real dynamic-budget Chrome/standard-OFF checks; isolated context/session only."""
import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qualify_web_browser import Browser


async def run(args):
    root = Path(__file__).resolve().parents[1]
    files = [root / p for p in ('ds41f_mlx/web.py', 'ds41f_mlx/web_tools.py', 'ds41f_mlx/web_budget.py', 'ds41f_mlx/serving/capacity.py', 'ds41f_mlx/serving/server.py', 'ds41f_mlx/serving/deepseek_recipe_backend.py', 'ds41f_mlx/serving/stateful_stream.py')] + sorted((root / 'ds41f_mlx/web_static').glob('*'))
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    receipt = {'status': 'RUNNING', 'url': args.web, 'sources': sources, 'steps': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def write(): args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    write()
    control, first, second = Browser(), Browser(), Browser()
    async with httpx.AsyncClient(timeout=30) as client:
        version = (await client.get(args.chrome + '/json/version')).json()
        await control.connect(version['webSocketDebuggerUrl'])
        context = await control.command('Target.createBrowserContext')
        async def page(browser):
            target = await control.command('Target.createTarget', {'url': args.web, 'browserContextId': context['browserContextId']})
            pages = (await client.get(args.chrome + '/json')).json()
            await browser.connect(next(p['webSocketDebuggerUrl'] for p in pages if p['id'] == target['targetId']))
            await browser.wait('typeof state!=="undefined" && state.session && !state.busy')
        async def check(name, expression, browser=first):
            value = await browser.evaluate(expression)
            receipt['steps'][name] = value; write(); assert value, (name, value)
        async def snapshot(name):
            await first.wait('!state.busy')
            value = await first.evaluate('(async()=>{const r=await api(path(state.session.id));return {id:state.session.id,count:state.session.count,problem:state.session.problem,pending:state.session.pending,replay:r.diagnostics.m8.total_prompt_replay_count,repack:r.diagnostics.m8.total_full_cache_repack_count,state:r.state,cancelled:r.last_turn?.cancelled};})()')
            receipt['steps'][name] = value; write()
            assert not value.get('problem') and not value.get('pending') and value['replay'] == value['repack'] == 0, value
        try:
            await page(first)
            await check('runtime_auto_default', "$('outputLimit').value==='auto' && settings().max_tokens==='auto' && $('reasoning').value==='max' && !$('toolCeiling')")
            await check('nonbinding_actual_recipe_budget', "(async()=>{const s=state.session;const b=await api('/api/budget','POST',{session_id:s.id,expected_count:s.count,request:{...generationRequest(),messages:[{role:'user',content:'Actual budget input.'}]}});const r=await api(path(s.id));window.__preview=b;return b.qualified_total_tokens===1048576 && b.prompt_tokens>0 && b.max_tokens===1048576-b.prompt_tokens && !b.binding && r.request_count===0 && r.state==='empty';})()")
            await check('secure_random_uuid', "(()=>{const values=Array.from({length:1000},()=>ChatPlatform.uuid());return new Set(values).size===1000 && values.every(v=>/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(v));})()")
            await page(second)
            await first.evaluate("window.__held=ChatPlatform.withLock('probe',async()=>{window.__entered=true;await new Promise(resolve=>window.__release=resolve);}); void 0;")
            await first.wait('window.__entered')
            await check('second_tab_blocked', "ChatPlatform.withLock('probe',()=>{window.__bad=true;}).then(()=>false,e=>e.message.includes('Another browser tab'))", second)
            await first.evaluate('window.__release(); void 0;')
            await check('release_allows_next', "ChatPlatform.withLock('probe',()=>true)", second)
            await check('exception_releases_lock', "ChatPlatform.withLock('probe',()=>{throw Error('probe');}).then(()=>false,e=>e.message==='probe')")
            await check('after_exception', "ChatPlatform.withLock('probe',()=>true)")
            await first.evaluate("window.__held=ChatPlatform.withLock('probe',async()=>{window.__entered2=true;await new Promise(()=>{});}); void 0;")
            await first.wait('window.__entered2')
            await first.reload(); await first.wait('state.session && !state.busy')
            await check('reload_releases_transaction', "ChatPlatform.withLock('probe',()=>true)", second)
            await first.evaluate("$('outputLimit').value='auto';$('input').value='Reply with a fenced Python code block containing print(42), no explanation.';send()")
            await snapshot('real_generation')
            await check('rendered_code', "!!document.querySelector('.codeblock code')")
            await check('actual_auto_admission', "state.session.capacity.binding && state.session.capacity.output_mode==='auto' && state.session.capacity.max_tokens===1048576-state.session.capacity.prompt_tokens")
            sid = await first.evaluate('state.session.id')
            await first.reload(); await first.wait('state.session && !state.busy')
            await check('reload_same_conversation', 'state.session.id===' + json.dumps(sid) + ' && state.session.count===1')
            await first.evaluate("$('maxTokens').value=2048;$('input').value='Write 200 numbered detailed programming tips.';window.__send=send();void 0;")
            await first.wait('state.busy && state.session.pending?.requestId && state.live?.content.length>60')
            await first.evaluate("$('stop').click(); void 0;")
            await snapshot('safe_stop')
            await check('user_stop', 'state.session.interrupted')
            await first.evaluate("window.confirm=()=>true;$('settingsButton').click();$('saveSession').onclick()")
            old = await first.evaluate('state.session.id')
            await first.evaluate("$('saves').value=" + json.dumps(old) + ";$('restoreSession').onclick()")
            await snapshot('fresh_restore')
            await check('fresh_uuid_restore', 'state.session.id!==' + json.dumps(old) + ' && state.session.count===0 && !state.session.pendingRestore')
            await first.evaluate("window.__toolReceipts=[];const originalAPI=api;api=async(...args)=>{const value=await originalAPI(...args);if(args[0]==='/api/tools')window.__toolReceipts.push(value);return value;};void 0;")
            await first.evaluate("$('input').value='Use fetch_url exactly once on https://en.wikipedia.org/wiki/Python_(programming_language) and then give one short sentence summarizing the returned excerpt.';send()")
            await snapshot('real_fetch_and_consumption')
            await check('fetch_over_old_4k_excerpt', "state.session.messages.some(m=>m.role==='tool' && JSON.parse(m.content).excerpt?.length>4000)")
            await check('search_capability_schema', "state.session.protocol.tools.find(t=>t.function.name==='web_search').function.parameters.properties.max_results.maximum===20")
            await check('tool_effect_observation_only', "(async()=>{const s=state.session;const r=await api('/api/tools/result?session_id='+s.id+'&request_count='+s.effects.count);return r.observation_only && r.result===null && window.__toolReceipts.length===1 && window.__toolReceipts[0].capacity.prompt_tokens>0 && !window.__toolReceipts[0].capacity.binding;})()")
            await check('custom_boundary_no_generation', "(async()=>{const s=state.session,before=await api(path(s.id));$('outputLimit').value='custom';$('maxTokens').value=1048576;let rejected=false;try{await generate();}catch(e){rejected=e.message.startsWith('No generation submitted:');}const after=await api(path(s.id));return rejected && !!s.capacityStop && !s.pending && after.request_count===before.request_count && after.last_turn.request_id===before.last_turn.request_id;})()")
            await check('stop_after_preview_no_admission', "(async()=>{const s=state.session,before=await api(path(s.id));const savedAPI=api;api=async(...args)=>{const value=await savedAPI(...args);if(args[0]==='/api/budget')state.stop=true;return value;};$('outputLimit').value='auto';$('input').value='This draft is deliberately stopped before admission.';try{await send();}finally{api=savedAPI;}const after=await api(path(s.id));return !state.session.pending && state.session.interrupted && after.request_count===before.request_count && after.last_turn.request_id===before.last_turn.request_id && $('notice').textContent.includes('Stopped before generation admission');})()")
            assert sources == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            receipt['status'] = 'PASS'; write()
        except Exception as error:
            receipt['status'] = 'FAIL'; receipt['error'] = str(error); write(); raise
        finally:
            try:
                sid = await first.evaluate('state.session?.id')
                receipt['qualification_session'] = sid; write()
                if sid: await first.evaluate("api(path(" + json.dumps(sid) + "),'DELETE')")
            finally:
                await control.command('Target.disposeBrowserContext', context)
                for browser in (first, second, control):
                    if browser.ws: await browser.ws.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web', default='http://127.0.0.1:8082/', help='isolated Web/runtime pair')
    parser.add_argument('--chrome', default='http://127.0.0.1:9222')
    parser.add_argument('--output', type=Path, default=Path('artifacts/dynamic-budget/browser.json'))
    asyncio.run(run(parser.parse_args()))
