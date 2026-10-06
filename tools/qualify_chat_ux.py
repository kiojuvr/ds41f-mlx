#!/usr/bin/env python3
"""Isolated real Chrome / existing runtime Chat UX collector. Never restarts runtime.
A private browser context keeps the user's selected conversation/storage untouched.
No generation/effect retry on failure. Receipts are incremental, never inferred PASS.
"""
import argparse
import base64
import hashlib
import asyncio
import json
import time
from pathlib import Path
import sys
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qualify_web_browser import Browser

async def run(args):
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'ds41f_mlx/web_static').glob('*'))}
    receipt = {'status': 'RUNNING', 'steps': {}, 'started': time.time(), 'sources': sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def write(): args.output.write_text(json.dumps(receipt, indent=2))
    write()
    browser = Browser(); control = Browser()
    async with httpx.AsyncClient(timeout=30) as client:
        version = (await client.get(args.chrome + '/json/version')).json()
        await control.connect(version['webSocketDebuggerUrl'])
        context = await control.command('Target.createBrowserContext')
        target = await control.command('Target.createTarget', {'url': args.web, 'browserContextId': context['browserContextId']})
        pages = (await client.get(args.chrome + '/json')).json()
        page = next(p for p in pages if p['id'] == target['targetId'])
        await browser.connect(page['webSocketDebuggerUrl'])
        await browser.command('Emulation.setDeviceMetricsOverride', {'width': 1100, 'height': 800, 'deviceScaleFactor': 1, 'mobile': False})
        async def check(name, expression):
            result = await browser.evaluate(expression)
            receipt['steps'][name] = result; write()
            assert result, (name, result)
        async def snapshot(name):
            await browser.wait('!state.busy')
            result = await browser.evaluate("""(async()=>{const r=await api(path(state.session.id)); return {id:state.session.id,count:state.session.count,problem:state.session.problem,pending:state.session.pending,finish:state.session.finishReason,status:$('status').textContent,termination:document.querySelector('.termination')?.textContent,replay:r.diagnostics?.m8?.total_prompt_replay_count,repack:r.diagnostics?.m8?.total_full_cache_repack_count,frontier:r.diagnostics?.m8?.frontier,imageEncoded:r.diagnostics?.last_image_encoded_count,cancelled:r.last_turn?.cancelled,requestId:r.last_turn?.request_id,applicationId:r.last_turn?.application_request_id,state:r.state,tools:state.session.messages.filter(m=>m.role==='tool').length};})()""")
            receipt['steps'][name] = result; write()
            assert not result.get('problem') and not result.get('pending') and result['replay'] == result['repack'] == 0, result
        async def send(prompt, tokens=256):
            await browser.evaluate(f"$('outputLimit').value='custom'; $('maxTokens').value={tokens}; $('input').value={json.dumps(prompt)}; send()")
            await browser.wait('!state.busy')
        try:
            await browser.wait('typeof state !== "undefined" && state.session && !state.busy && state.tools.length')
            await browser.evaluate("window.__confirmations=[]; window.confirm=message=>{window.__confirmations.push(message); return true;}")
            await browser.evaluate("window.__bulkReads=[]; const original=ChatStore.all; ChatStore.all=name=>{window.__bulkReads.push(name); return original(name);}")
            await check('quiet_header', "$('status').textContent==='Ready' && !$('settingsDialog').open && $('sessions').hidden && $('send').textContent==='Send'")
            await check('ime_event_wiring_before_generation', "(()=>{$('input').dispatchEvent(new CompositionEvent('compositionstart',{bubbles:true})); const active=composing; const e=new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true}); $('input').dispatchEvent(e); $('input').dispatchEvent(new CompositionEvent('compositionend',{bubbles:true})); return active && !composing && !e.defaultPrevented && state.session.count===0 && !state.busy;})()")
            await check('controls_mapping', "(()=>{$('thinking').value='on'; $('reasoning').value='low'; $('reasoning').onchange(); const a=settings(); $('reasoning').value='max'; $('reasoning').onchange(); const b=settings(); $('thinking').value='off'; $('thinking').onchange(); $('temperature').value='0.3'; $('topP').value='0.8'; $('temperature').onchange(); $('topP').onchange(); return a.reasoning_effort==='low' && b.reasoning_effort==='max' && settings().reasoning_effort==='none' && settings().temperature===0.3 && settings().top_p===0.8;})()")
            await send('Reply briefly in Markdown: a heading, bold and italic text, a two-item list, and a fenced Python code block containing print(42).', 256)
            await snapshot('normal_markdown')
            await check('markdown_code_copy', "!!document.querySelector('.assistant h1,.assistant h2,.assistant h3') && !!document.querySelector('.codeblock code') && [...document.querySelectorAll('.codeHeader button')].some(b=>b.textContent==='Copy')")
            await check('no_bulk_history_reads', "window.__bulkReads.length===0 && $('sessions').options.length===1")
            await check('protocol_locked', "$('thinking').disabled && $('reasoning').disabled && state.session.protocol.reasoning==='none'")
            await browser.command('Browser.grantPermissions', {'origin': args.web.rstrip('/'), 'permissions': ['clipboardReadWrite', 'clipboardSanitizedWrite'], 'browserContextId': context['browserContextId']})
            await browser.evaluate("document.querySelector('.codeHeader button').click()")
            await browser.wait("document.querySelector('.codeHeader button').textContent==='Copied'")
            await check('code_clipboard', "navigator.clipboard.readText().then(x=>x.includes('print'))")
            await send('Write a long numbered list of 150 distinct practical programming tips, one sentence per tip.', 512)
            await snapshot('output_limit')
            await check('inline_output_continue', "state.session.finishReason==='length' && [...document.querySelectorAll('.termination button')].some(b=>b.textContent==='Continue response')")
            await browser.evaluate("$('maxTokens').value=256; continueResponse()")
            await snapshot('output_continuation')
            # Actual stream while reading above latest; wheel dispatch represents user input.
            await browser.evaluate("$('maxTokens').value=2048; $('input').value='Write 250 numbered detailed tips for maintaining Python projects.'; window.__send=send(); void 0;")
            await browser.wait("state.busy && state.live?.content.length > 300")
            await browser.evaluate("$('messages').scrollTop=0")
            await asyncio.sleep(.3)
            top = await browser.evaluate("$('messages').scrollTop")
            await asyncio.sleep(1)
            await check('scroll_unfollow', f"!follow && !$('latest').hidden && Math.abs($('messages').scrollTop-{top})<3")
            await browser.evaluate("$('latest').click()")
            await check('latest_refollow', "follow && $('latest').hidden")
            await browser.evaluate("$('stop').click()")
            await browser.wait('!state.busy')
            await snapshot('stop_settlement')
            await check('stop_label', "state.session.interrupted && $('messages').textContent.includes('Stopped by user')")
            await check('stop_not_output_limit', "![...[...document.querySelectorAll('.assistant')].at(-1).querySelectorAll('.small')].some(n=>n.textContent==='Output limit reached.')")
            await browser.reload(); await browser.wait('!state.busy')
            await snapshot('reload_reconnect')
            await send('Without tools, write a detailed tutorial on Python generators covering lazy evaluation, yield, send, throw, close, itertools, memory tradeoffs, and testing. Include several code examples. Aim for about 2000 words.', 2048)
            await snapshot('long_output_2k')
            await check('long_output_follow_and_markdown', "state.session.messages.at(-1).content.length>4000 && follow && !!document.querySelector('.assistant .codeblock') && $('messages').scrollHeight-$('messages').clientHeight-$('messages').scrollTop<70")
            await check('safe_markdown', "(()=>{const p=document.createElement('div'); ChatRender.prose(p,'<img src=x onerror=alert(1)> [bad](javascript:alert(1))\\n\\n| a | b |\\n| --- | --- |\\n| 1 | 2 |'); return !p.querySelector('img,script,a') && !!p.querySelector('table');})()")
            await browser.evaluate("$('toolsMode').value='auto'; $('toolCeiling').value=5; $('maxTokens').value=512; $('input').value='Use fetch_url to retrieve https://example.com/?step=1 then step=2 then step=3 then step=4 then step=5 then step=6 then step=7. IMPORTANT: make exactly ONE tool call per assistant turn, sequentially, never batch calls. After each result request the next step. Do not answer until all seven fetches have finished.'; send()")
            await snapshot('tool_workflow')
            await check('tool_ceiling', "state.session.toolPause==='ceiling' && state.session.toolRounds===5 && document.querySelector('.termination').textContent.includes('Continue tools')")
            await check('inline_tool_stop_present', "[...document.querySelectorAll('.termination button')].some(b=>b.textContent==='Stop')")
            await browser.evaluate('pauseTools()')
            await check('paused_tools_stop', "state.session.toolPause==='stopped' && document.querySelector('.termination').textContent.includes('Pending tools have not been run')")
            await browser.evaluate('continueTools()'); await snapshot('inline_tool_continue')
            await check('tool_disclosure_unfollow', "(()=>{const summary=document.querySelector('.toolcall summary'); summary.click(); return !follow && !$('latest').hidden;})()")
            await send('Search the web for Python official documentation and fetch one result using fetch_url. Summarize briefly with a source link.', 512)
            await snapshot('search_fetch')
            await check('search_sources', "state.session.messages.some(m=>m.role==='assistant'&&m.tool_calls?.some(c=>c.function.name==='web_search')) && !!document.querySelector('.toolcall a')")
            await send('Use fetch_url to retrieve https://httpbin.org/status/403 exactly once. Do not retry or use another URL.', 256)
            await snapshot('tool_failure')
            await check('inline_tool_failure_no_retry', "state.session.toolPause==='error' && document.querySelector('.termination').textContent.includes('Tool failed') && !document.querySelector('.termination').textContent.includes('Retry')")
            await browser.evaluate('continueResponse()'); await snapshot('tool_failure_continue')
            # Actual network loss during an admitted stream; no automatic generation retry.
            await browser.evaluate("$('maxTokens').value=512; $('input').value='Write a long explanation of database indexes.'; window.__send=send(); void 0;")
            await browser.wait('state.live?.content.length>30 && state.session.pending?.requestId')
            await browser.command('Network.enable')
            await browser.command('Network.emulateNetworkConditions', {'offline': True, 'latency': 0, 'downloadThroughput': 0, 'uploadThroughput': 0})
            await browser.evaluate('state.controller.abort(); void 0;'); await browser.wait('!state.busy')
            await check('uncertainty_actions', "!!state.session.problem && !!state.session.pending && document.querySelector('.termination').textContent.includes('Reconcile') && !document.querySelector('.termination').textContent.includes('Continue response')")
            nonce = await browser.evaluate('state.session.pending.nonce')
            await browser.command('Network.emulateNetworkConditions', {'offline': False, 'latency': 0, 'downloadThroughput': -1, 'uploadThroughput': -1})
            await browser.evaluate("$('reconnect').onclick()")
            await snapshot('network_reconcile')
            await check('network_not_user_stop', "(()=>{const index=state.session.messages.findLastIndex(m=>m.role==='assistant'); return state.session.interruptionKinds[index]==='interrupted' && [...[...document.querySelectorAll('.assistant')].at(-1).querySelectorAll('.small')].some(n=>n.textContent.startsWith('Response interrupted.'));})()")
            await check('same_request_reconciled', f"api(path(state.session.id)).then(r=>r.last_turn.application_request_id==={json.dumps(nonce)})")
            # Reload itself disconnects the active browser stream; startup observes, never sends.
            await browser.evaluate("$('maxTokens').value=512; $('input').value='Write another long detailed explanation of database indexes.'; window.__send=send(); void 0;")
            await browser.wait('state.live?.content.length>30 && state.session.pending?.requestId')
            await browser.reload(); await browser.wait('!state.busy'); await snapshot('active_reload_reconcile')
            await browser.evaluate("window.__originalApi=api; api=async(...args)=>{const result=await window.__originalApi(...args); if(args[0]==='/api/tools')throw new Error('QUALIFICATION: tool result response lost after actual effect'); return result;}")
            await send('Use fetch_url exactly once to retrieve https://example.com/?observation=chat-ux and summarize briefly.', 256)
            await check('reserved_effect_loss', "state.session.effects?.state==='reserved' && document.querySelector('.termination').textContent.includes('will not be retried')")
            before_tools = await browser.evaluate("state.session.messages.filter(m=>m.role==='tool').length")
            await browser.reload(); await browser.wait('!state.busy'); await snapshot('effect_observation_only')
            await check('effect_result_recovered_once', f"state.session.effects.state==='completed' && state.session.toolAwaitingResponse && state.session.messages.filter(m=>m.role==='tool').length==={before_tools + 1} && document.querySelector('.termination').textContent.includes('Continue tools')")
            # Persist a received tool result before its model observation, then restore.
            # This must not lose the inline workflow or execute the effect again.
            await browser.evaluate("window.confirm=()=>true; $('settingsButton').click(); $('saveSession').onclick()")
            await check('saved_tool_workflow', "ChatStore.get('saves',state.session.id).then(x=>x.toolAwaitingResponse===true)")
            await browser.evaluate("$('saves').value=state.session.id; $('restoreSession').onclick()")
            await snapshot('restored_pending_tool_response')
            await check('restored_tool_workflow', "state.session.toolAwaitingResponse && !pendingCalls().length && document.querySelector('.termination').textContent.includes('Continue tools')")
            await browser.evaluate("$('settingsClose').click(); continueTools()")
            await snapshot('observed_tool_continuation')
            prior_tools = await browser.evaluate("state.session.messages.filter(m=>m.role==='tool').length")
            await browser.evaluate("$('toolsMode').value='ask'; $('toolsMode').onchange()")
            await send('Use fetch_url exactly once to retrieve https://example.com/?approval=chat-ux and summarize briefly.', 256)
            await check('ask_before_effect', f"pendingCalls().length===1 && state.session.messages.filter(m=>m.role==='tool').length==={prior_tools} && document.querySelector('.termination').textContent.includes('Allow requested tools')")
            await browser.evaluate('continueTools()'); await snapshot('approved_tool')
            await check('approved_effect_once', f"state.session.messages.filter(m=>m.role==='tool').length==={prior_tools + 1}")
            await browser.evaluate("window.__confirmations=[]; window.confirm=message=>{window.__confirmations.push(message); return true;}; $('newSession').onclick()")
            await check('new_chat_close_warning', "window.__confirmations.some(x=>x.includes('current native conversation will be closed')) && state.session.count===0")
            await browser.evaluate("$('thinking').value='on'; $('reasoning').value='low'; $('reasoning').onchange()")
            await send('What is 17 times 23? Answer briefly.', 128); await snapshot('real_thinking_low')
            await check('real_thinking_prefix_lock', "state.session.protocol.reasoning==='low' && $('thinking').disabled && $('reasoning').disabled")
            await check('public_reasoning_only', "(()=>{const value=state.session.messages.at(-1).reasoning_content; const details=document.querySelector('.reasoning'); return value ? !!details && !details.open : !details;})()")
            await check('disclosure_preservation', "(()=>{const old=state.session.messages, previousLive=state.live, previousFollow=follow; try { follow=false; rendered.clear(); $('messages').replaceChildren(); liveNode=null; const assistant={role:'assistant',content:'UI-only display fixture',tool_calls:[{id:'display_fixture',function:{name:'web_search',arguments:'{\"query\":\"display-only\"}'}}]}; state.session.messages=[{role:'user',content:'Display-only fixture'},assistant,{role:'tool',tool_call_id:'display_fixture',content:'{\"results\":[]}'}]; render(); document.querySelector('.toolcall').open=true; assistant.reasoning_content='Published-field rendering fixture, not a model trace.'; render(); const typed=document.querySelector('.toolcall').open && !document.querySelector('.reasoning').open; state.live={role:'assistant',content:'Live display fixture',reasoning_content:'Public-field live display fixture.'}; render(); liveNode.querySelector('.reasoning').open=true; state.session.messages.push(state.live); state.live=null; render(); return typed && document.querySelector('.reasoning').open && document.querySelector('.toolcall').open; } finally {state.session.messages=old; state.live=previousLive; liveNode=null; rendered.clear(); $('messages').replaceChildren(); follow=previousFollow; render();}})()")
            await browser.evaluate("$('settingsButton').click(); $('saveSession').onclick()")
            low_saved_id = await browser.evaluate('state.session.id')
            await browser.evaluate("$('settingsClose').click(); $('newSession').onclick(); void 0;"); await browser.wait('!state.busy && state.session.count===0')
            await browser.evaluate("$('thinking').value='off'; $('thinking').onchange(); $('toolsMode').value='off'; $('toolsMode').onchange()")
            from ds41f_mlx.config import load_runtime_config
            image_dir = load_runtime_config().checkpoint_path/'inference/examples/images'
            dom = await browser.command('DOM.getDocument')
            file_input = await browser.command('DOM.querySelector', {'nodeId': dom['root']['nodeId'], 'selector': '#images'})
            await browser.command('DOM.setFileInputFiles', {'nodeId': file_input['nodeId'], 'files': [str(image_dir/'corn.jpeg')]})
            await browser.wait('state.attachments.length===1')
            await check('image_preview', "!!document.querySelector('#attachments img')")
            await browser.evaluate("document.querySelector('#attachments button').click()")
            await check('preview_remove', "state.attachments.length===0 && !document.querySelector('#attachments img')")
            corn_data = base64.b64encode((image_dir/'corn.jpeg').read_bytes()).decode()
            await browser.evaluate(f"(()=>{{const bytes=Uint8Array.from(atob({json.dumps(corn_data)}),x=>x.charCodeAt(0)); const file=new File([bytes],'dropped-corn.jpeg',{{type:'image/jpeg'}}); const transfer=new DataTransfer(); transfer.items.add(file); $('composer').dispatchEvent(new DragEvent('drop',{{dataTransfer:transfer,bubbles:true,cancelable:true}}));}})()")
            await browser.wait('state.attachments.length===1')
            await send('Identify this food briefly.', 64); await snapshot('vision_picker')
            # Exercise the browser paste handler with exact original JPEG bytes, no canvas conversion.
            data = base64.b64encode((image_dir/'carrots.jpeg').read_bytes()).decode()
            await browser.evaluate(f"(()=>{{const bytes=Uint8Array.from(atob({json.dumps(data)}),x=>x.charCodeAt(0)); const file=new File([bytes],'pasted-carrots.jpeg',{{type:'image/jpeg'}}); const transfer=new DataTransfer(); transfer.items.add(file); $('input').dispatchEvent(new ClipboardEvent('paste',{{clipboardData:transfer,bubbles:true,cancelable:true}}));}})()")
            await browser.wait('state.attachments.length===1')
            expected = hashlib.sha256((image_dir/'carrots.jpeg').read_bytes()).hexdigest()
            await check('paste_original_bytes', f"(async()=>{{const bytes=Uint8Array.from(atob(state.attachments[0].url.split(',')[1]),x=>x.charCodeAt(0)); const hash=await crypto.subtle.digest('SHA-256',bytes); return [...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('')==={json.dumps(expected)};}})()")
            await send('Contrast the second food with the first briefly.', 64); await snapshot('vision_paste')
            await send('What color is the first food? Answer briefly.', 32); await snapshot('vision_history')
            await check('no_historical_image_reencode', "api(path(state.session.id)).then(r=>r.diagnostics.last_image_encoded_count===0)")
            await check('historical_image_display', "document.querySelectorAll('#messages img.attachment').length===2")
            await browser.evaluate("$('settingsButton').click(); $('saveSession').onclick()")
            await check('secondary_recovery_save', "$('settingsDialog').open && ChatStore.get('saves',state.session.id).then(x=>!!x?.artifact)")
            saved = await browser.evaluate("ChatStore.get('saves',state.session.id).then(x=>({id:x.id,frontier:x.frontier,images:x.messages.flatMap(m=>Array.isArray(m.content)?m.content:[]).filter(p=>p.type==='image_url').map(p=>p.image_url.url)}))")
            receipt['saved'] = {'frontier': saved['frontier'], 'image_sha256': [hashlib.sha256(base64.b64decode(url.split(',')[1])).hexdigest() for url in saved['images']]}; write()
            old_id = await browser.evaluate('state.session.id')
            await browser.evaluate("$('saves').value="+json.dumps(saved['id'])+"; $('restoreSession').onclick()")
            await snapshot('secondary_recovery_restore')
            await check('fresh_native_restore', f"state.session.id!=={json.dumps(old_id)} && state.session.frontier==={saved['frontier']} && !state.session.pendingRestore")
            await browser.evaluate("$('settingsClose').click()")
            await send('Which food is orange? Answer briefly.', 32); await snapshot('restored_vision_continuation')
            await check('no_restored_image_reencode', "api(path(state.session.id)).then(r=>r.diagnostics.last_image_encoded_count===0)")
            await check('bounded_dom', "(()=>{const old=state.session.messages; const f=follow; follow=false; rendered.clear(); state.session.messages=Array.from({length:2000},(_,i)=>({role:'user',content:'Display-only DOM bound test '+i})); visible=100; render(); const count=document.querySelectorAll('#messages .msg').length; state.session.messages=old; rendered.clear(); render(); follow=f; return count===100;})()")
            await check('composer_ime_and_resize', "(()=>{const old=$('input').value; $('input').value='one\\ntwo\\nthree'; resizeInput(); const height=$('input').offsetHeight; $('input').dispatchEvent(new CompositionEvent('compositionstart',{bubbles:true,data:'に'})); const e=new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true}); $('input').dispatchEvent(e); $('input').dispatchEvent(new CompositionEvent('compositionend',{bubbles:true,data:'日本'})); const enter=new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true}); $('input').value=''; $('input').dispatchEvent(enter); const shift=new KeyboardEvent('keydown',{key:'Enter',shiftKey:true,bubbles:true,cancelable:true}); $('input').dispatchEvent(shift); $('input').value='line\\n'.repeat(100); resizeInput(); const ok=!e.defaultPrevented && enter.defaultPrevented && !shift.defaultPrevented && height>60 && $('input').offsetHeight<=190 && $('input').scrollHeight>$('input').clientHeight; $('input').value=old; resizeInput(); return ok;})()")
            await browser.evaluate("$('settingsButton').click(); void 0;"); await browser.wait("$('saves').options.length>=3")
            await browser.evaluate("$('saves').value="+json.dumps(low_saved_id)+"; $('restoreSession').onclick()")
            await snapshot('restore_different_thinking_prefix')
            await check('restored_thinking_controls_match', "state.session.protocol.reasoning==='low' && $('thinking').value==='on' && $('reasoning').value==='low' && $('thinking').disabled && $('reasoning').disabled && $('thinkLabel').textContent==='Think 50'")
            await browser.evaluate("$('settingsClose').click(); $('latest').click(); void 0;")
            await asyncio.sleep(.2)
            screenshot = await browser.command('Page.captureScreenshot', {'format': 'png'})
            args.output.with_suffix('.png').write_bytes(base64.b64decode(screenshot['data']))
            current = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'ds41f_mlx/web_static').glob('*'))}
            assert current == sources, 'Sources changed during qualification: do not claim current-source PASS'
            receipt['status'] = 'PASS'; write()
        except Exception as error:
            receipt['status'] = 'FAIL'; receipt['error'] = str(error); write(); raise
        finally:
            # Preserve failed conversation evidence without closing the user's session.
            receipt['qualification_session'] = await browser.evaluate('state.session?.id')
            write()
            await browser.ws.close(); await control.ws.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chrome', default='http://127.0.0.1:9222')
    parser.add_argument('--web', default='http://127.0.0.1:8080/')
    parser.add_argument('--output', type=Path, default=Path('artifacts/chat-ux/browser.json'))
    asyncio.run(run(parser.parse_args()))
