"""Real Chrome PDF upload, page sources, model tools and retained reuse dogfood."""
import argparse
import asyncio
import json
from pathlib import Path
import httpx
from tools.qualify_web_browser import Browser
from tools.qualify_pdf_documents import text_pdf


async def run(args):
    directory = Path(args.output); directory.mkdir(parents=True, exist_ok=True)
    fixture = (directory / 'browser-source.pdf').resolve(); fixture.write_bytes(text_pdf(3))
    output = directory / 'browser.json'
    records = []
    async with httpx.AsyncClient() as http:
        version = (await http.get(args.chrome + '/json/version')).json()
        control = Browser(); await control.connect(version['webSocketDebuggerUrl'])
        context = await control.command('Target.createBrowserContext')
        target = await control.command('Target.createTarget', dict(url=args.app, browserContextId=context['browserContextId']))
        pages = (await http.get(args.chrome + '/json')).json()
        page = next(p for p in pages if p['id'] == target['targetId'])
        browser = Browser(); await browser.connect(page['webSocketDebuggerUrl'])
        try:
            await browser.wait('typeof state !== "undefined" && state.session && !state.busy && state.tools.length')
            await browser.evaluate("window.confirm=()=>true; $('outputLimit').value='custom'; $('maxTokens').value=700;")
            root = await browser.command('DOM.getDocument')
            node = await browser.command('DOM.querySelector', dict(nodeId=root['root']['nodeId'], selector='#pdfs'))
            await browser.command('DOM.setFileInputFiles', dict(nodeId=node['nodeId'], files=[str(fixture)]))
            await browser.wait("!state.busy && state.session.documents?.[0]?.state === 'retained'")
            initial = await browser.evaluate("({documents:state.session.documents, messageCount:state.session.messages.length, attachments:state.attachments.length})")
            records.append(dict(kind='upload', **initial))
            assert initial['documents'][0]['page_count'] == 3 and initial['messageCount'] == initial['attachments'] == 0
            await browser.evaluate("document.querySelector('#documents button').click(); $('documentPageNumber').value=3; $('openDocumentPage').click();")
            await browser.wait("!!document.querySelector('#documentPage img')")
            assert await browser.evaluate('state.attachments.length') == 0
            await browser.evaluate("$('documentClose').click(); $('input').value='Read PDF pages 1 and 3 using document_pdf, report inventories and cite pages.'; send()")
            await browser.wait('!state.busy')
            first = await browser.evaluate("({messages:state.session.messages,documents:state.session.documents,id:state.session.id,problem:state.session.problem})")
            assert not first['problem'] and first['documents'][0]['announced']
            assert '17' in first['messages'][-1]['content'] and '51' in first['messages'][-1]['content']
            records.append(dict(kind='model-read', **first))
            # Open the tool citation, not an arbitrary network/image URL.
            await browser.evaluate("[...document.querySelectorAll('.toolrow button')].find(b=>b.textContent==='Open source p. 3').click()")
            await browser.wait("$('documentDialog').open && !!document.querySelector('#documentPage img')")
            await browser.evaluate("[...document.querySelectorAll('#documentPage button')].find(b=>b.textContent.startsWith('Add this page')).click()")
            await browser.wait('state.attachments.length === 1')
            assert await browser.evaluate("state.session.messages.flatMap(m=>Array.isArray(m.content)?m.content:[]).filter(p=>p.type==='image_url').length") == 0
            await browser.evaluate("state.attachments=[]; preview(); $('documentClose').click()")
            await browser.reload(); await browser.wait('!state.busy && state.session.record?.state === "idle"')
            assert await browser.evaluate('state.session.id') == first['id']
            await browser.evaluate("$('input').value='Read page 2 of the same retained PDF; report inventory and cite the source. Do not reacquire.'; send()")
            await browser.wait('!state.busy')
            last = await browser.evaluate("(async()=>({documents:state.session.documents,messages:state.session.messages,record:await api(path(state.session.id)),problem:state.session.problem}))()")
            assert not last['problem'] and '34' in last['messages'][-1]['content']
            user = [m for m in last['messages'] if m['role'] == 'user']
            assert len(user) == 2 and 'Retained PDF:' not in user[-1]['content'], 'Duplicate catalogue announcement after exclusive journal reload'
            images = [p for m in last['messages'] if isinstance(m['content'], list) for p in m['content'] if p.get('type') == 'image_url']
            assert not images
            m8 = last['record']['diagnostics']['m8']
            assert m8['total_prompt_replay_count'] == m8['total_full_cache_repack_count'] == 0
            assert set(m8['cache_offsets_all']) == {m8['frontier']}
            source_state = await browser.evaluate('api("/api/documents/" + state.session.documents[0].artifact_id)')
            assert not any(p.get('raster_retained') for p in source_state['pages']), 'UI preview permanently cached page PNGs'
            records.append(dict(kind='reload-reuse', source_state=source_state, **last))
            print('PASS browser PDF workflow', first['id'], flush=True)
        finally:
            output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
            await browser.evaluate("state.session ? api(path(state.session.id),'DELETE') : null")
            await browser.ws.close()
            await control.command('Target.disposeBrowserContext', context)
            await control.ws.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', default='http://127.0.0.1:8090')
    parser.add_argument('--chrome', default='http://127.0.0.1:9223')
    parser.add_argument('--output', default='artifacts/pdf-documents')
    asyncio.run(run(parser.parse_args()))
