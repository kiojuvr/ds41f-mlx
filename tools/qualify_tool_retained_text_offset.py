"""Real browser/model retained HTML offset; no external URL acquisition."""
import asyncio,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.async_api import async_playwright
from tools.qualify_tool_binary_development import sources,wait
INPUT,OUT=map(Path,sys.argv[1:3])
async def main():
    r=dict(status='RUNNING_NOT_ACCEPTANCE',source_hashes=sources(),checks={},errors=[])
    def save():OUT.write_text(json.dumps(r,indent=2)+'\n')
    save()
    source=json.loads(INPUT.read_text())['checks']['restored_ordinary_continuation']
    original=json.loads(next(m['content'] for m in source['messages'] if m['role']=='tool'))
    aid=original['artifact_id']
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=await browser.new_page()
        try:
            await page.goto('http://127.0.0.1:8087/');await wait(page,'state.session!==null&&!state.busy')
            await page.evaluate("$('thinking').value='off';$('temperature').value='0';$('outputLimit').value='custom';$('maxTokens').value='512'")
            await page.locator('#input').fill(f'Use fetch_url with retained artifact_id {aid}, offset 2000, length 1000. Read the additional excerpt and summarize it briefly. Do NOT download its URL again.')
            await page.locator('#send').click();await wait(page,'state.session.messages.some(m=>m.role==="tool")',timeout=1800000);await wait(page,'!state.busy',timeout=1800000)
            s=await page.evaluate('state.session');assert not (s.get('problem') or s.get('capacityStop') or s.get('toolError'))
            requests=[c for m in s['messages'] for c in m.get('tool_calls',[])]
            assert requests and all(json.loads(c['function']['arguments']).get('artifact_id')==aid and not json.loads(c['function']['arguments']).get('url') for c in requests)
            value=json.loads(next(m['content'] for m in s['messages'] if m['role']=='tool'))
            assert value['artifact_id']==aid and value['sha256']==original['sha256'] and value['offset']==2000 and value['excerpt']
            assert s['messages'][-1]['role']=='assistant' and s['finishReason']=='stop'
            r['checks']['original_transfer_metadata']=original;r['checks']['retained_offset_model_answer']=s
            r['checks']['canonical']=await (await page.request.get('http://127.0.0.1:8087/api/session/'+s['id'])).json()
            r['status']='DEVELOPMENT_PROBE_FINISHED_NOT_ACCEPTANCE'
        except Exception as exc:r['status']='FAIL_NOT_ACCEPTANCE';r['errors'].append(str(exc))
        finally:r['source_stable']=r['source_hashes']==sources();save();await browser.close()
asyncio.run(main())
