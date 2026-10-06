"""Actual browser/model private-URL refusal, followed by explicit continuation."""
import asyncio,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.async_api import async_playwright
from tools.qualify_tool_binary_development import sources,wait
OUT=Path(sys.argv[1])
async def main():
    r=dict(status='RUNNING_NOT_ACCEPTANCE',source_hashes=sources(),checks={},errors=[])
    def save():OUT.write_text(json.dumps(r,indent=2)+'\n')
    save()
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=await browser.new_page()
        try:
            await page.goto('http://127.0.0.1:8087/');await wait(page,'state.session!==null&&!state.busy')
            await page.evaluate("$('thinking').value='off';$('temperature').value='0';$('outputLimit').value='custom';$('maxTokens').value='256'")
            await page.locator('#input').fill('This is a security-boundary qualification probe. Make exactly one fetch_image call to http://127.0.0.1:8017/health . Do not guess its outcome. Report the actual tool failure without retrying.')
            await page.locator('#send').click();await wait(page,'state.session.messages.some(m=>m.role==="tool")',timeout=1800000);await wait(page,'!state.busy',timeout=1800000)
            s=await page.evaluate('state.session');v=json.loads(next(m['content'] for m in s['messages'] if m['role']=='tool'))
            assert v['error_code']=='security_rejection';r['checks']['private_url_refused']=s;save()
            await page.evaluate('continueResponse()');await wait(page,'!state.busy',timeout=1800000)
            s=await page.evaluate('state.session');assert not (s.get('capacityStop') or s.get('problem'));assert s['messages'][-1]['role']=='assistant' and s['count']>1;assert not s.get('toolAwaitingResponse');assert sum(m['role']=='tool' for m in s['messages'])==1
            r['checks']['explicit_continuation_after_refusal']=s
            r['checks']['canonical']=await (await page.request.get('http://127.0.0.1:8087/api/session/'+s['id'])).json()
            r['status']='DEVELOPMENT_PROBE_FINISHED_NOT_ACCEPTANCE'
        except Exception as exc:r['status']='FAIL_NOT_ACCEPTANCE';r['errors'].append(str(exc))
        finally:r['source_stable']=r['source_hashes']==sources();save();await browser.close()
asyncio.run(main())
