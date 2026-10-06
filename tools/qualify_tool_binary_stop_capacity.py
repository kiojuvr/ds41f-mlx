"""Real browser Stop-after-Vision and completed-acquisition exhaustion probe."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.async_api import async_playwright
from tools.qualify_tool_binary_development import wait,sources

OUT=Path(sys.argv[1])

async def main():
    result=dict(schema='ds41f.tool_binary.stop_capacity.development.v1',status='RUNNING_NOT_ACCEPTANCE',source_hashes=sources(),checks={},errors=[])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    def save():OUT.write_text(json.dumps(result,indent=2)+'\n')
    save()
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=await browser.new_page()
        try:
            await page.goto('http://127.0.0.1:8087/');await wait(page,'state.session!==null&&!state.busy')
            await page.evaluate("$('temperature').value='0';$('thinking').value='off';$('outputLimit').value='custom';$('maxTokens').value='1024'")
            await page.locator('#input').fill('Use fetch_image to inspect https://www.gstatic.com/webp/gallery/1.jpg . After fetching, describe every visual detail of the actual photograph at great length, at least 2000 words.')
            await page.locator('#send').click()
            await wait(page,"state.busy&&state.session.messages.some(m=>m.role==='tool'&&Array.isArray(m.content))&&(state.live?.content?.length||0)>40",timeout=1800000)
            await page.locator('#stop').click();await wait(page,'!state.busy',timeout=1800000)
            stopped=await page.evaluate('state.session');assert stopped['interrupted']
            record=await page.request.get('http://127.0.0.1:8087/api/session/'+stopped['id'])
            canonical=await record.json();assert len(canonical['diagnostics']['image_identities'])==1
            result['checks']['stop_after_vision_consumption']=dict(application=stopped,canonical=canonical);save()
            await page.reload();await wait(page,'state.session!==null&&!state.busy')
            n=len(stopped['messages'])
            await page.locator('#input').fill('Now fetch and inspect ALL FOUR of these NEW images using fetch_image calls, preferably in one batch: https://www.gstatic.com/webp/gallery/2.webp , https://www.gstatic.com/webp/gallery/3.jpg , https://www.gstatic.com/webp/gallery/4.webp , https://raw.githubusercontent.com/github/explore/main/topics/python/python.png . Fetch all four before answering.')
            await page.locator('#send').click();await wait(page,f'state.session.messages.length>{n}')
            await wait(page,'!state.busy',timeout=1800000)
            exhausted=await page.evaluate('state.session')
            assert exhausted.get('capacityStop'),exhausted.get('toolError')
            acquired=[m for m in exhausted['messages'][n:] if m['role']=='tool' and isinstance(m['content'],list)]
            assert acquired,'No completed acquisitions before admission failure'
            result['checks']['completed_acquisitions_retained_at_admission_exhaustion']=exhausted
            ids=[json.loads(m['content'][0]['text'])['artifact_id'] for m in acquired]
            try:await page.evaluate('continueTools()')
            except Exception:pass  # expected same admission refusal, not retry
            await wait(page,'!state.busy',timeout=30000)
            after=await page.evaluate('state.session')
            now=[json.loads(m['content'][0]['text'])['artifact_id'] for m in after['messages'][n:] if m['role']=='tool' and isinstance(m['content'],list)]
            assert ids==now
            result['checks']['explicit_continuation_does_not_redownload']=after
            result['status']='DEVELOPMENT_PROBE_FINISHED_NOT_ACCEPTANCE'
        except Exception as exc:
            result['status']='FAIL_NOT_ACCEPTANCE';result['errors'].append(str(exc))
        finally:
            result['source_stable']=result['source_hashes']==sources();save();await browser.close()

asyncio.run(main())
