"""Real browser/reload/dedupe/save/restore development probe (not final acceptance)."""
import asyncio
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.async_api import async_playwright
from tools.qualify_tool_binary_development import wait, sources

parser=argparse.ArgumentParser()
parser.add_argument('output',type=Path)
parser.add_argument('--prompt',default='Use fetch_image to inspect https://raw.githubusercontent.com/github/explore/main/topics/python/python.png and describe the actual colors and shapes briefly.')
parser.add_argument('--material',choices=['image','text'],default='image')
args=parser.parse_args()
OUTPUT = args.output


async def main():
    evidence = dict(schema='ds41f.tool_binary.recovery.development.v1',status='RUNNING_NOT_ACCEPTANCE',source_hashes=sources(),checks={},errors=[])
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    def save(): OUTPUT.write_text(json.dumps(evidence,indent=2)+'\n')
    save()
    async with async_playwright() as playwright:
        browser=await playwright.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=await browser.new_page()
        page.on('dialog',lambda dialog: asyncio.create_task(dialog.accept()))
        obtained=asyncio.Event(); release=asyncio.Event(); done=asyncio.Event(); result={}
        async def intercept(route):
            response=await route.fetch(timeout=1800000)
            result.update(await response.json())
            obtained.set(); await release.wait()
            try:
                await route.abort()
            finally:
                done.set()
        try:
            await page.goto('http://127.0.0.1:8087/')
            await wait(page,'state.runtime !== null')
            await wait(page,'state.session !== null && !state.busy')
            await page.evaluate("$('temperature').value='0'; $('thinking').value='off'; $('outputLimit').value='custom'; $('maxTokens').value='256'")
            await page.route('**/api/tools',intercept)
            await page.locator('#input').fill(args.prompt)
            await page.locator('#send').click()
            await asyncio.wait_for(obtained.wait(),timeout=1800)
            assert not result.get('budget_error') and not any(r.get('error') for r in result['results']),result
            before=await page.evaluate('state.session')
            evidence['checks']['completed_acquisition_before_consumption']=dict(count=before['count'],effects=before['effects'],receipt=result)
            # Ask the same settled effect again before consuming: observation of
            # the same transfer must return the same original-byte receipt.
            duplicate=await page.request.post('http://127.0.0.1:8087/api/tools',data={'session_id':before['id'],'request_count':before['count']})
            duplicate_json=await duplicate.json()
            assert duplicate_json==result,duplicate_json
            evidence['checks']['canonical_effect_dedupe']=True
            release.set(); await asyncio.wait_for(done.wait(),timeout=10)
            await page.unroute('**/api/tools',intercept)
            await page.reload(); await wait(page,'state.session !== null && !state.busy',timeout=30000)
            recovered=await page.evaluate('state.session')
            assert recovered['count']==before['count'] and recovered['toolAwaitingResponse']
            assert any(m['role']=='tool' and (isinstance(m['content'],list) if args.material=='image' else isinstance(m['content'],str)) for m in recovered['messages'])
            evidence['checks']['reload_before_consumption']=recovered
            save()
            await page.evaluate('continueTools()')
            await wait(page,'!state.busy',timeout=1800000)
            consumed=await page.evaluate('state.session')
            assert consumed['count']>before['count'] and not consumed.get('problem')
            evidence['checks']['consumption_after_reload']=consumed
            await page.locator('#settingsButton').click()
            await page.locator('#saveSession').click()
            await wait(page,"$('notice').textContent.startsWith('Saved native')",timeout=300000)
            snapshots=await page.evaluate("ChatStore.all('saves')")
            assert snapshots
            await page.locator('#saves').select_option(snapshots[0]['id'])
            await page.locator('#restoreSession').click()
            await wait(page,"$('notice').textContent.startsWith('Fresh native restore')",timeout=300000)
            restored=await page.evaluate('state.session')
            assert restored['messages']==consumed['messages'] and restored['id']!=consumed['id']
            evidence['checks']['save_fresh_restore_exact_history']=restored
            await page.locator('#settingsClose').click()
            n=len(restored['messages'])
            await page.locator('#input').fill('Without tools, what is 2 plus 3?')
            await page.locator('#send').click(); await wait(page,f'state.session.messages.length>{n}')
            await wait(page,'!state.busy',timeout=1800000)
            evidence['checks']['restored_ordinary_continuation']=await page.evaluate('state.session')
            evidence['status']='DEVELOPMENT_PROBE_FINISHED_NOT_ACCEPTANCE'
        except Exception as exc:
            evidence['status']='FAIL_NOT_ACCEPTANCE'; evidence['errors'].append(str(exc))
        finally:
            release.set(); evidence['source_stable']=evidence['source_hashes']==sources();save();await browser.close()


asyncio.run(main())
