"""Real browser/GPU development probe, deliberately not milestone acceptance."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import monotonic
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('ds41f_mlx',) for p in (ROOT / directory).rglob('*')
            if p.is_file() and p.suffix in {'.py', '.js', '.html', '.css'}}


async def wait(page, expression, timeout=30000):
    # Poll through DevTools, without page eval()/CSP bypass. New Playwright
    # string wait_for_function internally evals, which this app forbids.
    deadline = monotonic() + timeout / 1000
    while monotonic() < deadline:
        if await page.evaluate(expression):
            return
        await asyncio.sleep(.1)
    raise TimeoutError('browser observation deadline: ' + expression)


async def run(args):
    evidence = dict(schema='ds41f.tool_binary.development.v1', status='RUNNING_NOT_ACCEPTANCE',
                    source_hashes=sources(), workflows=[], errors=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    save()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
        page = await browser.new_page()
        page.on('pageerror', lambda error: evidence['errors'].append(str(error)))
        try:
            await page.goto(args.web)
            await wait(page, 'state.runtime !== null')
            await wait(page, 'state.session !== null && !state.busy')
            await page.evaluate(f"$('temperature').value='0'; $('thinking').value='off'; $('outputLimit').value='custom'; $('maxTokens').value='{args.max_tokens}'; $('toolsMode').value='auto'")
            await page.locator('#input').fill(args.prompt)
            await page.locator('#send').click()
            started = monotonic()
            await wait(page, 'state.session.messages.length > 0', timeout=30000)
            await wait(page, '!state.busy', timeout=1800000)
            result = await page.evaluate("({id:state.session.id,messages: state.session.messages, record:state.session.record, capacity:state.session.capacity, problem:state.session.problem, error:state.session.toolError, capacityStop:state.session.capacityStop,finishReason:state.session.finishReason,terminationReason:state.session.terminationReason})")
            canonical = await page.request.get(args.web.rstrip('/') + '/api/session/' + result['id'])
            evidence['workflows'].append(dict(name=args.workflow, seconds=monotonic()-started, result=result, canonical=await canonical.json()))
            save()
            if result['problem'] or result['error'] or result['capacityStop']:
                raise RuntimeError('workflow stopped with an error/admission boundary')
            if not any(m['role']=='tool' and isinstance(m['content'],list) for m in result['messages']):
                raise RuntimeError('model did not consume an actual fetched image')
            prior = len(result['messages'])
            # Test production Auto, not the research probe's 1024-token reserve.
            await page.evaluate("$('outputLimit').value='auto'")
            await page.locator('#input').fill('Now answer an ordinary text question without tools: what is 2 plus 3?')
            await page.locator('#send').click()
            await wait(page, f'state.session.messages.length > {prior}', timeout=30000)
            await wait(page, '!state.busy', timeout=1800000)
            continuation=await page.evaluate('state.session')
            assert not (continuation.get('capacityStop') or continuation.get('problem') or continuation.get('toolError')), continuation.get('capacityStop') or continuation.get('problem') or continuation.get('toolError')
            assert continuation['messages'][-1]['role']=='assistant' and len(continuation['messages'])>prior+1
            evidence['workflows'].append(dict(name='ordinary_text_continuation', result=continuation))
            canonical = await page.request.get(args.web.rstrip('/') + '/api/session/' + result['id'])
            evidence['workflows'][-1]['canonical'] = await canonical.json()
            await page.reload()
            await wait(page, 'state.session !== null && !state.busy', timeout=30000)
            evidence['workflows'].append(dict(name='reload_after_consumption', result=await page.evaluate('state.session')))
            evidence['status']='DEVELOPMENT_PROBE_FINISHED_NOT_ACCEPTANCE'
        except Exception as error:
            evidence['status']='FAIL_NOT_ACCEPTANCE'
            evidence['errors'].append(str(error))
        finally:
            evidence['source_stable'] = evidence['source_hashes']==sources()
            save()
            await browser.close()
    return 0 if evidence['status']=='DEVELOPMENT_PROBE_FINISHED_NOT_ACCEPTANCE' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--web',default='http://127.0.0.1:8087/')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--workflow',default='fetched_png_to_real_vision')
    parser.add_argument('--prompt', default='Use fetch_image to acquire https://raw.githubusercontent.com/github/explore/main/topics/python/python.png . Inspect the actual image using Vision and briefly describe its colors and shapes. Do not just describe its URL.')
    raise SystemExit(asyncio.run(run(parser.parse_args())))
