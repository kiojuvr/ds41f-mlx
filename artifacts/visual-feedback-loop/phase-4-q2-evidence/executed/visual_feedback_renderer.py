#!/usr/bin/env python3
"""One-shot, network-denied static-page renderer for Phase 2 acceptance."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import signal

CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'


def check_png(path):
    from PIL import Image, ImageStat
    with Image.open(path) as image:
        if image.size != (1280, 720):
            raise ValueError('invalid screenshot dimensions')
        if max(ImageStat.Stat(image.convert('RGB')).stddev) < 2:
            raise ValueError('blank screenshot')


def render(source, output):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True, chromium_sandbox=True,
            args=['--disable-gpu', '--disable-background-networking', '--disable-component-update',
                  '--disable-sync', '--no-first-run', '--force-color-profile=srgb',
                  '--host-resolver-rules=MAP * ~NOTFOUND'])
        try:
            context = browser.new_context(viewport={'width': 1280, 'height': 720},
                device_scale_factor=1, java_script_enabled=False, locale='en-US',
                timezone_id='UTC', color_scheme='light', reduced_motion='reduce',
                service_workers='block')
            context.route('**/*', lambda route: route.abort())
            page = context.new_page()
            page.set_default_timeout(10000)
            html = source.read_text()
            csp = "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; style-src 'unsafe-inline'; font-src data:; script-src 'none'; connect-src 'none'\">"
            font = base64.b64encode(Path(FONT).read_bytes()).decode()
            style = "<style>@font-face{font-family:AcceptanceArial;src:url(data:font/ttf;base64," + font + ")}*{font-family:AcceptanceArial!important;animation:none!important;transition:none!important}</style>"
            page.set_content(csp + style + html, wait_until='load', timeout=10000)
            page.evaluate('document.fonts.ready')
            page.screenshot(path=str(output), timeout=10000)
            # Independent evaluation only: this is never returned as tool feedback.
            metrics = page.evaluate('''() => {
                const ids=['page','header','cards','card-1','card-2','card-3','footer'];
                let result={body_background:getComputedStyle(document.body).backgroundColor,overflow:document.documentElement.scrollWidth>1280 || document.documentElement.scrollHeight>720};
                for(const id of ids){const e=document.getElementById(id);if(!e){result[id]=null;continue;}
                  const r=e.getBoundingClientRect(),s=getComputedStyle(e);
                  result[id]={x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,right:r.right,
                    background:s.backgroundColor,color:s.color,display:s.display,text:e.innerText};}
                return result;
            }''')
            check_png(output)
            return dict(status='OK', metrics=metrics, browser=browser.version,
                font_sha256=hashlib.sha256(Path(FONT).read_bytes()).hexdigest())
        finally:
            browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    def terminate(signum, frame):
        raise TimeoutError('renderer terminated by execution budget')
    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps(render(args.source, args.output)))
