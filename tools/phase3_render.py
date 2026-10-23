#!/usr/bin/env python3
"""Fixed Phase 3 invocation of the existing renderer; no model/client dispatch."""
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    workspace = Path(os.environ['PHASE3_WORKSPACE']).resolve(strict=True)
    evidence = Path(os.environ['PHASE3_EVIDENCE']).resolve(strict=True)
    if evidence != workspace/'phase3-stage' or evidence.is_symlink():
        raise ValueError('evidence must be the private workspace staging directory')
    if Path.cwd().resolve() != workspace:
        raise ValueError('renderer cwd differs from admitted workspace')
    source = workspace/'index.html'
    if source.resolve(strict=True) != source or not source.is_file():
        raise ValueError('source must be a regular non-symlink file')
    spec = importlib.util.spec_from_file_location('phase2_validation', ROOT/'tools/accept_visual_feedback_loop.py')
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    html = source.read_text()
    validation.validate_html(html)
    records = sorted(evidence.glob('render-*.json'))
    number = len(records)
    if [p.name for p in records] != [f'render-{i:02}.json' for i in range(number)]:
        raise ValueError('nonsequential render metadata')
    if number > 5:
        raise ValueError('five-correction budget exhausted')
    output = workspace/f'render-{number:02}.png'
    if output.exists():
        raise ValueError('refusing uncertain/repeated render effect')
    artifact = evidence/f'artifact-{number:02}.html'
    # Exclusive reservation survives partial failure: never retry uncertain effects.
    with artifact.open('x') as f:
        f.write(html)
    if number:
        previous = (evidence/f'artifact-{number-1:02}.html').read_text()
        with (evidence/f'artifact-{number:02}.diff').open('x') as f:
            f.write(''.join(difflib.unified_diff(previous.splitlines(True), html.splitlines(True), fromfile=f'artifact-{number-1:02}.html', tofile=artifact.name)))
    # Use the Phase 2 bounded process-tree cleanup and the unchanged Chromium renderer.
    result = json.loads(validation.bounded_process([sys.executable, str(ROOT/'tools/visual_feedback_renderer.py'), str(source), str(output)], 22))
    png = output.read_bytes()
    with (evidence/output.name).open('xb') as f:
        f.write(png)
    result.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), screenshot_sha256=hashlib.sha256(png).hexdigest(), screenshot_bytes=len(png), screenshot=output.name)
    with (evidence/f'render-{number:02}.json').open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    # Never return independent DOM measurements, quality results, or visual hints.
    print(json.dumps(dict(status='OK', screenshot=str(output), viewport='1280x720', dpr=1, sha256=result['screenshot_sha256'])))

if __name__ == '__main__':
    main()
