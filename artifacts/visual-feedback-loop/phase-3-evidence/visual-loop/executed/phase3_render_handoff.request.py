#!/usr/bin/env python3
"""Narrow evaluation handoff: request (confined) or supervise (host only).

No agent decisions, model calls, or tool-result rewriting. Chromium is launched
only by the external supervise mode, never by the approved request command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    # Atomic publication without replacing any previous render identity.
    os.link(temp, path)
    temp.unlink()

def request(workspace, timeout=24):
    stage = workspace / '.render-handoff'
    # Supervisor provisions this directory; a lock survives uncertain outcomes.
    lock = stage / 'pending'
    lock.mkdir()
    n = len(list(stage.glob('request-*.json')))
    if n >= 6:
        raise ValueError('six-render budget exhausted')
    source = (workspace / 'index.html').read_bytes()
    record = dict(id=n, source_sha256=sha(source), source_bytes=len(source))
    atomic(stage / f'request-{n:02}.json', record)
    end = time.monotonic() + timeout
    completion = stage / f'completion-{n:02}.json'
    while time.monotonic() < end:
        if completion.exists():
            result = json.loads(completion.read_text())
            assert result['id'] == n and result['source_sha256'] == record['source_sha256']
            if result['status'] != 'OK':
                raise RuntimeError('RENDER_ERROR: ' + result['error'])
            png = workspace / result['screenshot']
            assert sha(png.read_bytes()) == result['image_sha256']
            assert sha((workspace / 'index.html').read_bytes()) == record['source_sha256']
            lock.rmdir()
            print(json.dumps(dict(screenshot=str(png), source_sha256=result['source_sha256'], image_sha256=result['image_sha256'])))
            return
        time.sleep(.05)
    atomic(stage / f'timeout-{n:02}.json', dict(**record, status='ERROR', error='render completion timeout'))
    raise TimeoutError('RENDER_ERROR: render completion timeout; no automatic retry')

def serve_one(workspace, evidence, record):
    # Import only in the host mode. Existing validation, renderer and cleanup.
    sys.path.insert(0, str(ROOT / 'tools'))
    from accept_visual_feedback_loop import validate_html, bounded_process, quality
    stage = workspace / '.render-handoff'
    n = record['id']
    result = dict(id=n, source_sha256=record['source_sha256'], status='ERROR')
    try:
        assert type(n) is int and 0 <= n < 6
        source = (workspace / 'index.html').read_bytes()
        assert sha(source) == record['source_sha256'] and len(source) == record['source_bytes'], 'source revision changed before snapshot'
        validate_html(source.decode())
        snapshot = evidence / f'artifact-{n:02}.html'
        with snapshot.open('xb') as f:
            f.write(source)
        output = workspace / f'render-{n:02}.png'
        assert not output.exists(), 'render ID already used'
        renderer = json.loads(bounded_process(['/tmp/ds41f-phase2-renderer/bin/python', str(ROOT / 'tools/visual_feedback_renderer.py'), str(snapshot), str(output)], 22))
        assert (workspace / 'index.html').read_bytes() == source, 'source changed while rendering'
        assert not (stage / f'timeout-{n:02}.json').exists(), 'request timed out'
        image = output.read_bytes()
        with (evidence / output.name).open('xb') as f:
            f.write(image)
        atomic(evidence / f'metrics-{n:02}.json', dict(renderer=renderer, quality=quality(renderer['metrics'])))
        result.update(status='OK', screenshot=output.name, image_sha256=sha(image))
    except Exception as exc:
        result['error'] = str(exc)
    # Completion contains identity/status only, never metrics or corrective hints.
    atomic(stage / f'completion-{n:02}.json', result)
    atomic(evidence / f'completion-{n:02}.json', result)
    return result

def supervise(workspace, evidence, stop, timeout):
    stage = workspace / '.render-handoff'
    end = time.monotonic() + timeout
    seen = set()
    while time.monotonic() < end and not stop.exists():
        for path in sorted(stage.glob('request-*.json')):
            if path.name in seen:
                continue
            seen.add(path.name)
            record = json.loads(path.read_text())
            assert path.name == f'request-{record["id"]:02}.json'
            atomic(evidence / path.name, record)
            serve_one(workspace, evidence, record)
        time.sleep(.05)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['request', 'supervise'])
    p.add_argument('--workspace', type=Path)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--stop', type=Path)
    p.add_argument('--timeout', type=float, default=5400)
    a = p.parse_args()
    if a.mode == 'request':
        request(Path(os.environ['PHASE3_WORKSPACE']).resolve())
    else:
        supervise(a.workspace.resolve(), a.evidence.resolve(), a.stop, a.timeout)
