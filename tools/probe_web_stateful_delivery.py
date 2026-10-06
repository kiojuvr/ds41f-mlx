#!/usr/bin/env python3
"""Baseline probe, NOT Web application qualification.

Run against an already launched standard-OFF runtime and existing Web server.
Does not retry generation or execute tools. Owns/deletes only its test sessions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import time

import httpx


async def probe(args, receipt):
    async with httpx.AsyncClient(timeout=1800) as client:
        async def request(method, path, body=None):
            response = await client.request(method, args.runtime + path, json=body)
            response.raise_for_status()
            return response.json()

        receipt['health'] = await request('GET', '/health')
        sessions = []
        try:
            # Warm via the actual Web application's current /api/chat boundary.
            rec = await request('POST', '/v1/sessions', {})
            sessions.append(rec['id'])
            started = time.monotonic()
            response = await client.post(args.web + '/api/chat', json={
                'session_id': rec['id'], 'message': 'Reply with just hello.',
                'transcript': [], 'tools_enabled': False, 'max_tokens': 16,
            })
            response.raise_for_status()
            receipt['web_turn'] = {'seconds': time.monotonic() - started,
                                   'content_type': response.headers.get('content-type'),
                                   'response': response.json()}
            args.output.write_text(json.dumps(receipt, indent=2))

            rec = await request('POST', '/v1/sessions', {})
            sid = rec['id']
            sessions.append(sid)
            body = {'model': 'deepseek-v4.1-flash', 'messages': [
                {'role': 'user', 'content': 'List integers from 1 to 200, one per line. No explanation.'}
            ], 'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': 128, 'stream': True}
            started = time.monotonic()
            frames = []
            async with client.stream('POST', args.runtime + f'/v1/sessions/{sid}/chat/completions', json=body) as response:
                response.raise_for_status()
                headers_at = time.monotonic() - started
                at_headers = await request('GET', f'/v1/sessions/{sid}')
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        frames.append({'seconds': time.monotonic() - started, 'data': line[6:]})
            after = await request('GET', f'/v1/sessions/{sid}')
            delivered = ''.join(json.loads(frame['data']).get('choices', [{}])[0].get('delta', {}).get('content') or ''
                                for frame in frames if frame['data'] != '[DONE]')
            canonical = after['last_turn']['response_json']['choices'][0]['message'].get('content') or ''
            receipt['stateful_stream'] = {'headers_seconds': headers_at, 'record_at_headers': at_headers,
                                          'record_after': after, 'frames': frames,
                                          'delivered_text': delivered, 'canonical_text': canonical,
                                          'text_matches': delivered == canonical}
            args.output.write_text(json.dumps(receipt, indent=2))

            # Close the socket while generation is busy, before SSE headers exist.
            rec = await request('POST', '/v1/sessions', {})
            sid = rec['id']
            sessions.append(sid)
            async def consume():
                async with client.stream('POST', args.runtime + f'/v1/sessions/{sid}/chat/completions', json=body) as response:
                    response.raise_for_status()
                    async for _ in response.aiter_bytes():
                        pass
            task = asyncio.create_task(consume())
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                active = await request('GET', f'/v1/sessions/{sid}')
                if active['state'] == 'busy':
                    break
                if task.done():
                    raise RuntimeError('generation finished before active disconnect probe')
                await asyncio.sleep(.05)
            else:
                raise RuntimeError('session did not enter busy')
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            disconnected_at = time.monotonic()
            while time.monotonic() - disconnected_at < 180:
                settled = await request('GET', f'/v1/sessions/{sid}')
                if settled['state'] != 'busy':
                    break
                await asyncio.sleep(.1)
            else:
                raise RuntimeError('disconnected session did not settle')
            receipt['active_disconnect'] = {'active': active, 'settled': settled,
                                            'settlement_seconds': time.monotonic() - disconnected_at}
        finally:
            receipt['cleanup'] = []
            for sid in sessions:
                try:
                    receipt['cleanup'].append(await request('DELETE', f'/v1/sessions/{sid}'))
                except Exception as exc:
                    receipt['cleanup'].append({'id': sid, 'error': str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', default='http://127.0.0.1:8000')
    parser.add_argument('--web', default='http://127.0.0.1:8080')
    parser.add_argument('--output', type=Path, default=Path('artifacts/web-client-investigation/baseline.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'RUNNING', 'scope': 'baseline gap investigation; NOT application qualification',
               'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    args.output.write_text(json.dumps(receipt, indent=2))
    try:
        asyncio.run(probe(args, receipt))
        receipt['status'] = 'OBSERVED'
    except BaseException as exc:
        receipt.update(status='FAIL', error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
