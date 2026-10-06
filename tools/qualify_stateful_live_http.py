#!/usr/bin/env python3
"""Real standard-OFF stateful HTTP delivery qualification; no MTP selector.

Starts no model itself. Launch ds41f_mlx.serve first. Test-owned sessions are
explicitly retired. Every continuation uses ordinary messages from GET, no retry.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time
import sys
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compact(rec):
    turn = rec.get('last_turn') or {}
    m8 = (rec.get('diagnostics') or {}).get('m8') or {}
    return dict(id=rec['id'], state=rec['state'], request_count=rec['request_count'],
                active_request_id=rec.get('active_request_id'), recovery_state=rec.get('recovery_state'),
                cancelled=turn.get('cancelled'), reconstruction=turn.get('reconstruction'),
                response=turn.get('response_json'), generated_count=len(turn.get('generated_tokens', [])),
                frontier=m8.get('frontier'), offsets=m8.get('cache_offsets_all'),
                replay=m8.get('total_prompt_replay_count'), repack=m8.get('total_full_cache_repack_count'),
                image_encoded_count=(rec.get('diagnostics') or {}).get('last_image_encoded_count'),
                diagnostic_event_count=len(turn.get('stream_events', [])), last_error=rec.get('last_error'))


async def exercise(args, result):
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    async with httpx.AsyncClient(timeout=1800) as client:
        sessions = set()
        async def api(method, path, body=None):
            r = await client.request(method, args.runtime + path, json=body)
            r.raise_for_status()
            return r.json()
        async def create():
            rec = await api('POST', '/v1/sessions', {})
            sessions.add(rec['id'])
            return rec['id']
        async def close(sid):
            await api('DELETE', '/v1/sessions/' + sid)
            sessions.remove(sid)
        def body(messages, maximum=192, **extra):
            return dict(model=cfg.model_id, messages=messages, stream=True, reasoning_effort='none',
                        temperature=0, max_tokens=maximum, **extra)
        async def settled(sid):
            for _ in range(600):
                rec = await api('GET', '/v1/sessions/' + sid)
                if rec['state'] != 'busy':
                    return rec
                await asyncio.sleep(.05)
            raise AssertionError('settlement timed out')
        async def stream(sid, request, *, disconnect_after=None, stop_after=None, slow=False):
            started = time.monotonic()
            events = []
            first_content = first_state = None
            request_id = None
            explicit = None
            before = await api('GET', '/v1/sessions/' + sid)
            nonce = str(uuid4())
            async with client.stream('POST', args.runtime + f'/v1/sessions/{sid}/chat/completions', json=request,
                    headers={'X-DS41F-Expected-Request-Count': str(before['request_count']), 'X-DS41F-Application-Request-ID': nonce}) as r:
                r.raise_for_status()
                headers_at = time.monotonic()-started
                request_id = r.headers['x-ds41f-request-id']
                async for line in r.aiter_lines():
                    if not line.startswith('data: '): continue
                    data = line[6:]
                    if data == '[DONE]': break
                    event = json.loads(data)
                    events.append(event)
                    delta = event.get('choices', [{}])[0].get('delta') or {}
                    if first_content is None and (delta.get('content') or delta.get('reasoning_content')):
                        first_content = time.monotonic()-started
                        first_state = compact(await api('GET', '/v1/sessions/' + sid))
                    if stop_after and len(events) >= stop_after:
                        explicit = await api('POST', f'/v1/sessions/{sid}/cancel', {'request_id': request_id})
                        break
                    if disconnect_after and len(events) >= disconnect_after: break
                    if slow and len(events) % 8 == 0: await asyncio.sleep(.8)
            rec = await settled(sid)
            observation = dict(headers_seconds=headers_at, first_content_seconds=first_content,
                               total_seconds=time.monotonic()-started, first_content_state=first_state,
                               request_id=request_id, events=events, final=compact(rec),
                               explicit_cancel=explicit is not None)
            assert rec['state'] == 'idle', compact(rec)
            assert rec['last_turn']['application_request_id'] == nonce
            diag = rec['diagnostics']['m8']
            assert diag['cache_layer_count'] == 40 and diag['all_cache_offsets_equal_frontier']
            assert diag['total_prompt_replay_count'] == diag['total_full_cache_repack_count'] == 0
            received = ''.join((e.get('choices', [{}])[0].get('delta') or {}).get('content') or '' for e in events)
            canonical = rec['last_turn']['response_json']['choices'][0]['message'].get('content') or ''
            assert canonical.startswith(received), (received, canonical)
            if not disconnect_after and not stop_after:
                assert canonical == received
                calls = {}
                for event in events:
                    for delta in (event.get('choices', [{}])[0].get('delta') or {}).get('tool_calls') or []:
                        call = calls.setdefault(delta['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        if delta.get('id'): call['id'] = delta['id']
                        function = delta.get('function') or {}
                        call['function']['name'] += function.get('name') or ''
                        call['function']['arguments'] += function.get('arguments') or ''
                assert [calls[k] for k in sorted(calls)] == (message(rec).get('tool_calls') or [])
                reasons = [e['choices'][0]['finish_reason'] for e in events if e.get('choices') and e['choices'][0].get('finish_reason')]
                assert reasons == [rec['last_turn']['response_json']['choices'][0]['finish_reason']]
            else:
                assert rec['last_turn']['cancelled']
                assert rec['last_turn']['reconstruction']['representable']
                assert len(rec['last_turn']['generated_tokens']) < request['max_tokens']
            return observation, rec
        def message(rec): return rec['last_turn']['response_json']['choices'][0]['message']
        def save(): args.output.write_text(json.dumps(result, indent=2))
        try:
            count = [dict(role='user', content='List integers from 1 to 200, one per line. No explanation.')]
            sid = await create()
            observation, rec = await stream(sid, body(count))
            result['long'] = observation; save()
            # Stale admission must reject before any mutation, not regenerate.
            stale = await client.post(args.runtime + f'/v1/sessions/{sid}/chat/completions', json=body(count),
                    headers={'X-DS41F-Expected-Request-Count': '0'})
            assert stale.status_code == 409
            unchanged = await api('GET', '/v1/sessions/' + sid)
            assert unchanged['request_count'] == rec['request_count'] and unchanged['diagnostics']['m8']['frontier'] == rec['diagnostics']['m8']['frontier']
            result['stale_admission_status'] = stale.status_code; save()
            assert len(observation['events']) > 64
            assert observation['first_content_state']['state'] == 'busy'
            assert observation['total_seconds'] - observation['first_content_seconds'] > 1
            await close(sid)

            sid = await create()
            observation, rec = await stream(sid, body(count), slow=True)
            result['slow'] = observation; save()
            assert observation['final']['response']['choices'][0]['message']['content'] == result['long']['final']['response']['choices'][0]['message']['content']
            await close(sid)

            result['cancellations'] = []
            for explicit in (False, True, False, True):
                sid = await create()
                observation, rec = await stream(sid, body(count, 256), **({'stop_after': 12} if explicit else {'disconnect_after': 12}))
                messages = count + [message(rec), dict(role='user', content='What was the first number? Answer briefly.')]
                continuation, rec = await stream(sid, body(messages, 32))
                result['cancellations'].append(dict(interruption=observation, reuse=continuation)); save()
                await close(sid)

            tool = {'type': 'function', 'function': {'name': 'lookup_weather', 'description': 'Look up weather',
                    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}
            sid = await create()
            messages = [dict(role='user', content='Use lookup_weather for Paris and then answer briefly.')]
            call_stream, rec = await stream(sid, body(messages, 96, tools=[tool], tool_choice={'type':'function','function':{'name':'lookup_weather'}}))
            calls = message(rec)['tool_calls']
            assert rec['last_turn']['reconstruction']['executable_tools']
            assert len(calls) == 1
            messages += [message(rec), dict(role='tool', tool_call_id=calls[0]['id'], content='{"city":"Paris","weather":"sunny"}')]
            continuation, rec = await stream(sid, body(messages, 64, tools=[tool], tool_choice='auto'))
            result['tool'] = dict(call=call_stream, continuation=continuation, client_effect_count=1); save()
            await close(sid)

            # Interrupt inside a DSML/tool prefix, before it is execution permission.
            sid = await create()
            request = body([dict(role='user', content='Use lookup_weather for Paris.')], 256,
                tools=[tool], tool_choice={'type':'function','function':{'name':'lookup_weather'}})
            async with client.stream('POST', args.runtime + f'/v1/sessions/{sid}/chat/completions', json=request) as response:
                response.raise_for_status()
                request_id = response.headers['x-ds41f-request-id']
                lines = response.aiter_lines()  # keep iterator alive: break is not a disconnect
                async for line in lines:
                    if line.startswith('data: '): break
                await asyncio.sleep(.2)
                await api('POST', f'/v1/sessions/{sid}/cancel', {'request_id': request_id})
            partial = await settled(sid)
            cert = partial['last_turn']['reconstruction']
            assert partial['state'] in ('idle', 'unrecoverable') and not cert['executable_tools']
            assert partial['last_turn']['cancelled'] and len(partial['last_turn']['generated_tokens']) < 256
            result['tool_prefix_cancel'] = dict(final=compact(partial), client_effect_count=0,
                note='explicit protocol-unrecoverable retirement is allowed; not universal tool-prefix resume')
            save(); await close(sid)

            def image(name):
                data = (cfg.checkpoint_path/'inference/examples/images'/name).read_bytes()
                result.setdefault('images', {})[name] = hashlib.sha256(data).hexdigest()
                return dict(type='image_url', image_url={'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()})
            sid = await create()
            messages = [dict(role='user', content=[image('corn.jpeg'), dict(type='text', text='Identify the food briefly.')])]
            vision, rec = await stream(sid, body(messages, 48))
            assert vision['final']['image_encoded_count'] == 1
            messages += [message(rec), dict(role='user', content='What color is it?')]
            text, rec = await stream(sid, body(messages, 48))
            assert text['final']['image_encoded_count'] == 0
            messages += [message(rec), dict(role='user', content=[image('carrots.jpeg'), dict(type='text', text='Contrast this food with the first briefly.')])]
            second, rec = await stream(sid, body(messages, 48))
            assert second['final']['image_encoded_count'] == 1
            messages += [message(rec), dict(role='user', content='Describe both foods in great detail in 20 paragraphs.')]
            cancelled, rec = await stream(sid, body(messages, 256), stop_after=12)
            assert cancelled['final']['image_encoded_count'] == 0
            messages += [message(rec), dict(role='user', content='Which food is orange? Answer briefly.')]
            reused, rec = await stream(sid, body(messages, 32))
            result['vision'] = dict(initial=vision, text=text, additional=second, cancel=cancelled, reuse=reused); save()
            await close(sid)
        finally:
            result['cleanup'] = []
            for sid in list(sessions):
                try: await close(sid); result['cleanup'].append({'id':sid,'closed':True})
                except Exception as exc: result['cleanup'].append({'id':sid,'error':str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/stateful-live/http.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = ['ds41f_mlx/runtime/live_turn.py', 'ds41f_mlx/serving/stateful_stream.py',
               'ds41f_mlx/serving/server.py', 'ds41f_mlx/serving/deepseek_recipe_backend.py',
               'ds41f_mlx/serving/recovery_certificate.py']
    result = {'status':'RUNNING','scope':'real standard-OFF HTTP; not full Web/R1',
              'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    args.output.write_text(json.dumps(result, indent=2))
    try:
        asyncio.run(exercise(args, result))
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally: args.output.write_text(json.dumps(result, indent=2))

if __name__ == '__main__': main()
