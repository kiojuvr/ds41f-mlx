"""Installed production-only session acceptance against an already READY runtime.

No qualification injection, replay, OFF profile, or native state inspection.
Canonical GET (not received SSE text) owns the retained application history.
"""
import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stages = []
    receipt = {'status': 'RUNNING', 'stages': stages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    save()
    client = httpx.Client(base_url=args.base_url, timeout=180, trust_env=False)
    sid = 'accept-' + str(uuid4())
    path = '/v1/sessions/' + sid
    messages = []
    count = 0
    last_frontier = None
    tools = [{'type': 'function', 'function': {'name': 'multiply',
        'description': 'Multiply two integers. Use for multiplication questions.',
        'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                       'required': ['a','b'], 'additionalProperties': False}}}]
    ids = []
    def api(path, method='GET', value=None, headers=None):
        response = client.request(method, path, json=value, headers=headers)
        response.raise_for_status()
        return response.json()
    def request(maximum=128):
        return dict(model='deepseek-v4.1-flash', messages=messages, stream=True,
                    max_tokens=maximum, temperature=0, top_p=1, reasoning_effort='none',
                    tools=tools, tool_choice='auto')
    def fenced(application_id=None, expected=None):
        app = application_id or str(uuid4())
        return app, {'X-DS41F-Expected-Request-Count': str(count if expected is None else expected),
                     'X-DS41F-Application-Request-ID': app}
    def observe(application_id, label, request_id):
        nonlocal count, last_frontier
        for _ in range(1200):
            rec = api(path)
            if rec['state'] != 'busy':
                break
            time.sleep(.1)
        receipt['last_observed_session'] = rec
        assert rec['request_count'] == count+1, rec
        turn = rec['last_turn']
        assert turn['application_request_id'] == application_id and turn['request_id'] == request_id, rec
        diag = rec['diagnostics']['m8']
        C = diag['frontier']
        assert C == rec['canonical_frontier']
        assert diag['all_cache_offsets_equal_frontier']
        assert set(diag['cache_offsets_all']+diag['dspark_offsets']) == {C}
        assert diag['total_prompt_replay_count'] == diag['total_full_cache_repack_count'] == 0
        assert turn['metrics']['cache_published']
        assert rec['state'] == 'idle', rec
        previous = last_frontier
        count = rec['request_count']; messages.append(turn['response_json']['choices'][0]['message'])
        stages.append(dict(stage=label, count=count, frontier=C, aligned=True, replay=0, repack=0,
                           cancelled=turn['cancelled'], cached_tokens=turn['metrics']['cached_tokens'],
                           application_id=application_id, request_id=request_id,
                           executable_tools=turn['reconstruction']['executable_tools']))
        if previous is not None:
            assert turn['metrics']['cached_tokens'] == previous, (previous, stages[-1])
        last_frontier = C
        ids.append(application_id); save(); print(label, count, C, flush=True)
        return rec
    def generate(label, *, maximum=128, interruption=None):
        app, headers = fenced()
        events = 0
        with client.stream('POST', path+'/chat/completions', json=request(maximum), headers=headers) as response:
            response.raise_for_status(); rid = response.headers['x-ds41f-request-id']
            for line in response.iter_lines():
                if line.startswith('data: ') and line[6:] != '[DONE]':
                    value = json.loads(line[6:]); events += 1
                    if interruption and events >= 12:
                        active = api(path)
                        assert active['active_request_id'] == rid and active['request_count'] == count
                        assert client.delete(path, headers={'Content-Type':'application/json'}).status_code == 409
                        if interruption == 'cancel':
                            api(path+'/cancel', 'POST', {'request_id': rid})
                        break
        assert events > 0
        rec = observe(app, label, rid)
        if interruption:
            assert rec['last_turn']['cancelled']
        else:
            assert not rec['last_turn']['cancelled']
        receipt.pop('last_observed_session', None)
        return rec
    try:
        health = api('/health'); assert health['profile'] == 'mtp-serving-v1' and health['status'] == 'ready'
        assert health['capabilities']['stateful_chat'] == 'production-sessions-v1'
        receipt['dependency_identity'] = health['dependency_identity']
        receipt['profile'] = health['profile']
        created = api('/v1/sessions', 'POST', {'id': sid})
        assert created['state'] == 'empty' and created['request_count'] == 0
        assert client.post('/v1/sessions', json={}).status_code == 409
        stages.append({'stage':'session create', 'id':sid}); save()
        messages.append({'role':'user','content':'こんにちは。短く日本語で挨拶してください。ツールは不要です。'})
        budget = api(path+'/budget', 'POST', request('auto'))
        assert budget['request_count'] == 0 and budget['model_output_ceiling'] == 393216
        generate('first streamed turn + canonical GET')
        messages.append({'role':'user','content':'Now say hello in English, one short sentence. No tools.'})
        generate('multi-turn exact continuation')
        # Ordinary requests share the SAME physical authority, but cannot change
        # logical truth or evict its upstream-refcounted canonical pair.
        ordinary_settlements = []
        for index in range(4):
            api('/v1/chat/completions', 'POST', dict(model='deepseek-v4.1-flash',
                messages=[{'role':'user','content':f'Cache pressure probe {index}: reply with OK only.'}],
                max_tokens=32, temperature=0))
            with httpx.Client(base_url='http://127.0.0.1:18441', trust_env=False) as control:
                metric = control.get('/ds41f/status').json()['last_request']['metrics']
            assert metric['aligned'] and metric['prompt_replay'] == metric['full_cache_repack'] == 0
            ordinary_settlements.append({k:metric[k] for k in ('canonical_frontier','aligned','prompt_replay','full_cache_repack')})
            assert api(path)['request_count'] == count and api(path)['canonical_frontier'] == last_frontier
        stages.append({'stage':'ordinary cache pressure preserves session truth and pinned pair',
                       'ordinary_settlements':ordinary_settlements}); save()
        messages.append({'role':'user','content':'Call the multiply tool with a=6 and b=7, then tell me its result. You MUST use the tool.'})
        rec = generate('completed tool call')
        calls = rec['last_turn']['response_json']['choices'][0]['message'].get('tool_calls') or []
        assert calls and rec['last_turn']['reconstruction']['executable_tools']
        for call in calls:
            assert call['function']['name'] == 'multiply'
            value = json.loads(call['function']['arguments'])
            messages.append({'role':'tool','tool_call_id':call['id'], 'content':json.dumps({'result':value['a']*value['b']})})
        generate('certified tool result continuation')
        messages.append({'role':'user','content':'Write integers from 1 to 1000, one per line, without tools or omissions.'})
        generate('mid-stream disconnect + canonical reconciliation', maximum=2048, interruption='disconnect')
        messages.append({'role':'user','content':'Stop the list. Reply with one short sentence confirming you can continue. No tools.'})
        generate('safe continuation after disconnect')
        messages.append({'role':'user','content':'Write integers from 1 to 1000, one per line, without tools or omissions.'})
        generate('explicit cancellation + canonical reconciliation', maximum=2048, interruption='cancel')
        messages.append({'role':'user','content':'Reply with OK, no tools.'})
        current = request()
        _, headers = fenced(expected=count-1)
        assert client.post(path+'/chat/completions', json=current, headers=headers).status_code == 409
        _, headers = fenced(application_id=ids[-1])
        assert client.post(path+'/chat/completions', json=current, headers=headers).status_code == 409
        _, headers = fenced(application_id='invalid')
        assert client.post(path+'/chat/completions', json=current, headers=headers).status_code == 400
        assert api(path)['request_count'] == count
        stages.append({'stage':'stale count and reused/malformed application identities rejected', 'count':count}); save()
        generate('safe continuation after explicit cancel and rejected requests')
        for unsupported in (path+'/persist','/v1/sessions/restore'):
            assert client.post(unsupported, json={}).status_code == 400
        api(path, 'DELETE', headers={'Content-Type':'application/json'})
        assert client.get(path).status_code == 404
        assert client.post('/v1/sessions', json={'id':sid}).status_code == 409
        stages.append({'stage':'session close; absent and nonreusable identity'})
        other = api('/v1/sessions', 'POST', {'id':'fresh-'+str(uuid4())})
        other_path = '/v1/sessions/'+other['id']
        _, headers = fenced(expected=0)
        assert client.post(other_path+'/chat/completions', json=request(), headers=headers).status_code == 400
        fresh = api(other_path)
        assert fresh['state'] == 'empty' and fresh['request_count'] == 0 and fresh['last_turn'] is None
        api(other_path, 'DELETE', headers={'Content-Type':'application/json'})
        stages.append({'stage':'new session rejects another session canonical history; cache is not session truth'})
        receipt['status'] = 'PASS'; save()
    except BaseException as exc:
        receipt.update(status='FAIL', error=repr(exc)); save(); raise
    finally:
        try:
            rec = api(path)
            if rec.get('active_request_id'):
                api(path+'/cancel','POST',{'request_id':rec['active_request_id']})
            client.delete(path, headers={'Content-Type':'application/json'})
        except Exception:
            pass
        client.close()


if __name__ == '__main__':
    main()
