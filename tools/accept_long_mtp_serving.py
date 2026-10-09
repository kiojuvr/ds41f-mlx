#!/usr/bin/env python3
"""Finite production-boundary acceptance using existing qualified execution."""
import json
import time
from pathlib import Path

import httpx

from accept_generic_mtp_tools import run as generic_tools


def run():
    # Normal maintenance context, meaningfully beyond the former 8K envelope.
    context = ''.join(f'Journal entry {i}: immutable checkpoints preserve committed state; edits branch from a valid prefix.\n' for i in range(900))
    body = dict(model='deepseek-v4.1-flash', temperature=0, max_tokens=32,
                messages=[dict(role='user', content=context+'\nReply with exactly LONG_OK.')])
    receipt = {}
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=1800) as client:
        def post(label, request):
            response = client.post('/v1/chat/completions', json=request)
            response.raise_for_status()
            value = receipt[label] = response.json()
            print(label, value['usage'], flush=True)
            return value
        first = post('fresh', body)
        assert first['usage']['prompt_tokens'] > 12000
        assert 'LONG_OK' in first['choices'][0]['message']['content']
        repeat = post('repeat', body)
        cached = repeat['usage']['prompt_tokens_details']['cached_tokens']
        assert cached > 8192
        append = body | dict(messages=body['messages']+[first['choices'][0]['message'], dict(role='user', content='Now reply with exactly APPEND_OK.')])
        appended = post('append', append)
        assert appended['usage']['prompt_tokens_details']['cached_tokens'] >= cached
        assert 'APPEND_OK' in appended['choices'][0]['message']['content']
        branch = append | dict(messages=body['messages']+[first['choices'][0]['message'], dict(role='user', content='Instead reply with exactly BRANCH_OK.')])
        edited = post('branch', branch)
        assert 0 < edited['usage']['prompt_tokens_details']['cached_tokens'] <= appended['usage']['prompt_tokens_details']['cached_tokens']
        assert 'BRANCH_OK' in edited['choices'][0]['message']['content']
        # One representative disconnect after real streamed generation.
        cancelled = body | dict(stream=True, max_tokens=256,
            messages=body['messages']+[dict(role='assistant', content='LONG_OK'), dict(role='user', content='Write a detailed explanation of checkpoint ownership.')])
        with client.stream('POST', '/v1/chat/completions', json=cancelled) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith('data: ') and line[6:] != '[DONE]':
                    event = json.loads(line[6:])
                    if any(choice.get('delta', {}).get('content') for choice in event.get('choices', [])):
                        receipt['disconnect_event'] = event
                        break
        for _ in range(100):
            health = client.get('/health').json()
            if not health['active_requests'] and not health['queued_requests']:
                break
            time.sleep(.1)
        assert not health['fatal_error'] and not health['active_requests']
        receipt['after_disconnect'] = health
        recovered = post('post_disconnect', body)
        # A repeated *earlier* prompt may have been evicted by the intervening
        # append/branch/cancel publications. Recovery must be healthy, not a
        # guaranteed hit on an expired checkpoint.
        assert 'LONG_OK' in recovered['choices'][0]['message']['content']
        receipt['models'] = client.get('/v1/models').json()
    receipt['tools'] = generic_tools(context=context)
    return receipt


if __name__ == '__main__':
    result = run()
    Path('artifacts/long-mtp-serving/acceptance.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS: beyond-8K execution, paired repeat/append/branch, disconnect, generic tool continuation')
