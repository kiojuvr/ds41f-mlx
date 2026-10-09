#!/usr/bin/env python3
"""Small real ordinary HTTP acceptance; server must use mtp-serving-v1."""
import json
from pathlib import Path
import httpx


def run(url='http://127.0.0.1:8000'):
    tools = [dict(type='function', function=dict(name=name, description=description,
        parameters=dict(type='object', properties=dict(path=dict(type='string')), required=['path'])))
        for name, description in [('read_file', 'Read a file.'), ('stat_file', 'Get file metadata.')]]
    body = dict(model='deepseek-v4.1-flash', temperature=0, reasoning_effort='none', max_tokens=128,
        tools=tools, messages=[dict(role='user', content='Call read_file with path /tmp/example.txt. Do not answer until you receive the file content.')])
    receipt = {}
    with httpx.Client(base_url=url, timeout=1800) as client:
        def post(label, request):
            r = client.post('/v1/chat/completions', json=request)
            r.raise_for_status()
            receipt[label] = r.json()
            return r.json()
        first = post('call', body)
        message = first['choices'][0]['message']
        assert first['choices'][0]['finish_reason'] == 'tool_calls', first
        calls = message['tool_calls']
        assert len(calls) == 1 and calls[0]['function']['name'] == 'read_file', calls
        assert json.loads(calls[0]['function']['arguments']) == {'path': '/tmp/example.txt'}
        result = dict(role='tool', tool_call_id=calls[0]['id'], content='The file contains: GENERIC_TOOL_SUCCESS. Report this exact marker.')
        continuation = body | dict(messages=body['messages']+[message, result])
        second = post('continuation', continuation)
        assert second['choices'][0]['finish_reason'] == 'stop'
        assert 'GENERIC_TOOL_SUCCESS' in second['choices'][0]['message']['content']
        cached = second['usage']['prompt_tokens_details']['cached_tokens']
        assert cached > 0, second
        branch = body | dict(messages=body['messages']+[message, result | dict(content='The file contains: BRANCH_SUCCESS. Report this exact marker.')])
        changed = post('branch', branch)
        assert 'BRANCH_SUCCESS' in changed['choices'][0]['message']['content']
        assert changed['usage']['prompt_tokens_details']['cached_tokens'] <= cached
        # Repeat generation as ordinary SSE. Recipe deltas, not a second parser.
        with client.stream('POST', '/v1/chat/completions', json=body | dict(stream=True)) as r:
            r.raise_for_status()
            events = [line[6:] for line in r.iter_lines() if line.startswith('data: ')]
        receipt['sse'] = events
        assert events[-1] == '[DONE]'
        chunks = [json.loads(e) for e in events[:-1]]
        assert any(c.get('choices', [{}])[0].get('finish_reason') == 'tool_calls' for c in chunks if c.get('choices'))
        names = ''.join(call.get('function', {}).get('name', '') for c in chunks for choice in c.get('choices', []) for call in choice.get('delta', {}).get('tool_calls', []))
        assert names == 'read_file', names
        for malformed in (body | dict(tools=[{}]), body | dict(messages=body['messages']+[message, result | dict(tool_call_id='foreign')])):
            r = client.post('/v1/chat/completions', json=malformed)
            assert r.status_code == 400 and 'weather' not in r.text, r.text
        receipt['health'] = client.get('/health').json()
    return receipt


if __name__ == '__main__':
    receipt = run()
    Path('artifacts/generic-mtp-tools').mkdir(exist_ok=True)
    Path('artifacts/generic-mtp-tools/acceptance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('PASS: generic calls, continuation, paired reuse, branch, SSE, malformed requests')
