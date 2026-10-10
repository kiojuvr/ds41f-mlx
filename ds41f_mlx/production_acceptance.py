"""Bounded acceptance of an operator-owned ordinary MTP server via public HTTP."""
import argparse
import json
from pathlib import Path
import httpx


def run(url):
    from .mtp_identity import config, inspect
    admission = inspect(config(profile='mtp-serving-v1'))
    summary = {k: admission[k] for k in ('status', 'identity_sha256', 'build', 'hardware', 'actual_origins', 'compatibility')}
    summary['checkpoint_verification'] = admission['checkpoint']['verification']
    result = dict(schema='ds41f.installed-production-acceptance.v1', admission=summary,
                  status='RUNNING', cases={})
    tools = [dict(type='function', function=dict(name=name, description=description,
        parameters=dict(type='object', properties=dict(path=dict(type='string')), required=['path'])))
        for name, description in [('read_file', 'Read a file.'), ('stat_file', 'Get file metadata.')]]
    body = dict(model='deepseek-v4.1-flash', temperature=0, reasoning_effort='none', max_tokens=128,
        tools=tools, messages=[dict(role='user', content='Call read_file with path /tmp/example.txt. Do not answer until you receive the file content.')])
    with httpx.Client(base_url=url, timeout=1800) as client:
        def post(label, request):
            r = client.post('/v1/chat/completions', json=request)
            r.raise_for_status()
            value = r.json(); result['cases'][label] = value
            return value
        first = post('tool_call', body)
        message = first['choices'][0]['message']
        assert first['choices'][0]['finish_reason'] == 'tool_calls', first
        calls = message['tool_calls']
        assert len(calls) == 1 and calls[0]['function']['name'] == 'read_file', calls
        assert json.loads(calls[0]['function']['arguments']) == {'path': '/tmp/example.txt'}
        tool_result = dict(role='tool', tool_call_id=calls[0]['id'],
                           content='The file contains: GENERIC_TOOL_SUCCESS. Report this exact marker.')
        continuation = body | dict(messages=body['messages']+[message, tool_result])
        second = post('continuation', continuation)
        assert second['choices'][0]['finish_reason'] == 'stop'
        assert 'GENERIC_TOOL_SUCCESS' in second['choices'][0]['message']['content']
        cached = second['usage']['prompt_tokens_details']['cached_tokens']
        assert cached > 0
        branch = body | dict(messages=body['messages']+[message, tool_result | dict(
            content='The file contains: BRANCH_SUCCESS. Report this exact marker.')])
        changed = post('branch', branch)
        assert 'BRANCH_SUCCESS' in changed['choices'][0]['message']['content']
        assert changed['usage']['prompt_tokens_details']['cached_tokens'] <= cached
        with client.stream('POST', '/v1/chat/completions', json=body | dict(stream=True)) as r:
            r.raise_for_status()
            events = [line[6:] for line in r.iter_lines() if line.startswith('data: ')]
        result['cases']['sse'] = events
        assert events[-1] == '[DONE]'
        chunks = [json.loads(e) for e in events[:-1]]
        assert any(c.get('choices', [{}])[0].get('finish_reason') == 'tool_calls' for c in chunks if c.get('choices'))
        names = ''.join(call.get('function', {}).get('name', '') for c in chunks for choice in c.get('choices', [])
                        for call in choice.get('delta', {}).get('tool_calls', []))
        assert names == 'read_file'
        for malformed in (body | dict(tools=[{}]), body | dict(messages=body['messages']+[message, tool_result | dict(tool_call_id='foreign')])):
            r = client.post('/v1/chat/completions', json=malformed)
            assert r.status_code == 400
        result['health'] = client.get('/health').json()
    result['status'] = 'PASS'
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-url', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        result = run(args.runtime_url)
    except Exception as exc:
        args.output.write_text(json.dumps(dict(status='FAIL', error=repr(exc)),indent=2)+'\n')
        raise
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: installed admission, generic tools, paired continuation, branch, SSE, malformed requests')


if __name__ == '__main__':
    main()
