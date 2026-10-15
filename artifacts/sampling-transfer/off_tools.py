"""Bounded OFF tool/result fixture over existing recipe HTTP APIs."""
import json
from pathlib import Path
import sys
import httpx

with httpx.Client(base_url='http://127.0.0.1:8000', timeout=1800, trust_env=False) as c:
    body = dict(model='deepseek-v4.1-flash', temperature=0, reasoning_effort='none', max_tokens=128,
        tools=[dict(type='function', function=dict(name='read_file', description='Read a file.',
            parameters=dict(type='object', properties=dict(path=dict(type='string')), required=['path'])))],
        messages=[dict(role='user', content='Call read_file with path /tmp/example.txt. Do not answer until you receive the file content.')])
    r=c.post('/v1/chat/completions', json=body);r.raise_for_status();first=r.json()
    msg=first['choices'][0]['message'];calls=msg['tool_calls']
    assert first['choices'][0]['finish_reason']=='tool_calls' and len(calls)==1
    assert calls[0]['function']['name']=='read_file'
    assert json.loads(calls[0]['function']['arguments'])=={'path':'/tmp/example.txt'}
    body['messages'] += [msg,dict(role='tool',tool_call_id=calls[0]['id'],content='The file contains: GENERIC_TOOL_SUCCESS. Report this exact marker.')]
    with c.stream('POST','/v1/chat/completions',json=body|dict(stream=True)) as r:
        r.raise_for_status();events=[line[6:] for line in r.iter_lines() if line.startswith('data: ')]
    assert events[-1]=='[DONE]'
    chunks=[json.loads(e) for e in events[:-1]]
    text=''.join(x.get('delta',{}).get('content','') or '' for chunk in chunks for x in chunk.get('choices',[]))
    assert 'GENERIC_TOOL_SUCCESS' in text
    Path(sys.argv[1]).write_text(json.dumps(dict(status='PASS',call=first,result_text=text,sse=events),indent=2)+'\n')
print('PASS: OFF tool call and result SSE')
