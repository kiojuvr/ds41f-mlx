"""Two real Native Messages turns against an isolated installed Flash-1017 candidate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import httpx

O = Path(__file__).resolve().parent
ENV = Path('/private/tmp/ds41f-sse-heartbeat-env')
PRODUCTION = Path('/Volumes/SDXC-512/ds41f-1017')
BASE = 'http://127.0.0.1:18089'
CONTROL = 'http://127.0.0.1:18441'


def save(name, value):
    (O/name).write_text(json.dumps(value, indent=2)+'\n')


def production_files():
    # Runtime/Recipe payloads, package metadata and entrypoints, not mutable pyc.
    paths = list((PRODUCTION/'bin').glob('*'))
    site = PRODUCTION/'lib/python3.13/site-packages'
    for pattern in ('ds41f_mlx*', 'deepseek_recipe*'):
        for root in site.glob(pattern):
            paths.extend(root.rglob('*'))
    return {str(p.relative_to(PRODUCTION)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file() and p.suffix != '.pyc'}


result = dict(status='RUNNING', turns=[])
assert not (O/'native-settled.jsonl').exists(), 'refusing to mix acceptance runs'
before = production_files()
save('production-before.json', before)
client = httpx.Client(timeout=30, trust_env=False)
for url in (BASE+'/health', CONTROL+'/ds41f/status'):
    try:
        client.get(url)
    except httpx.ConnectError:
        pass
    else:
        raise RuntimeError('existing listener; refusing to start')
env = dict(os.environ, DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
log = (O/'native-runtime.log').open('wb')
process = subprocess.Popen([str(ENV/'bin/python'), '-I', str(O/'candidate-launch.py'),
    '--profile', 'mtp-serving-v1', '--host', '127.0.0.1', '--port', '18089'],
    env=env, cwd='/tmp', stdout=log, stderr=subprocess.STDOUT)
result['pid'] = process.pid
save('native-acceptance.json', result)


def run_turn(label, body):
    save(label+'-request.json', body)
    started = time.monotonic()
    wire, chunks = bytearray(), []
    # Inactivity/read bound, deliberately NOT an absolute duration bound.
    with client.stream('POST', BASE+'/v1/messages', json=body,
                       timeout=httpx.Timeout(60, read=20)) as response:
        assert response.status_code == 200, response.read()
        headers = dict(response.headers)
        for chunk in response.iter_raw():
            wire.extend(chunk)
            chunks.append(dict(elapsed_s=time.monotonic()-started, bytes=len(chunk), text=chunk.decode()))
    (O/(label+'.sse')).write_bytes(wire)
    save(label+'-wire.json', dict(headers=headers, chunks=chunks))
    events = [json.loads(line[6:]) for line in wire.decode().splitlines() if line.startswith('data: ')]
    assert events[0]['type'] == 'message_start' and events[-1]['type'] == 'message_stop'
    # Read the SAME native canonical accumulator observed after settlement.
    # No second Recipe parser or response accumulator is introduced.
    settled = [json.loads(line) for line in (O/'native-settled.jsonl').read_text().splitlines()]
    canonical, = [r for r in settled if r['request_id'] == headers['x-ds41f-request-id']]
    response_body = canonical['response']
    assert events[0]['message']['id'] == response_body['id']
    assert next(e['delta']['stop_reason'] for e in events if e['type'] == 'message_delta') == response_body['stop_reason']
    save(label+'-response.json', response_body)
    state = client.get(CONTROL+'/ds41f/status').json()
    save(label+'-status.json', state)
    last = state['last_request']
    assert last['canonical_settled'] and last['cache_published'] and last['native_started'], last
    assert last['metrics']['aligned'] and last['metrics']['prompt_replay'] == last['metrics']['full_cache_repack'] == 0
    assert not state['failures'] and state['active_requests'] == state['queued_requests'] == 0
    result['turns'].append(dict(label=label, response=response_body, telemetry=last,
        heartbeats=wire.count(b': keepalive\n\n'), elapsed_s=time.monotonic()-started))
    save('native-acceptance.json', result)
    return response_body, wire


try:
    deadline = time.monotonic()+420
    while True:
        assert process.poll() is None, 'candidate exited'
        try:
            health = client.get(BASE+'/health').json()
            if health.get('model_ready'):
                break
        except httpx.HTTPError:
            pass
        assert time.monotonic() < deadline, 'startup deadline'
        time.sleep(1)
    save('initial-health.json', health)
    body = dict(model='deepseek-flash', stream=True, max_tokens=256, thinking={'type': 'disabled'},
        system='Call report_value exactly once with value 7. After its tool result, reply only ACK and do not call any more tools.',
        messages=[dict(role='user', content='Report value 7 now.')],
        tools=[dict(name='report_value', description='Record the integer requested by the user.',
                    input_schema=dict(type='object', properties={'value': {'type': 'integer'}}, required=['value']))])
    response, wire = run_turn('tool', body)
    assert wire.startswith(b': keepalive\n\n'), 'no heartbeat during controlled prefill silence'
    calls = [b for b in response['content'] if b['type'] == 'tool_use']
    assert len(calls) == 1 and calls[0]['name'] == 'report_value' and calls[0]['input'] == {'value': 7}, response
    assert response['stop_reason'] == 'tool_use'
    exact_id = calls[0]['id']
    body['messages'] += [dict(role='assistant', content=response['content']), dict(role='user', content=[
        dict(type='tool_result', tool_use_id=exact_id, content='Recorded value 7.')])]
    response, _ = run_turn('continuation', body)
    assert response['stop_reason'] == 'end_turn'
    assert not any(b['type'] == 'tool_use' for b in response['content'])
    assert 'ACK' in ''.join(b.get('text', '') for b in response['content'])
    result['exact_tool_id'] = exact_id
    result['final_health'] = client.get(BASE+'/health').json()
    assert result['final_health']['fatal_error'] is None
    result['status'] = 'PASS'
except BaseException:
    result.update(status='FAIL', failure=traceback.format_exc())
    raise
finally:
    try:
        if process.poll() is None:
            receipt = client.post(CONTROL+'/ds41f/control/shutdown')
            assert receipt.status_code == 202
            result['shutdown'] = receipt.json()
        result['exit_code'] = process.wait(timeout=240)
    except BaseException:
        result['shutdown_error'] = traceback.format_exc()  # No force kill of native work.
        result['status'] = 'FAIL'
    log.close()
    after = production_files()
    save('production-after.json', after)
    result['production_unchanged'] = before == after
    if before != after or result.get('exit_code') != 0:
        result['status'] = 'FAIL'
    save('native-acceptance.json', result)
    client.close()
assert result['status'] == 'PASS', result
