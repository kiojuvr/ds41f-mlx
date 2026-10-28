"""Short installed ordinary readiness/API check, no external tools or session replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

envroot, output, port = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
output.parent.mkdir(exist_ok=True, parents=True)
base = f'http://127.0.0.1:{port}'
control = 'http://127.0.0.1:18441'
def call(url, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(url, data=data, method=method, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=180) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)
for address in (base+'/health', control+'/ds41f/status'):
    try:
        call(address)
    except URLError:
        pass
    else:
        raise RuntimeError('existing listener; refusing to start')
env = dict(os.environ, DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
receipt = {}
with open(os.devnull, 'wb') as null:
    proc = subprocess.Popen([envroot+'/bin/ds41f', 'start', '--host', '127.0.0.1', '--port', str(port)],
                            env=env, cwd='/tmp', stdout=null, stderr=null)
try:
    deadline = time.monotonic()+360
    while True:
        assert proc.poll() is None, 'startup process exited'
        try:
            code, h = call(base+'/health')
            if code == 200 and h['model_ready']:
                break
        except URLError:
            pass
        assert time.monotonic() < deadline, 'readiness deadline'
        time.sleep(1)
    receipt['initial_health'] = h
    assert call(control+'/ds41f/status')[1]['state'] == 'READY'
    body = dict(model='deepseek-v4.1-flash', max_tokens=16, temperature=-1,
                messages=[dict(role='user', content='Synthetic admission check.')])
    code, _ = call(base+'/v1/messages', body)
    receipt['rejection_http_status'] = code
    assert code in (400, 422), (code, _)
    status = call(control+'/ds41f/status')[1]
    assert status['state'] == 'READY' and status['last_request']['finish_reason'] == 'error'
    assert status['failures'][-1]['stage'] == 'preparation'
    assert status['failures'][-1]['source']['module'].startswith('ds41f_mlx.')
    receipt['recoverable_failure'] = status['failures'][-1]
    assert call(base+'/health')[1]['status'] == 'ready'
    logs = call(control+'/ds41f/logs?limit=128')[1]
    assert any('request failure' in r['message'] for r in logs['records'])
    receipt['logs_available_with_devnull'] = True
    body.update(max_tokens=16, temperature=0, messages=[dict(role='user', content='Reply only OK.')])
    code, response = call(base+'/v1/messages', body)
    assert code == 200 and response['content']
    status = call(control+'/ds41f/status')[1]
    receipt['successful_request'] = status['last_request']
    assert status['state'] == 'READY' and status['last_request']['finish_reason'] != 'error'
    health = call(base+'/health')[1]
    assert health['active_requests'] == health['queued_requests'] == 0
    assert health['status'] == 'ready' and health['fatal_error'] is None
    receipt['final_health'] = health
finally:
    if proc.poll() is None:
        code, _ = call(control+'/ds41f/control/shutdown', method='POST')
        assert code == 202
    receipt['exit_code'] = proc.wait(timeout=120)
    assert receipt['exit_code'] == 0
    receipt['gracefully_stopped'] = True
    output.write_text(json.dumps(receipt, indent=2))
print('PASS', output)
