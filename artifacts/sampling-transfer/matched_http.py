"""Bounded ownership-transfer collector; uses existing ordinary/control APIs."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import httpx

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--pid', type=int)
a = p.parse_args()
receipt = {'status': 'RUNNING', 'requests': []}
a.output.parent.mkdir(parents=True, exist_ok=True)
with httpx.Client(base_url='http://127.0.0.1:8000', trust_env=False, timeout=1800) as c, httpx.Client(base_url='http://127.0.0.1:18441', trust_env=False, timeout=30) as ctl:
    receipt['health'] = c.get('/health').json()
    def settled():
        for _ in range(600):
            s = ctl.get('/ds41f/status').json()
            if not s['active_requests'] and not s['queued_requests']:
                return s
            time.sleep(.1)
        raise TimeoutError('not settled; never resend')
    for temperature in (0., 1.):
        body = dict(model='deepseek-v4.1-flash', reasoning_effort='none', temperature=temperature,
                    top_p=.95, top_k=40, min_p=.02, seed=713, max_tokens=64,
                    messages=[dict(role='user', content='Explain why committed cache frontiers must align. Give four concise points.')])
        for repeat in range(4):
            start = time.perf_counter()
            r = c.post('/v1/chat/completions', json=body)
            r.raise_for_status()
            elapsed = time.perf_counter() - start
            s = settled()
            last = s['last_request']
            assert last is None or last['finish_reason'] != 'error', last
            metrics = (last or {}).get('metrics', {})
            if 'aligned' in metrics:
                assert metrics['aligned'] and metrics['prompt_replay'] == metrics['full_cache_repack'] == 0, metrics
            receipt['requests'].append(dict(temperature=temperature, repeat=repeat, warmup=repeat == 0,
                                            body=body, response=r.json(), elapsed_s=elapsed, settlement=last,
                                            rss_kib=int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(a.pid)])) if a.pid else None))
            a.output.write_text(json.dumps(receipt, indent=2)+'\n')
    cancel_body = body | dict(max_tokens=4096, stream=True,
        messages=[dict(role='user', content='Explain transactional cache ownership at length, with many examples and sections.')])
    with c.stream('POST', '/v1/chat/completions', json=cancel_body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith('data: ') and line != 'data: [DONE]':
                event = json.loads(line[6:])
                if any(x.get('delta', {}).get('content') for x in event.get('choices', [])):
                    break
        else:
            raise AssertionError('no streamed content before disconnect')
    s = settled()
    receipt['cancel_settlement'] = s['last_request']
    assert s['last_request'] is None or s['last_request']['finish_reason'] == 'abort', s
    # No automatic resend; a new explicit request succeeds after settlement.
    time.sleep(1)
    assert settled()['last_request'] == s['last_request']
    r = c.post('/v1/chat/completions', json=body | dict(temperature=0., max_tokens=16,
        messages=[dict(role='user', content='Reply with exactly RECOVERY_OK.')]))
    r.raise_for_status()
    receipt['recovery'] = r.json()
    receipt['recovery_settlement'] = settled()['last_request']
    assert 'RECOVERY_OK' in r.json()['choices'][0]['message']['content'], r.json()
    receipt['status'] = 'PASS'
    a.output.write_text(json.dumps(receipt, indent=2)+'\n')
print('PASS: matched greedy/stochastic, active disconnect and explicit recovery')
