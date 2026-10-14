"""Bounded ordinary MTP profile acceptance over existing production APIs."""
import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx
from accept_generic_mtp_tools import run as generic_tools


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    receipts = []
    with httpx.Client(base_url='http://127.0.0.1:18441', trust_env=False, timeout=30) as control:
        def settlement():
            status = control.get('/ds41f/status').json()
            last = status['last_request']
            metrics = last['metrics']
            assert last['finish_reason'] != 'error', last
            assert metrics['aligned'] and metrics['prompt_replay'] == metrics['full_cache_repack'] == 0, metrics
            assert metrics['proposed_drafts'] > 0, metrics
            receipts.append(last)
        # Reuse the existing ordinary tools/append/branch/SSE fixture. Capture
        # each completed HTTP generation before its malformed-request negatives.
        ordinary = generic_tools(on_settlement=settlement)
        assert ordinary['health']['glm_profile'] == 'native'
        stages = []
        with httpx.Client(base_url='http://127.0.0.1:8000', trust_env=False, timeout=1800) as client:
            sid = 'native-'+str(uuid4())
            path = '/v1/sessions/'+sid
            created = client.post('/v1/sessions', json={'id': sid})
            created.raise_for_status()
            messages = [{'role':'user', 'content':'Reply with exactly NATIVE_OK. No tools.'}]
            previous = 0
            try:
                for count in range(2):
                    body = dict(model='deepseek-v4.1-flash', messages=messages,
                        temperature=0, max_tokens=32, reasoning_effort='none', stream=True)
                    headers = {'X-DS41F-Expected-Request-Count':str(count),
                               'X-DS41F-Application-Request-ID':str(uuid4())}
                    with client.stream('POST', path+'/chat/completions', json=body, headers=headers) as response:
                        response.raise_for_status()
                        events = [line[6:] for line in response.iter_lines() if line.startswith('data: ')]
                    assert events[-1] == '[DONE]'
                    for _ in range(100):
                        rec = client.get(path).json()
                        if rec['state'] == 'idle':
                            break
                        time.sleep(.1)
                    assert rec['state'] == 'idle' and rec['request_count'] == count+1, rec
                    diag = rec['diagnostics']['m8']
                    frontier = rec['canonical_frontier']
                    assert diag['frontier'] == frontier and diag['all_cache_offsets_equal_frontier']
                    assert set(diag['cache_offsets_all']+diag['dspark_offsets']) == {frontier}
                    assert diag['total_prompt_replay_count'] == diag['total_full_cache_repack_count'] == 0
                    turn = rec['last_turn']
                    assert turn['metrics']['cache_published'] and not turn['cancelled']
                    if count:
                        assert turn['metrics']['cached_tokens'] == previous
                    stages.append(rec)
                    messages += [turn['response_json']['choices'][0]['message'],
                                 {'role':'user', 'content':'Now reply with exactly CONTINUE_OK. No tools.'}]
                    previous = frontier
                response = client.delete(path, headers={'Content-Type':'application/json'})
                response.raise_for_status()
            finally:
                client.delete(path, headers={'Content-Type':'application/json'})
        receipt = dict(status='PASS', glm_profile='native', ordinary=ordinary,
            ordinary_settlements=receipts, paired_session_stages=stages,
            replay=0, repack=0, selected_path_receipt='selected-path.json')
        args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    print('PASS: ordinary tools/append/branch/SSE and exact paired session frontiers; native; replay=0 repack=0')


if __name__ == '__main__':
    main()
