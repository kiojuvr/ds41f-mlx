#!/usr/bin/env python3
"""Finite real-socket ordinary Vision/MTP delta acceptance; one model only."""
import base64
import hashlib
import json
from pathlib import Path
import threading
import time


def run(output):
    import httpx
    import uvicorn
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    cfg = config(port=18071, profile='mtp-serving-v1')
    # Admit the unchanged installed dependencies/assets in isolated Python.
    # This is development delta acceptance, NOT admission of a new release.
    import subprocess
    import sys
    identity = json.loads(subprocess.check_output([sys.executable, '-I', '-c',
        'import json; from ds41f_mlx.mtp_identity import config, inspect; '
        'print(json.dumps(inspect(config(profile="mtp-serving-v1"))))'], text=True))
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level='warning'))
    thread = threading.Thread(target=server.run)
    thread.start()
    sources = ['serving/ordinary_admission.py', 'serving/production_mtp.py',
               'serving/paired_checkpoint.py', 'serving/internal_mtp.py',
               'serving/production_sessions.py', 'prefill_fp8_mlx/p6_append.py', 'runtime/multimodal.py']
    root = Path(__file__).resolve().parents[1]/'ds41f_mlx'
    receipt = dict(status='RUNNING', scope='checkout production-path delta, installed dependency/asset admission; not release admission',
                   identity={k: identity[k] for k in ('identity_sha256', 'actual_origins', 'hardware', 'compatibility')},
                   source_sha256={s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources}, cases=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(receipt, indent=2, default=str)+'\n')
    def trace_projection():
        keys = {'cached_tokens', 'remaining_suffix_tokens', 'generated', 'canonical_frontier',
                'target_offsets', 'dspark_offsets', 'mtp_stats', 'vision_encoded_count',
                'prompt_replay', 'full_cache_repack', 'cancelled', 'cache_published',
                'prediction_retired', 'queue_empty', 'execution_sampling', 'sampler_authority',
                'generation_policy', 'quiescence', 'authority_connection'}
        return {k: v for k, v in backend.serving_traces[-1].items() if k in keys or k.endswith('_s')}
    def image(name):
        path = cfg.checkpoint_path/'inference/examples/images'/name
        data = path.read_bytes()
        receipt.setdefault('images', {})[name] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
        return dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(data).decode()))
    def body(messages, **kw):
        return dict(model=cfg.model_id, temperature=0, max_tokens=96, messages=messages) | kw
    def idle(client):
        for _ in range(18000):
            health = client.get('/health').json()
            if not health['active_requests'] and not health['queued_requests']:
                assert not health['fatal_error'], health
                return
            time.sleep(.1)
        raise AssertionError('settlement timeout')
    save()
    try:
        while not server.started and thread.is_alive():
            time.sleep(.1)
        with httpx.Client(base_url=f'http://{cfg.host}:{cfg.port}', timeout=1800) as client:
            def post(label, messages, encoded=None, **kw):
                response = client.post('/v1/chat/completions', json=body(messages, **kw))
                if response.is_error:
                    receipt['failure'] = dict(label=label, status=response.status_code, body=response.text,
                        trace=trace_projection() if backend.serving_traces else None)
                    receipt['status'] = 'FAIL'
                    save()
                response.raise_for_status()
                result = response.json()
                idle(client)
                trace = trace_projection()
                receipt['cases'].append(dict(label=label, response=result, trace=trace))
                save()
                assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
                assert set(trace['target_offsets']+trace['dspark_offsets']) == {trace['canonical_frontier']}
                assert trace['cache_published'] and sum(trace['mtp_stats']['depth_drafted']) > 0
                if encoded is not None:
                    assert trace['vision_encoded_count'] == encoded
                print(label, result['choices'][0]['message'], trace['mtp_stats'], flush=True)
                return result['choices'][0]['message'], trace
            text = [dict(role='user', content='Explain immutable checkpoints in three short sentences.')]
            text_first, text_seed = post('text_seed', text)
            text_repeat, text_hit = post('text_hit', text)
            assert text_repeat == text_first and text_hit['cached_tokens'] > 0
            messages = [dict(role='user', content=[image('corn.jpeg'), dict(type='text', text='Identify this food, its color and shape in three short sentences.')])]
            first, seed = post('image_seed', messages, 1)
            assert 'corn' in first['content'].lower()
            _, repeat = post('image_repeat', messages, 0)
            assert repeat['cached_tokens'] > 0
            continuation = messages+[first, dict(role='user', content='Explain how this food is usually cooked in three short sentences.')]
            _, continued = post('text_after_image', continuation, 0)
            assert continued['cached_tokens'] >= repeat['cached_tokens']
            branch = messages+[first, dict(role='user', content='Instead describe its visible texture in two sentences.')]
            branch_reply, _ = post('image_branch', branch, 0)
            cancelled = branch+[branch_reply, dict(role='user', content='Write a long detailed essay about cooking this food.')]
            with client.stream('POST', '/v1/chat/completions', json=body(cancelled, stream=True, max_tokens=512)) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: ') and line[6:] != '[DONE]':
                        event = json.loads(line[6:])
                        if any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                            break
            idle(client)
            trace = trace_projection()
            receipt['cases'].append(dict(label='image_cancel', trace=trace))
            save()
            assert trace['cancelled'] and trace['vision_encoded_count'] == 0
            assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
            assert set(trace['target_offsets']+trace['dspark_offsets']) == {trace['canonical_frontier']}
            after_cancel, _ = post('image_after_cancel', cancelled, 0)
            second = cancelled+[after_cancel, dict(role='user', content=[image('carrots.jpeg'),
                dict(type='text', text='Identify this new food and contrast its color with the first food in two short sentences.')])]
            post('new_image_append', second, 1)
            # Changed past image must not select same-token KV or re-encode history.
            changed = [dict(role='user', content=[image('carrots.jpeg'), messages[0]['content'][1]]), first,
                       dict(role='user', content='Describe the food.')]
            response = client.post('/v1/chat/completions', json=body(changed))
            assert response.status_code >= 400
            idle(client)
            receipt['historical_edit_rejection'] = dict(status=response.status_code, body=response.text)
            text_final, _ = post('text_final', text)
            assert text_final == text_first
            receipt['status'] = 'PASS'
            save()
    finally:
        server.should_exit = True
        thread.join(timeout=1800)
        backend.close()
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/mtp-vision-acceptance.json'))
    run(parser.parse_args().output)
