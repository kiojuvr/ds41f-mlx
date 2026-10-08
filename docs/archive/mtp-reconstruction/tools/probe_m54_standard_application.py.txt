"""Official-checkpoint standard HTTP/socket probe, NOT M54 tool-envelope PASS.

Normal FastAPI stateful routes, real loopback HTTP client, no candidate scheduler.
Tools fail closed. This is partial correctness evidence, not a matched benchmark.
"""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', choices=['off', 'first-party-mtp-development'], default='first-party-mtp-development')
    args = parser.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import httpx
    import uvicorn
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg, execution_strategy=args.strategy)
    output = dict(decision='NOT PASS', strategy=args.strategy, phase='load', cases=[])
    start = time.perf_counter()
    counts = dict(target=0, forbidden=0)
    def save(phase):
        output['phase'] = phase
        output['elapsed_s'] = time.perf_counter() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + '\n')
        print(output['elapsed_s'], phase, flush=True)
    def guard(frame, event, arg):
        if event != 'call':
            return
        name = frame.f_code.co_filename.replace('\\', '/')
        if name.endswith('/runtime/target_forward.py') and frame.f_code.co_name == 'forward':
            counts['target'] += 1
        if (name.endswith('/model_execution/language.py') and frame.f_code.co_name == '_forward'
            or '/omlx/patches/mlx_lm_mtp/' in name
            or '/omlx/patches/deepseek_v41/' in name and name.endswith(('mtp.py', 'dspark.py', 'language.py'))):
            counts['forbidden'] += 1
            raise RuntimeError('forbidden donor/diagnostic execution: ' + name)
    async def run():
        await backend._call(backend.load)
        await backend._call(lambda: sys.setprofile(guard))
        from ds41f_mlx.serving.server import create_app
        app = create_app(backend=backend, runtime_config=cfg)
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        sock.listen(128)
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, lifespan='off', log_level='warning'))
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError('server did not start')
            await asyncio.sleep(.01)
        try:
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}', timeout=600.) as client:
                system = dict(role='system', content='A cache stores frequently used data. ' * 32)
                messages = [system, dict(role='user', content='Write OK exactly eight times separated by spaces and nothing else.')]
                rec = await backend.create_stateful_session()
                url = f'/v1/sessions/{rec.session_id}/chat/completions'
                for sequence, streaming in [(1, False), (2, True)]:
                    body = json.dumps(dict(model=cfg.model_id, messages=messages, temperature=0.,
                        reasoning_effort='none', max_tokens=32, stream=streaming), separators=(',', ':')).encode()
                    headers = {'X-DS41F-Request-Sequence': str(sequence), 'Content-Type':'application/json'}
                    save('standard ' + ('SSE' if streaming else 'JSON'))
                    response = await client.post(url, content=body, headers=headers)
                    assert response.status_code == 200, response.text
                    before = (rec.request_count, rec.m11.m8.frontier, counts['target'])
                    retry = await client.post(url, content=body, headers=headers)
                    assert retry.status_code == 200 and retry.content == response.content
                    assert before == (rec.request_count, rec.m11.m8.frontier, counts['target'])
                    diag = rec.m11.diagnostics()['m8']
                    assert diag['all_cache_offsets_equal_frontier']
                    assert diag['total_prompt_replay_count'] == diag['total_full_cache_repack_count'] == 0
                    output['cases'].append(dict(kind='dialogue', sequence=sequence, stream=streaming,
                        response_bytes=len(response.content), exact_retry=True, turn=rec.last_turn,
                        diagnostics=diag, metrics=getattr(rec.m11, 'last_cycle_metrics', None), target_calls=counts['target']))
                    save('turn completed')
                    message = rec.last_turn['response_json']['choices'][0]['message']
                    messages += [dict(role='assistant', content=message.get('content') or ''),
                                 dict(role='user', content='Write YES exactly eight times separated by spaces and nothing else.')]
                if args.strategy != 'off':
                    tool = dict(type='function', function=dict(name='lookup', parameters=dict(type='object')))
                    body = json.dumps(dict(model=cfg.model_id, messages=messages, tools=[tool],
                        max_tokens=16, stream=True, reasoning_effort='none')).encode()
                    before = (rec.request_count, rec.m11.m8.frontier, counts['target'], rec.response_reservation.sequence)
                    refused = await client.post(url, content=body, headers={'X-DS41F-Request-Sequence':'3'})
                    assert refused.status_code == 400, refused.text
                    assert before == (rec.request_count, rec.m11.m8.frontier, counts['target'], rec.response_reservation.sequence)
                    output['tool_fail_closed'] = dict(status=refused.status_code, body=refused.json(), mutation=False)
                await backend.close_stateful_session(rec.session_id)
                for partial in (False, True):
                    rec = await backend.create_stateful_session()
                    url = f'/v1/sessions/{rec.session_id}/chat/completions'
                    body = json.dumps(dict(model=cfg.model_id, messages=[system, dict(role='user',
                        content='Count from 1 to 100 separated by spaces.')], temperature=0.,
                        reasoning_effort='none', max_tokens=32 if partial else 8, stream=True), separators=(',', ':')).encode()
                    headers = {'X-DS41F-Request-Sequence':'1', 'Content-Type':'application/json'}
                    save('socket disconnect' if partial else 'max-token boundary')
                    prefix = b''
                    if partial:
                        async with client.stream('POST', url, content=body, headers=headers) as response:
                            assert response.status_code == 200
                            async for chunk in response.aiter_raw():
                                prefix += chunk
                                if prefix.count(b'data:') >= 3:
                                    break
                        for _ in range(2000):
                            if not rec.busy:
                                break
                            await asyncio.sleep(.01)
                        assert not rec.busy
                    else:
                        response = await client.post(url, content=body, headers=headers)
                        assert response.status_code == 200, response.text
                        prefix = response.content
                    slot = rec.response_reservation
                    assert slot.state == 'completed', rec.to_json()
                    frozen = b''.join(slot.chunks)
                    assert frozen.startswith(prefix)
                    before = (rec.request_count, rec.m11.m8.frontier, counts['target'])
                    retry = await client.post(url, content=body, headers=headers)
                    assert retry.status_code == 200 and retry.content == frozen
                    assert before == (rec.request_count, rec.m11.m8.frontier, counts['target'])
                    if not partial:
                        assert rec.last_turn['finish_reason'] == 'length'
                    else:
                        assert rec.last_turn['cancelled']
                    output['cases'].append(dict(kind='disconnect' if partial else 'length',
                        prefix_bytes=len(prefix), frozen_bytes=len(frozen), exact_retry=True,
                        turn=rec.last_turn, diagnostics=rec.m11.diagnostics()['m8'],
                        metrics=getattr(rec.m11, 'last_cycle_metrics', None), target_calls=counts['target']))
                    await backend.close_stateful_session(rec.session_id)
            child = getattr(backend._model.language_model, '_ds41f_proposal_child', None)
            output['idle_child'] = None if child is None else dict(receipts=len(child.receipts), producers=len(child.producers))
            if args.strategy != 'off':
                assert sum(c['metrics']['cycles'] for c in output['cases']) > 0, 'no real M52 cycle executed'
            output['counts'] = dict(counts)
            output['decision'] = 'PARTIAL STANDARD SOCKET PROBE PASS; M54 NOT PASS (tool EOF preview unqualified)'
        finally:
            server.should_exit = True
            await server_task
            sock.close()
            await backend._call(lambda: sys.setprofile(None))
        admission = backend._runtime.admission
        child = getattr(backend._model.language_model, '_ds41f_proposal_child', None)
        await backend._call(backend.close)
        output['resources'] = dict(parent_active=admission.active,
            child_active=None if child is None else child.active,
            receipts=None if child is None else len(child.receipts), producers=None if child is None else len(child.producers))
    try:
        save('start')
        asyncio.run(run())
    except BaseException:
        output['error'] = traceback.format_exc()
        raise
    finally:
        backend.close()
        save('complete')


if __name__ == '__main__':
    main()
