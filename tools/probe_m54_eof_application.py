"""Real checkpoint / standard socket routes / existing Web effect authority.

Correctness only. No candidate scheduler, synthetic generation or performance run.
"""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import sys
import time
import traceback
from threading import Event


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--strategy', choices=['off', 'first-party-mtp-development'], default='first-party-mtp-development')
    args = ap.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import httpx
    import uvicorn
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.serving.server import create_app
    from ds41f_mlx import web
    from ds41f_mlx.web_tools import ToolRegistry, ToolResult
    from tools.run_m11_tool_boundary_qualification import initial_body, tool_def
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg, execution_strategy=args.strategy)
    output = dict(decision='NOT PASS', strategy=args.strategy, cases=[])
    counts = dict(target=0, forbidden=0, effects=0)
    eof_entered, eof_release = Event(), Event()
    start = time.perf_counter()
    def save(phase):
        output.update(phase=phase, elapsed_s=time.perf_counter()-start, counts=dict(counts))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2)+'\n')
        print(output['elapsed_s'], phase, flush=True)
    def guard(frame, event, arg):
        if event != 'call': return
        name = frame.f_code.co_filename.replace('\\', '/')
        if name.endswith('/runtime/target_forward.py') and frame.f_code.co_name == 'forward': counts['target'] += 1
        if (name.endswith('/model_execution/language.py') and frame.f_code.co_name == '_forward'
            or '/omlx/patches/mlx_lm_mtp/' in name
            or '/omlx/patches/deepseek_v41/' in name and name.endswith(('mtp.py', 'dspark.py', 'language.py'))):
            counts['forbidden'] += 1
            raise RuntimeError('forbidden target execution: '+name)
    class WeatherEffect:
        name = 'lookup_weather'
        schema = tool_def()
        def run(self, arguments, context=None):
            counts['effects'] += 1
            # A real first-party registry effect, with an observable append-only
            # test file. The application ledger alone authorizes and deduplicates.
            with args.output.with_suffix('.effects.jsonl').open('a') as f:
                f.write(json.dumps(dict(session=context.session_id, arguments=arguments))+'\n')
            return ToolResult(json.dumps(dict(city=arguments['city'], weather='sunny 21C')), dict(ok=True))
    async def run():
        save('load')
        await backend._call(backend.load)
        await backend._call(lambda: sys.setprofile(guard))
        sock = socket.socket(); sock.bind(('127.0.0.1',0)); sock.listen(128)
        url = 'http://127.0.0.1:'+str(sock.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(create_app(backend=backend, runtime_config=cfg), lifespan='off', log_level='warning'))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if task.done(): await task
            await asyncio.sleep(.01)
        old_registry = web.registry_from_env
        web.registry_from_env = lambda: ToolRegistry([WeatherEffect()])
        try:
            app = web.create_app(runtime_base_url=url)
        finally:
            web.registry_from_env = old_registry
        async with httpx.AsyncClient(base_url=url, timeout=600) as client, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1', timeout=600) as effects:
            async def request(rec, body, sequence, *, partial=False):
                raw = json.dumps(body,separators=(',',':')).encode()
                headers = {'X-DS41F-Request-Sequence':str(sequence), 'Content-Type':'application/json'}
                path = f'/v1/sessions/{rec.session_id}/chat/completions'
                before_target = counts['target']
                prefix = b''
                if partial:
                    async def receive():
                        nonlocal prefix
                        async with client.stream('POST',path,content=raw,headers=headers) as response:
                            assert response.status_code == 200, await response.aread()
                            async for chunk in response.aiter_raw():
                                prefix += chunk
                                if ((partial == 'completed-tool' and b'"finish_reason":"tool_calls"' in prefix)
                                        or (partial not in ('completed-tool','worker-eof') and prefix.count(b'data:') >= 3)):
                                    break
                    if partial == 'worker-eof':
                        reader = asyncio.create_task(receive())
                        route_entered = asyncio.Event()
                        original_cancel = backend.cancel_stateful_stream
                        async def observed_cancel(session_id, request_id):
                            # Disconnect itself may already set stream.cancel.
                            # Observe entry of the real explicit route separately;
                            # do not release EOF just because disconnect won.
                            active = rec.active_stream
                            assert active is not None and active.request_id == request_id
                            route_entered.set()
                            return await original_cancel(session_id, request_id)
                        backend.cancel_stateful_stream = observed_cancel
                        try:
                            for _ in range(10000):
                                if eof_entered.is_set(): break
                                if reader.done(): await reader; raise AssertionError('EOF worker cut not reached')
                                await asyncio.sleep(.01)
                            assert eof_entered.is_set()
                            reader.cancel()
                            try: await reader
                            except asyncio.CancelledError: pass
                            stream = rec.active_stream
                            cancellation = asyncio.create_task(client.post(f'/v1/sessions/{rec.session_id}/cancel',
                                json={'request_id':stream.request_id}))
                            for _ in range(1000):
                                if route_entered.is_set() and stream.cancel.is_set(): break
                                if cancellation.done(): await cancellation
                                await asyncio.sleep(.01)
                            assert route_entered.is_set() and stream.cancel.is_set(), 'standard cancellation route did not reach protected worker'
                        finally:
                            eof_release.set()
                            backend.cancel_stateful_stream = original_cancel
                        reconciled = await cancellation
                        assert reconciled.status_code == 200, reconciled.text
                    else:
                        await receive()
                    for _ in range(10000):
                        if not rec.busy: break
                        await asyncio.sleep(.01)
                    assert not rec.busy
                else:
                    response = await client.post(path,content=raw,headers=headers)
                    assert response.status_code == 200, response.text
                    prefix = response.content
                slot = rec.response_reservation
                assert slot.state == 'completed'
                frozen = b''.join(slot.chunks)
                assert frozen.startswith(prefix)
                if partial == 'worker-eof':
                    # Reconstruct actual delivered/retried tool argument deltas,
                    # not just its [DONE] sentinel or byte-prefix certificate.
                    arguments = ''
                    finishes = []
                    for line in frozen.splitlines():
                        if not line.startswith(b'data: ') or line == b'data: [DONE]': continue
                        event = json.loads(line[6:])
                        for choice in event.get('choices',[]):
                            finishes.append(choice.get('finish_reason'))
                            for call in choice.get('delta',{}).get('tool_calls',[]):
                                arguments += call.get('function',{}).get('arguments','')
                    expected = rec.last_turn['response_json']['choices'][0]['message']['tool_calls'][0]['function']['arguments']
                    assert arguments == expected and 'tool_calls' in finishes
                    assert not rec.last_turn['cancelled']
                    output['worker_eof_disconnect'] = dict(exact_complete_tool_deltas=True,
                        semantic_terminal_preserved=True, worker_result_loss_reconciled=True)
                before = (rec.request_count, rec.m11.m8.frontier, counts['target'], counts['effects'])
                retry = await client.post(path,content=raw,headers=headers)
                assert retry.status_code == 200 and retry.content == frozen
                assert before == (rec.request_count, rec.m11.m8.frontier, counts['target'], counts['effects'])
                bad = await client.post(path,content=raw+b' ',headers=headers)
                assert bad.status_code in (400, 409), bad.text
                stale = await client.post(path,content=raw,headers=dict(headers, **{'X-DS41F-Request-Sequence':str(sequence-1)}))
                assert stale.status_code in (400, 409), stale.text
                assert before == (rec.request_count, rec.m11.m8.frontier, counts['target'], counts['effects'])
                metrics = getattr(rec.m11,'last_cycle_metrics',None)
                if metrics is not None:
                    assert counts['target']-before_target == 1+metrics['planned_target_inputs'], 'hidden target execution'
                diag = rec.m11.diagnostics()['m8']
                assert diag['all_cache_offsets_equal_frontier']
                assert diag['total_prompt_replay_count'] == diag['total_full_cache_repack_count'] == 0
                output['cases'].append(dict(sequence=sequence, partial=partial, stream=body.get('stream'),
                    exact_retry=True, mismatch_status=bad.status_code, stale_status=stale.status_code,
                    prefix_bytes=len(prefix), frozen_bytes=len(frozen), target_calls=counts['target']-before_target,
                    turn=rec.last_turn, diagnostics=diag, metrics=getattr(rec.m11,'last_cycle_metrics',None)))
                save('response completed')
                return rec.last_turn['response_json']['choices'][0]['message']
            async def close_application_session(rec):
                closed = await effects.delete('/api/session/'+rec.session_id)
                assert closed.status_code == 200, closed.text
                assert rec.response_reservation.state == 'retired'
                assert not rec.response_reservation.chunks and not rec.response_reservation.request_bytes
            try:
                # Tool call JSON and SSE on separate standard sessions.
                for streaming in (False, True, 'completed-tool', 'worker-eof'):
                    rec = await backend.create_stateful_session()
                    body = initial_body('Paris'); body.update(max_tokens=64, stream=bool(streaming))
                    save('tool '+('SSE' if streaming else 'JSON'))
                    from ds41f_mlx.runtime.tool_eof_preview import ToolEOFPreview
                    original_completed = ToolEOFPreview.completed
                    if streaming == 'worker-eof':
                        # Earlier cases release this shared gate in cleanup.
                        # Rearm it: otherwise wait() returns immediately and the
                        # supposed in-worker cut is only a timing-dependent race.
                        eof_entered.clear()
                        eof_release.clear()
                        def held_completed(oracle, ordinal):
                            complete = original_completed(oracle, ordinal)
                            if complete and not eof_entered.is_set():
                                eof_entered.set()
                                assert eof_release.wait(30), 'EOF worker cut release deadline'
                            return complete
                        ToolEOFPreview.completed = held_completed
                    try:
                        msg = await request(rec,body,1,partial=streaming if streaming in ('completed-tool','worker-eof') else False)
                    finally:
                        eof_release.set()
                        ToolEOFPreview.completed = original_completed
                    assert rec.last_turn['finish_reason'] == 'tool_calls' and msg['tool_calls']
                    output['tool_certificate'] = rec.last_turn['reconstruction']
                    save('tool completion / effect authorization')
                    payload = dict(session_id=rec.session_id, request_count=rec.request_count)
                    before_effect = counts['effects']
                    if streaming == 'completed-tool':
                        class DropOnce(httpx.AsyncBaseTransport):
                            def __init__(self): self.inner = httpx.ASGITransport(app=app)
                            async def handle_async_request(self, req):
                                res = await self.inner.handle_async_request(req)
                                await res.aclose()
                                raise httpx.ReadError('injected effect-response delivery loss')
                        async with httpx.AsyncClient(transport=DropOnce(), base_url='http://127.0.0.1') as dropped:
                            try:
                                await dropped.post('/api/tools',json=payload)
                                raise AssertionError('delivery loss not injected')
                            except httpx.ReadError:
                                pass
                        assert counts['effects'] == before_effect + 1
                    effect = await effects.post('/api/tools',json=payload)
                    output['effect_response'] = dict(status=effect.status_code, body=effect.json())
                    save('effect result')
                    assert effect.status_code == 200, effect.text
                    before = counts['effects']
                    repeated = await effects.post('/api/tools',json=payload)
                    observed = await effects.get('/api/tools/result',params=payload)
                    assert repeated.json() == effect.json() == observed.json()['result']
                    assert counts['effects'] == before == before_effect + 1
                    output.setdefault('effect_cases', []).append(dict(partial=streaming in ('completed-tool','worker-eof'),
                        delivery_loss=streaming == 'completed-tool', executions=1, duplicate_effects=0))
                    continuation = dict(body, messages=body['messages']+[msg]+effect.json()['messages'], tool_choice='auto')
                    save('tool result continuation')
                    answer = await request(rec,continuation,2)
                    duplicate = await client.post(f'/v1/sessions/{rec.session_id}/chat/completions',content=json.dumps(continuation,separators=(',',':')).encode(),headers={'X-DS41F-Request-Sequence':'3'})
                    assert duplicate.status_code in (400,409), duplicate.text
                    again = dict(continuation,messages=continuation['messages']+[answer,dict(role='user',content='Reply OK only.')])
                    save('repeated assistant generation')
                    final = await request(rec,again,3)
                    assert rec.last_turn['finish_reason'] == 'stop'
                    if streaming == 'completed-tool':
                        second = dict(again, messages=again['messages']+[final,dict(role='user',content='Use lookup_weather for Berlin, then answer concisely.')],
                            tool_choice={'type':'function','function':{'name':'lookup_weather'}})
                        save('second tool cycle')
                        second_msg = await request(rec,second,4)
                        assert second_msg.get('tool_calls')
                        result = await effects.post('/api/tools',json=dict(session_id=rec.session_id,request_count=rec.request_count))
                        assert result.status_code == 200, result.text
                        save('second tool result continuation')
                        await request(rec,dict(second,messages=second['messages']+[second_msg]+result.json()['messages'],tool_choice='auto',max_tokens=16),5)
                    await close_application_session(rec)
                # A pre-completion tool disconnect must NOT authorize any effect.
                rec = await backend.create_stateful_session()
                body = initial_body('Paris'); body.update(max_tokens=64,stream=True)
                save('tool pre-completion partial SSE disconnect')
                await request(rec,body,1,partial=True)
                before = counts['effects']
                denied = await effects.post('/api/tools',json=dict(session_id=rec.session_id,request_count=rec.request_count))
                assert denied.status_code == 400 and counts['effects'] == before
                output['precompletion_effect_denied'] = denied.json()
                await close_application_session(rec)
                # Ordinary socket disconnect -> exact retry -> canonical re-entry -> generation.
                rec = await backend.create_stateful_session()
                body = dict(model=cfg.model_id,messages=[dict(role='system',content='A cache stores frequently used data. '*32),
                    dict(role='user',content='Count from 1 to 100 separated by spaces.')],temperature=0,reasoning_effort='none',max_tokens=32,stream=True)
                save('ordinary disconnect')
                msg = await request(rec,body,1,partial=True)
                assert rec.last_turn['reconstruction']['representable']
                follow = dict(body,messages=body['messages']+[msg,dict(role='user',content='Write YES exactly eight times separated by spaces and nothing else.')],max_tokens=16)
                save('ordinary re-entry / new generation')
                await request(rec,follow,2)
                await close_application_session(rec)
                output['decision'] = 'TOOL / EFFECT / RETRY / ORDINARY REENTRY SOCKET PROBE PASS'
                output['duplicate_effects'] = 0
            finally:
                for sid in list(backend.sessions):
                    await backend.close_stateful_session(sid)
        server.should_exit = True
        await task
        sock.close()
        await backend._call(lambda: sys.setprofile(None))
        admission = backend._runtime.admission
        child = getattr(backend._model.language_model,'_ds41f_proposal_child',None)
        output['idle_resources'] = None if child is None else dict(receipts=len(child.receipts),producers=len(child.producers))
        if child is not None:
            assert not child.receipts and not child.producers
        await backend._call(backend.close)
        output['retired'] = dict(parent=not admission.active,child=None if child is None else not child.active,
            receipts=None if child is None else len(child.receipts), producers=None if child is None else len(child.producers))
    try:
        asyncio.run(run())
    except BaseException:
        output['error'] = traceback.format_exc()
        raise
    finally:
        if backend._runtime is not None and 'retired' not in output:
            admission = backend._runtime.admission
            child = getattr(backend._model.language_model,'_ds41f_proposal_child',None)
            backend.close()
            output['retired'] = dict(parent=not admission.active, child=None if child is None else not child.active,
                receipts=None if child is None else len(child.receipts), producers=None if child is None else len(child.producers))
        else:
            backend.close()
        save('complete')


if __name__ == '__main__': main()
