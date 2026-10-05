"""Real inline-image production conversation/state qualification (not R1).

Run initial, then --restore <initial receipt> in a fresh process. Receipts are
written before expensive work; a failed/incomplete run is never PASS.
"""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
from dataclasses import replace
import threading
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--restore', type=Path)
    args = p.parse_args()
    from tools.qualify_m48 import displaced
    import importlib.abc
    class NoDonor(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if displaced(fullname): raise AssertionError('displaced numerical import: '+fullname)
    sys.meta_path.insert(0,NoDonor())
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config(); cfg.apply_import_paths()
    from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    import mlx.core as mx
    import numpy as np
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    tokenizer = load_v41_tokenizer()
    result = dict(schema='ds41f.multimodal.qualification.v1', status='RUNNING', turns=[],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in ('ds41f_mlx/model_execution','ds41f_mlx/runtime','ds41f_mlx/prefill_fp8_mlx','ds41f_mlx/serving')
                       for p in (ROOT/folder).glob('*.py')})
    counts={}
    p7_events={}
    cancellation_phase={}
    phase_entered=threading.Event()
    discarded=[]
    def observe(frame,event,arg):
        if event=='return' and frame.f_code.co_qualname=='LivePrefillResult.discard':
            cache=frame.f_locals.get('cache') or ()
            discarded.append(dict(burned_layers=sum(bool(c._p6_append_failed and c._p6_append_invalid) for c in cache),
                certificate_detached='p6_commit_authority' not in frame.f_locals['setup'].__dict__))
        if event!='call': return
        name=frame.f_globals.get('__name__','')
        if displaced(name): raise AssertionError('displaced numerical call: '+name)
        if ((cancellation_phase.get('phase')=='encoding' and frame.f_code.co_qualname=='Model.encode_image_span')
                or (cancellation_phase.get('phase')=='prefill' and frame.f_code.co_qualname=='DenseP0P7PrefillSession.prefill')
                or (cancellation_phase.get('phase')=='stateful' and frame.f_code.co_qualname=='M11RecipeToolSession.run_current_assistant_turn')):
            phase_entered.set()
        if name=='ds41f_mlx.prefill_fp8_mlx.p7_scheduling' and frame.f_code.co_qualname=='SchedulingTelemetry.record':
            e=frame.f_locals['event']; fields=frame.f_locals['fields']
            p7_events[e]=p7_events.get(e,0)+1
            if e=='engram_consume': assert fields['logical_match'], 'foreground SSD fallback'
        if name.startswith('ds41f_mlx.model_execution.vision') or (name=='ds41f_mlx.model_execution.model' and frame.f_code.co_name=='encode_image_span'):
            key=name+':'+frame.f_code.co_qualname
            counts[key]=counts.get(key,0)+1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.output.write_text(json.dumps(result, indent=2)+'\n')
    def inventory(cache):
        return [dict(layer=i, slot=j, shape=list(x.shape), dtype=str(x.dtype),
                     sha256=hashlib.sha256(np.asarray(x.view(mx.uint8)).tobytes()).hexdigest())
                for i,c in enumerate(cache) for j,x in enumerate(c.cache)]
    async def slots(cache):
        # CPU-loaded arrays retain their creating stream. Read/hash operations
        # belong to the same numerical worker, never a new cross-thread graph.
        return await backend._call(inventory,cache)
    def memory():
        import psutil
        process=psutil.Process()
        return dict(active=mx.get_active_memory(),cache=mx.get_cache_memory(),peak=mx.get_peak_memory(),
                    rss=process.memory_info().rss,threads=process.num_threads(),fds=process.num_fds(),
                    swap=psutil.swap_memory().used,system_available=psutil.virtual_memory().available)
    def prepare(messages, max_tokens=128):
        return prepare_request('chat_completions', json.dumps(dict(model=cfg.model_id,
            messages=messages, reasoning_effort='none', temperature=0, max_tokens=max_tokens)).encode(), tokenizer=tokenizer)
    def image(name):
        path = cfg.checkpoint_path/'inference/examples/images'/name
        data = path.read_bytes()
        result.setdefault('image_assets', []).append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest()))
        return dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(data).decode()))
    save()
    async def run():
        t = perf_counter(); await backend._call(backend.load)
        result['load_seconds'] = perf_counter()-t
        result['admission'] = backend._runtime.admission.describe()
        result['loaded_memory'] = memory(); save()
        await backend._call(lambda:sys.setprofile(observe))
        if args.restore:
            original = json.loads(args.restore.read_text())
            assert original['status'] == 'PASS_INITIAL'
            rec = await backend.restore_stateful_session(artifact_path=Path(original['artifact']['path']), tokenizer=tokenizer)
            m8 = rec.m11.m8
            assert await slots(m8.live_cache) == original['saved_slots']
            assert m8.token_history == original['saved_history']
            assert rec.m11.image_identities == original['image_identities']
            messages=json.loads(json.dumps(original['canonical_probe_messages']))
            for message in messages:
                if isinstance(message.get('content'),list):
                    for part in message['content']:
                        if part.get('type')=='image_url':
                            part.update(image(part['image_url']['url'].removeprefix('fixture:')))
            live_list=m8.live_cache
            turn=await backend.run_stateful_chat_turn(rec.session_id,prepare(messages),tokenizer=tokenizer)
            tokens=list(turn.generated_tokens)
            assert tokens == original['probe_tokens']
            assert await slots(m8.live_cache) == original['probe_slots']
            assert m8.live_cache is live_list and rec.m11.last_image_encoded_count==0
            result.update(restore_exact=True, canonical_probe_turn=turn.to_json(),probe_tokens=tokens,probe_slots=await slots(m8.live_cache),
                          no_image_reencode_calls=dict(counts),prompt_replay_count=m8.total_prompt_replay_count,full_cache_repack_count=m8.total_full_cache_repack_count)
            assert not counts
            messages.append(dict(role='assistant',content=turn.response_json['choices'][0]['message']['content']))
            messages.append(dict(role='user',content=[image('corn.jpeg'),dict(type='text',text='Is this the same food as the first image? Answer briefly.')]))
            added=await backend.run_stateful_chat_turn(rec.session_id,prepare(messages),tokenizer=tokenizer)
            assert m8.live_cache is live_list and rec.m11.last_image_encoded_count==1
            assert all(c.size()==m8.frontier for c in live_list)
            result['restored_additional_image_turn']=added.to_json()
            result['restored_additional_image_trace']=backend.session_traces[-1]
            result['status'] = 'PASS_RESTORE'
            return
        rec = await backend.create_stateful_session()
        messages = [dict(role='user', content=[image('corn.jpeg'),dict(type='text',text='Identify the food in this image. Answer in one short sentence.')])]
        live_list = None
        for i in range(3):
            t = perf_counter(); request = prepare(messages)
            prep = perf_counter()-t
            turn = await backend.run_stateful_chat_turn(rec.session_id, request, tokenizer=tokenizer)
            m8 = rec.m11.m8
            assert m8.state == 'idle'
            if live_list is None: live_list = m8.live_cache
            assert m8.live_cache is live_list
            assert all(c.size() == m8.frontier for c in live_list)
            assert m8.total_prompt_replay_count == m8.total_full_cache_repack_count == 0
            result['turns'].append(dict(index=i, preprocessing_seconds=prep, turn=turn.to_json(),
                trace=backend.session_traces[-1], memory=memory(), image_identities=rec.m11.image_identities))
            save()
            response = turn.response_json['choices'][0]['message']
            messages.append(dict(role='assistant',content=response.get('content') or ''))
            if i == 0:
                messages.append(dict(role='user',content='What color is that food? Answer briefly.'))
            elif i == 1:
                messages.append(dict(role='user',content=[image('carrots.jpeg'),dict(type='text',text='Identify this second food and contrast its color with the first. One short sentence.')]))
        # Identity change is rejected before any KV mutation.
        before = await slots(live_list)
        changed = json.loads(json.dumps(messages))
        changed[0]['content'][0] = image('carrots.jpeg')
        changed.append(dict(role='user',content='Continue.'))
        try:
            await backend.run_stateful_chat_turn(rec.session_id, prepare(changed), tokenizer=tokenizer)
        except ValueError:
            pass
        else: raise AssertionError('changed historical image accepted')
        assert await slots(live_list) == before
        result['changed_image_rejected_without_mutation'] = True
        # Persist a completed ordinary conversation, then compare canonical re-entry
        # against the same turn in a genuinely fresh process (no hidden replay).
        result['artifact'] = await backend.persist_stateful_session(rec.session_id, artifact_root=cfg.kv_root/'vision-qualification')
        result['saved_slots'] = await slots(live_list)
        result['saved_history'] = list(m8.token_history)
        result['image_identities'] = rec.m11.image_identities
        messages.append(dict(role='user',content='Which of the two foods is orange? Answer briefly.'))
        template=json.loads(json.dumps(messages)); names=iter(('corn.jpeg','carrots.jpeg'))
        for message in template:
            if isinstance(message.get('content'),list):
                for part in message['content']:
                    if part.get('type')=='image_url': part['image_url']['url']='fixture:'+next(names)
        result['canonical_probe_messages']=template
        probe=await backend.run_stateful_chat_turn(rec.session_id,prepare(messages),tokenizer=tokenizer)
        result['probe_tokens']=list(probe.generated_tokens)
        result['probe_slots']=await slots(m8.live_cache)
        result['canonical_probe_turn']=probe.to_json()
        assert rec.m11.last_image_encoded_count==0 and m8.live_cache is live_list
        messages.append(dict(role='assistant',content=probe.response_json['choices'][0]['message']['content']))
        # Cancellation is between complete target transactions, with same-list idle return.
        cancel_request = prepare(messages + [dict(role='user',content='List twenty vegetables, numbered, without explanation.')])
        await backend._call(lambda: rec.m11.continue_from_prepared(cancel_request,max_tokens=32))
        report = await backend._call(m8.next_token)
        assert report is not None and report.finish_reason is None
        cancelled_tokens = [report.token]
        await backend._call(m8.ensure_idle,'vision_qualification_cancel')
        assert m8.live_cache is live_list
        assert all(c.size() == m8.frontier for c in live_list)
        result['cancelled_tokens'] = cancelled_tokens
        # Affected ordinary serving path after multimodal session.
        request = prepare([dict(role='user',content='What is two plus two? Answer briefly.')])
        async for _ in backend.infer(request): pass
        result['text_trace'] = backend.last_trace.to_json()
        assert result['text_trace']['cleanup_called'] and backend.active_generation_sessions == 0
        # Actual async production stream cancellation, not a model-forward smoke.
        stream_start=perf_counter()
        cancel_stream = backend.infer(prepare([dict(role='user',content=[image('corn.jpeg'),dict(type='text',text='Describe the image in detail.')])]))
        seen = 0
        stream_timing={}
        try:
            async for chunk in cancel_stream:
                seen += 1
                if seen==1: stream_timing['ready_seconds']=perf_counter()-stream_start
                if seen==2: stream_timing['first_token_seconds']=perf_counter()-stream_start
                if seen == 4: break
        finally: await cancel_stream.aclose()
        result['cancel_trace'] = backend.last_trace.to_json()
        result['stateless_image_stream_timing']=stream_timing
        assert result['cancel_trace']['cancelled'] and result['cancel_trace']['cleanup_called']
        assert backend.active_generation_sessions == 0
        # Async cancellation inside protected Vision encoding: drain the same
        # worker before lock release, discard temporary rows, then ordinary use.
        phase_entered.clear();cancellation_phase['phase']='encoding'
        cancelled_encoding=backend.infer(prepare([dict(role='user',content=[image('carrots.jpeg'),dict(type='text',text='Describe this food.')])]))
        pending=asyncio.create_task(anext(cancelled_encoding))
        while not phase_entered.is_set() and not pending.done(): await asyncio.sleep(.001)
        assert phase_entered.is_set()
        pending.cancel()
        try: await pending
        except asyncio.CancelledError: pass
        else: raise AssertionError('encoding cancellation not acknowledged')
        await cancelled_encoding.aclose();cancellation_phase.clear()
        result['encoding_cancel_trace']=backend.last_trace.to_json()
        assert result['encoding_cancel_trace']['cancelled'] and result['encoding_cancel_trace']['cleanup_called']
        assert not backend._lock.locked() and backend.active_generation_sessions==0
        phase_entered.clear();cancellation_phase['phase']='prefill'
        cancelled_prefill=backend.infer(prepare([dict(role='user',content=[image('corn.jpeg'),dict(type='text',text='Describe this food.')])]))
        pending=asyncio.create_task(anext(cancelled_prefill))
        while not phase_entered.is_set() and not pending.done(): await asyncio.sleep(.001)
        assert phase_entered.is_set();pending.cancel()
        try: await pending
        except asyncio.CancelledError: pass
        else: raise AssertionError('prefill cancellation not acknowledged')
        await cancelled_prefill.aclose();cancellation_phase.clear()
        result['prefill_cancel_trace']=backend.last_trace.to_json()
        result['discarded_prefills']=discarded
        assert discarded==[dict(burned_layers=40,certificate_detached=True)]
        assert not backend._lock.locked() and backend.active_generation_sessions==0
        # Existing stateful policy commits the completed turn before emitting its
        # protocol record, even when the waiting caller is cancelled mid-worker.
        phase_entered.clear();cancellation_phase['phase']='stateful'
        # The preceding core cancellation intentionally created a partial turn.
        # Use a fresh ordinary session for this protocol-level cancellation case.
        cancelled_rec=await backend.create_stateful_session()
        request=prepare([dict(role='user',content=[image('corn.jpeg'),dict(type='text',text='Describe the image in detail.')])],64)
        pending=asyncio.create_task(backend.run_stateful_chat_turn(cancelled_rec.session_id,request,tokenizer=tokenizer))
        while not phase_entered.is_set() and not pending.done(): await asyncio.sleep(.001)
        assert phase_entered.is_set();pending.cancel()
        try: await pending
        except asyncio.CancelledError: pass
        else: raise AssertionError('stateful cancellation not acknowledged')
        cancellation_phase.clear()
        assert not cancelled_rec.busy and cancelled_rec.last_turn is not None
        assert cancelled_rec.m11.m8.state=='idle' and cancelled_rec.request_count==1
        result['stateful_cancel_record']=cancelled_rec.to_json()
        result['stateful_cancel_trace']=backend.session_traces[-1]
        result['after_cancel_memory']=memory()
        # Representative maximum geometry/count/context, on canonical serving.
        from PIL import Image
        from io import BytesIO
        content = []
        variants = []
        for i,size in enumerate(((2048,2048),(1024,2048),(2048,1024),(128,128))):
            path=cfg.checkpoint_path/'inference/examples/images'/('corn.jpeg' if i%2==0 else 'carrots.jpeg')
            with Image.open(path) as original: variant=original.convert('RGB').resize(size)
            b=BytesIO(); fmt='WEBP' if i==3 else 'PNG'; variant.save(b,format=fmt)
            data=b.getvalue()
            variants.append(dict(size=list(size),format=fmt,sha256=hashlib.sha256(data).hexdigest()))
            content += [dict(type='text',text=f'Image {i+1}: '),dict(type='image_url',image_url=dict(url=f'data:image/{fmt.lower()};base64,'+base64.b64encode(data).decode()))]
        content.append(dict(type='text',text='Identify each food in order, with its color.'))
        maximum_messages=[dict(role='user',content=content)]
        maximum=prepare(maximum_messages,32)
        # The filler is tokenizer/recipe construction, not hidden prefix replay.
        gap=8160-len(maximum.token_ids)
        for _ in range(5):
            content[-1]['text']='Background notes: '+' note'*gap+' Identify each food in order, with its color.'
            maximum=prepare(maximum_messages,1)
            difference=8160-len(maximum.token_ids)
            if difference==0: break
            gap+=difference
        assert len(maximum.token_ids)==8160
        preprocessing_start=perf_counter();maximum=prepare(maximum_messages,32)
        maximum_preprocessing_seconds=perf_counter()-preprocessing_start
        matrix_rec=await backend.create_stateful_session()
        t=perf_counter()
        matrix_turn=await backend.run_stateful_chat_turn(matrix_rec.session_id,maximum,tokenizer=tokenizer)
        matrix_m8=matrix_rec.m11.m8
        def reach_ceiling():
            remaining=8192-matrix_m8.frontier
            if remaining:
                matrix_m8.config=replace(matrix_m8.config,stop_token_ids=())
                matrix_m8.begin_turn_from_suffix([tokenizer.encode(' note')[0]],max_tokens=max(1,remaining-1))
                if remaining>1: list(matrix_m8.generation.generate(remaining-1))
                matrix_m8.ensure_idle()
            assert matrix_m8.frontier==8192
            assert all(c.size()==8192 for c in matrix_m8.live_cache)
        await backend._call(reach_ceiling)
        result['maximum_matrix']=dict(variants=variants,spans=maximum.multimodal.identities(),preprocessing_seconds=maximum_preprocessing_seconds,
            initial_prompt_tokens=8160,consumed_frontier=matrix_m8.frontier,elapsed_seconds=perf_counter()-t,
            turn=matrix_turn.to_json(),trace=backend.session_traces[-1],memory=memory(),diagnostics=matrix_rec.m11.diagnostics())
        assert matrix_m8.total_prompt_replay_count==matrix_m8.total_full_cache_repack_count==0
        result['status'] = 'PASS_INITIAL'
    try:
        asyncio.run(run())
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc)); raise
    finally:
        backend.close()
        result['owned_vision_calls']=counts
        result['p7_events']=p7_events
        result['displaced_numerical_imports_and_calls_forbidden']=True
        result['closed_memory'] = memory()
        result['active_generation_sessions_after_close'] = backend.active_generation_sessions
        if 'admission' in result:
            admission=result['admission']
            old_cache=mx.set_cache_limit(admission['previous_allocator_cache_limit_bytes'])
            old_wired=mx.set_wired_limit(admission['previous_wired_limit_bytes'])
            result['retired_admission']=backend._runtime.admission.describe()
            result['resource_policy_restored_exactly']=(old_cache==admission['previous_allocator_cache_limit_bytes'] and old_wired==admission['previous_wired_limit_bytes'] and not result['retired_admission']['active'])
            if not result['resource_policy_restored_exactly']: result.update(status='FAIL',error='resource retirement mismatch')
        save()


if __name__ == '__main__': main()
