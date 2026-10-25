"""Step 2B: narrow scope plus existing production lifecycle, no weights."""
from types import SimpleNamespace as NS
import pytest

from test_cold_messages_reconstruction import prepare_fixture, idle_scheduler
from test_production_mtp import pair
from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
from tools.qualify_cold_messages_continuation import cold_miss_scope, verify_fixture


def queued():
    from omlx.request import Request, SamplingParams
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    from time import perf_counter
    prepared, tokenizer, _, _ = prepare_fixture()
    scheduler = idle_scheduler()
    scheduler._pending_abort_ids = set()
    request = Request(request_id='step2b', prompt=prepared.token_ids, sampling_params=SamplingParams())
    request.prompt_token_ids = prepared.token_ids
    request.prepared, request.recipe_tokenizer = prepared, tokenizer
    request.execution = QualificationSession('step2b')
    request.logical_session = None
    request.required_image_frontier = 191
    request.image_identities = tuple(prepared.multimodal.identities())
    request.native_started = False
    request.prompt_checkpoint, request.delivery_cursor = None, 0
    request.trace = dict(t0=perf_counter(), response_id='step2b', generated=0, decode_s=0., first_canonical_s=None)
    scheduler.requests[request.request_id] = request
    scheduler.waiting.append(request)
    return scheduler, request


@pytest.mark.parametrize('fault', ['missing', 'modified', 'identity', 'budget', 'logical', 'lease', 'target_only', 'output_budget'])
def test_scope_refuses_before_start(fault):
    from dataclasses import replace
    scheduler, request = queued()
    mm = request.prepared.multimodal
    if fault == 'missing': mm.release_sources()
    if fault == 'modified': mm._sources[2] = b'changed'
    if fault in ('identity', 'budget'):
        field = dict(digest='changed') if fault == 'identity' else dict(grid=(200, 200))
        request.prepared.multimodal = replace(mm, spans=(replace(mm.spans[0], **field),))
    if fault == 'logical': request.logical_session = NS(canonical=(1,))
    if fault == 'lease': scheduler.session_leases['lost'] = (0, b'lost')
    if fault == 'target_only': request.execution.cache = pair(211)[0]
    if fault == 'output_budget': request.prepared.resolved_max_tokens = 2
    with pytest.raises(ValueError):
        with cold_miss_scope(scheduler, request):
            pytest.fail('invalid exception')
    assert not request.native_started and not scheduler.checkpoints._payloads
    assert '_historical_images_covered' not in scheduler.__dict__
    request.prepared.multimodal.release_sources()


def test_observer_requires_identity_before_load(monkeypatch):
    from ds41f_mlx import execution_identity
    from tools.qualify_cold_messages_continuation import observe_continuation
    calls = []
    backend = NS(load=lambda: calls.append('load'))
    def unbound():
        raise RuntimeError('numerical execution requires installed MTP admission')
    monkeypatch.setattr(execution_identity, 'numerical_profile', unbound)
    with pytest.raises(RuntimeError, match='installed MTP admission'):
        observe_continuation(backend, None, None, {})
    assert calls == []


@pytest.mark.parametrize('fault', ['source', 'identity', 'lease', 'scope_exit'])
def test_recheck_and_scope_exit_keep_normal_refusal(fault):
    scheduler, request = queued()
    with cold_miss_scope(scheduler, request):
        if fault == 'source': request.prepared.multimodal._sources[2] = b'changed'
        if fault == 'identity': request.image_identities = ()
        if fault == 'lease': scheduler.session_leases['late'] = (0, b'lost')
        if fault != 'scope_exit': scheduler._schedule_waiting()
    if fault == 'scope_exit': scheduler._schedule_waiting()
    assert not request.native_started
    assert scheduler._completed[-1].error
    assert not scheduler.checkpoints._payloads and scheduler.backend.fatal_error is None
    assert not request.prepared.multimodal._sources
    assert '_historical_images_covered' not in scheduler.__dict__


@pytest.mark.parametrize('fault', ['none', 'cancel', 'mutation', 'settlement', 'incomplete'])
def test_existing_start_scheduler_route(monkeypatch, fault):
    import mlx.core as mx
    import importlib
    from ds41f_mlx import prefill_fp8_mlx as p6
    from ds41f_mlx.runtime import mtp_lifecycle as lifecycle
    scheduler, request = queued()
    verify_fixture(request.prepared)
    transitions = []
    lm = NS(_config=NS(dspark_target_layer_ids=(0, 1, 2)), layers=[lambda h: h]*40,
            make_cache=lambda: pair(0)[0], make_mtp_cache=lambda: pair(211)[1])
    # Controlled math/native interfaces, NOT an alternative lifecycle. _start,
    # _schedule_waiting, step and _retire themselves remain the real methods.
    def append_hidden(hidden, rings, start_offset):
        for ring in rings: ring.offset = start_offset + hidden.shape[1]
    lm.dspark_append_context = append_hidden
    backend = object.__new__(ProductionMTPBackend)
    backend._model = NS(language_model=lm)
    backend.load = lambda: None
    backend._prefill_tap_start = lambda *a: 0
    backend._encode_new_images = lambda mm, C: (mm.validate_workload(C), transitions.append(('vision', C)), NS(rows=[(2, None)], seconds=0))[-1]
    backend._retire = lambda rec: setattr(rec, 'owner', None)
    backend.fatal_error = None
    backend.make_sampler = lambda policy: None
    backend.omlx_path, backend.checkpoint = None, None
    backend.model_id = 'deepseek-v4.1-flash'
    scheduler.backend = backend
    def create(model, cache, ids, *, committed_frontier=0, **kw):
        C, end = committed_frontier, len(ids)
        app = NS(commit_certificate=NS(cache=cache, end=end), final_execution=NS(runner=NS(full_cache_repack_count=0)))
        def execute():
            transitions.append(('append', C, end))
            if fault == 'mutation': raise RuntimeError('uncertain mutation')
            for layer in cache: layer.cache[0][0] = end
            for i in (0, 1, 2): lm.layers[i](mx.zeros((1, end-C, 1, 1)))
        app.execute_all = execute
        return app
    monkeypatch.setattr(p6.DeferredPrefillAppend, 'create', create)
    monkeypatch.setattr(p6.DeferredPrefillAppend, 'continue_from_commit', lambda model, cert, ids, **kw:
        create(model, cert.cache, ids, committed_frontier=cert.end, **kw))
    monkeypatch.setattr(p6.LivePrefillResult, 'from_committed', lambda *a, **kw: NS())
    monkeypatch.setattr(lifecycle.DSparkCommittedContext, 'from_native', lambda *a, **kw: NS())
    generate = importlib.import_module('mlx_lm.generate')
    monkeypatch.setattr(generate, 'BatchGenerator', lambda *a, **kw: NS(_generation_batch=NS(samplers=[], fallback_sampler=None)))
    def handoff(live, model, *, terminal_prompt_token, **kw):
        transitions.append(('P5', terminal_prompt_token))
        assert transitions[-2] == ('append', 211, 212)
        assert not scheduler.checkpoints._payloads
        return NS(uid=17, _bg=scheduler._new_batch(None), prompt_replay_count=0)
    monkeypatch.setattr(p6, 'handoff_to_generation', handoff)
    def settle(rec, trace):
        transitions.append('settle')
        if fault == 'settlement': raise RuntimeError('failed settlement')
        rec.cache, rec.rings, _ = pair(214)
        rec.canonical = request.prompt_token_ids + [42]
        rec.owner = None
        if fault != 'incomplete': trace['quiescence'] = {}
        trace['canonical_generated'] = [42]
        trace['response'] = dict(stop_reason='max_tokens')
    backend._settle = settle
    def advance(rec, trace):
        assert scheduler.request_id_to_uid == {'step2b': 17}
        assert scheduler.uid_to_request_id == {17: 'step2b'}
        assert rec.owner._bg is scheduler.batch_generator
        transitions.append('advance')
        # First worker turn leaves it live to observe pending publication.
        if transitions.count('advance') == 2:
            backend._settle(rec, trace)
            trace['application_terminal'] = True
    backend._advance_application = advance
    with cold_miss_scope(scheduler, request):
        if fault == 'cancel': scheduler._do_abort_request('step2b')
        else:
            result = scheduler.step()
            if fault != 'mutation':
                assert not result.outputs or not result.outputs[-1].error, result.outputs[-1].error
                assert request.prompt_checkpoint is not None, scheduler.last_settlement
                assert request.prompt_checkpoint.tokens == tuple(request.prompt_token_ids[:-2])
                assert not scheduler.checkpoints._payloads
                scheduler.step()
    assert '_historical_images_covered' not in scheduler.__dict__
    assert not request.prepared.multimodal._sources
    assert not scheduler.requests and not scheduler.running and not scheduler.waiting
    assert not scheduler.request_id_to_uid and not scheduler.uid_to_request_id
    assert scheduler.batch_generator is None and request.execution.owner is None
    assert bool(scheduler.checkpoints._payloads) == (fault == 'none')
    if fault == 'none':
        assert transitions[:4] == [('vision', 0), ('append', 0, 211), ('append', 211, 212), ('P5', request.prompt_token_ids[-1])]
        assert scheduler.checkpoints.acquire(request.prompt_token_ids, images=request.image_identities)[2] == request.prompt_token_ids[:-2]
    if fault == 'cancel': assert transitions == []
    if fault == 'mutation': assert transitions == [('vision', 0), ('append', 0, 211)]
    assert bool(backend.fatal_error) == (fault in ('mutation', 'settlement', 'incomplete'))
    scheduler.checkpoints.clear()
