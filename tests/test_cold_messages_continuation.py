"""Normal C=0 admission and existing MTP lifecycle, no weights."""
from types import SimpleNamespace as NS
import pytest

from test_cold_messages_reconstruction import prepare_fixture, idle_scheduler
from test_production_mtp import pair
from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
from tools.qualify_cold_messages_continuation import verify_fixture


def test_p5_observation_uses_actual_dspark_offsets_property():
    from ds41f_mlx.runtime.mtp_lifecycle import DSparkCommittedContext
    from tools.qualify_cold_messages_continuation import p5_observation
    context = DSparkCommittedContext.from_native([NS(offset=212) for _ in range(3)],
        frontier=212, target_layer_ids=(0, 1, 2))
    batch = object()
    owner = NS(uid=17, _bg=batch, history=NS(prompt_tokens=(1, 2, 3)),
        active_cache_offsets=lambda: (213,)*40, dspark_context=context)
    result = p5_observation(NS(_state='started'), owner, NS(batch_generator=batch))
    assert result['dspark_offsets'] == (212,)*3
    assert result['target_offsets'] == (213,)*40
    assert result['uid'] == 17 and result['scheduler_batch']


def test_candidate_precedes_independent_reference(monkeypatch):
    import tools.qualify_prefill_fp8_mlx as reference_tool
    from tools.qualify_cold_messages_continuation import observe_continuation
    prepared, tokenizer, _, _ = prepare_fixture()
    calls = []
    def enqueue(*args):
        calls.append('enqueue')
        raise RuntimeError('controlled pre-native stop')
    backend = NS(load=lambda: calls.append('load'),
        _model=NS(language_model=NS()), _enqueue=enqueue)
    monkeypatch.setattr(reference_tool, 'ordinary_prefill_reference',
        lambda *a, **k: pytest.fail('reference ran before Candidate settlement'))
    with pytest.raises(RuntimeError, match='controlled pre-native stop'):
        observe_continuation(backend, prepared, tokenizer, {})
    assert calls == ['load', 'enqueue']
    prepared.multimodal.release_sources()


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


@pytest.mark.parametrize('fault', ['missing', 'modified', 'identity', 'budget', 'pixel', 'logical', 'target_only', 'short_pair', 'intersection', 'chat'])
def test_normal_admission_refuses_before_start(fault):
    from dataclasses import replace
    scheduler, request = queued()
    mm = request.prepared.multimodal
    if fault == 'missing': mm.release_sources()
    if fault == 'modified': mm._sources[2] = b'changed'
    if fault in ('identity', 'budget', 'pixel'):
        field = (dict(digest='changed') if fault == 'identity' else
                 dict(grid=(200, 200)) if fault == 'budget' else dict(source_pixels=17*1024**2))
        request.prepared.multimodal = replace(mm, spans=(replace(mm.spans[0], **field),))
    if fault == 'logical':
        request.logical_session = NS(canonical=(1,), session_id='lost')
        scheduler.backend.production_sessions = NS(publish=lambda *a: None)
    if fault == 'target_only': request.execution.cache = pair(211)[0]
    if fault == 'short_pair':
        from test_cold_messages_reconstruction import publish
        publish(scheduler, request.prepared, 2, images=())
    if fault == 'intersection':
        scheduler.checkpoints.acquire = lambda *a, **k: pair(190)[:2] + (request.prompt_token_ids[:190],)
    if fault == 'chat': request.prepared.protocol = 'chat'
    scheduler.backend._start = lambda *a, **k: pytest.fail('native start on invalid admission')
    scheduler._schedule_waiting()
    assert not request.native_started
    error = scheduler._completed[-1].error
    assert error
    if fault == 'logical': assert error == 'canonical session pair is not retained; no replay'
    if fault == 'intersection': assert 'intersects' in error
    if fault in ('budget', 'pixel'): assert 'budget exceeded' in error
    assert not scheduler.requests and not scheduler.waiting and not scheduler.running
    assert scheduler.backend.fatal_error is None
    assert not request.prepared.multimodal._sources
    scheduler.checkpoints.clear()


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


@pytest.mark.parametrize('fault', ['none', 'cancel', 'mutation', 'settlement', 'incomplete', 'post_handoff_observation',
    'registered_observation', 'generation_observation', 'publication_observation'])
def test_existing_start_scheduler_route(monkeypatch, fault):
    historical = run_start_scheduler_route(monkeypatch, fault, historical=True)
    fresh = run_start_scheduler_route(monkeypatch, fault, historical=False)
    # Deterministic same-schedule controls, not sampled numerical trajectories.
    assert historical == fresh


def run_start_scheduler_route(monkeypatch, fault, *, historical):
    import mlx.core as mx
    import importlib
    from ds41f_mlx import prefill_fp8_mlx as p6
    from ds41f_mlx.runtime import mtp_lifecycle as lifecycle
    scheduler, request = queued()
    verify_fixture(request.prepared)
    if not historical:
        request.prepared.image_is_historical = (False,)
        request.required_image_frontier = 0
    ids = tuple(request.prompt_token_ids)
    source = request.prepared.multimodal._sources[2]
    embeddings = NS(rows=[(2, mx.array([[1., 2., 3.]]))], seconds=0)
    transitions = []
    lm = NS(_config=NS(dspark_target_layer_ids=(0, 1, 2)), layers=[lambda h: h]*40,
            make_cache=lambda: pair(0)[0], make_mtp_cache=lambda: pair(0)[1])
    # Controlled math/native interfaces, NOT an alternative lifecycle. _start,
    # _schedule_waiting, step and _retire themselves remain the real methods.
    def append_hidden(hidden, rings, start_offset):
        for ring in rings: ring.offset = start_offset + hidden.shape[1]
    lm.dspark_append_context = append_hidden
    backend = object.__new__(ProductionMTPBackend)
    backend._model = NS(language_model=lm)
    backend.load = lambda: None
    backend._prefill_tap_start = lambda *a: 0
    def encode(mm, C):
        assert C == 0 and mm._sources[2] == source and tuple(mm.token_ids) == ids
        mm.validate_workload(C)
        transitions.append(('vision', C))
        return embeddings
    backend._encode_new_images = encode
    backend._retire = lambda rec: setattr(rec, 'owner', None)
    backend.fatal_error = None
    backend.make_sampler = lambda policy: None
    backend.omlx_path, backend.checkpoint = None, None
    backend.model_id = 'deepseek-v4.1-flash'
    scheduler.backend = backend
    def create(model, cache, ids, *, committed_frontier=0, **kw):
        C, end = committed_frontier, len(ids)
        assert tuple(ids) == tuple(request.prompt_token_ids[:end])
        assert kw['image_embeddings'] is embeddings
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
        owner = NS(uid=17, _bg=scheduler._new_batch(None), prompt_replay_count=0)
        if fault == 'post_handoff_observation':
            raise TypeError("'tuple' object is not callable")
        return owner
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
    diagnostic = {}
    def broken_optional():
        # Exact previous diagnostic error, never an execution operation.
        offsets = (212, 212, 212)
        return offsets()
    from tools.qualify_cold_messages_continuation import optional_observation
    if fault == 'registered_observation':
        scheduled, _ = scheduler._schedule_waiting()
        assert scheduled == [request]
        optional_observation(diagnostic, 'registered', broken_optional)
        assert request.execution.owner.uid == 17
        assert scheduler.request_id_to_uid == {'step2b': 17}
    if fault == 'cancel': scheduler._do_abort_request('step2b')
    else:
        result = scheduler.step()
        if fault not in ('mutation', 'post_handoff_observation'):
            assert not result.outputs or not result.outputs[-1].error, result.outputs[-1].error
            saved = request.prompt_checkpoint
            assert saved is not None, scheduler.last_settlement
            assert saved.tokens == tuple(request.prompt_token_ids[:-2])
            assert [c.size() for c in saved.target] == [211]*40
            assert [r.offset for r in saved.rings] == [211]*3
            assert [c.size() for c in request.execution.cache] == [212]*40
            assert [r.offset for r in request.execution.rings] == [212]*3
            assert not scheduler.checkpoints._payloads
            if fault == 'generation_observation':
                optional_observation(diagnostic, 'generated', broken_optional)
                assert request.execution.owner is not None
            scheduler.step()
            if fault == 'publication_observation':
                optional_observation(diagnostic, 'published', broken_optional)
                assert scheduler.checkpoints._payloads
    success = fault in ('none', 'registered_observation', 'generation_observation', 'publication_observation')
    if fault.endswith('_observation') and fault != 'post_handoff_observation':
        assert list(diagnostic['unavailable'].values()) == ['TypeError']
    assert not request.prepared.multimodal._sources
    assert not scheduler.requests and not scheduler.running and not scheduler.waiting
    assert not scheduler.request_id_to_uid and not scheduler.uid_to_request_id
    assert scheduler.batch_generator is None and request.execution.owner is None
    assert bool(scheduler.checkpoints._payloads) == success
    if success:
        assert transitions[:4] == [('vision', 0), ('append', 0, 211), ('append', 211, 212), ('P5', request.prompt_token_ids[-1])]
        assert scheduler.checkpoints.acquire(request.prompt_token_ids, images=request.image_identities)[2] == request.prompt_token_ids[:-2]
    if fault == 'cancel': assert transitions == []
    if fault == 'mutation': assert transitions == [('vision', 0), ('append', 0, 211)]
    if fault == 'post_handoff_observation':
        assert "'tuple' object is not callable" in backend.fatal_error
        assert transitions[-1] == ('P5', request.prompt_token_ids[-1])
        assert transitions.count(('append', 211, 212)) == 1
        assert request.prompt_checkpoint is None
    assert bool(backend.fatal_error) == (fault in ('mutation', 'settlement', 'incomplete', 'post_handoff_observation'))
    # Compare every controlled physical value, canonical input and lifecycle
    # transition across the two wire-origin classifications.
    pairs = [(s.tokens, s.images,
              tuple(tuple(v.tobytes() for v in c.cache) for c in s.target),
              tuple((r.offset, r.keys.tobytes()) for r in s.rings))
             for s in scheduler.checkpoints._payloads.values()]
    result = (ids, source, embeddings.rows[0][1].tolist(), transitions, pairs,
              request.trace.get('generation_policy'), request.trace.get('execution_sampling'),
              request.cached_tokens, bool(backend.fatal_error))
    scheduler.checkpoints.clear()
    return result
