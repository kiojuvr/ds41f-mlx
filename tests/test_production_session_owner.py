"""Scheduler publication occurs after retirement, independently of delivery."""
from collections import deque
from time import perf_counter
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from test_production_mtp import pair, MX


def test_missing_session_pair_never_reprefills_or_enters_native_start():
    pytest.importorskip('omlx'); pytest.importorskip('mlx.core')
    from ds41f_mlx.serving.production_mtp import ProductionScheduler
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    scheduler = object.__new__(ProductionScheduler)
    errors = []
    request = NS(prompt_token_ids=list(range(212)), logical_session=NS(session_id='logical', canonical=tuple(range(210))),
                 execution=QualificationSession('r'), trace={}, prepared=None, recipe_tokenizer=None)
    scheduler.waiting = deque([request]); scheduler.running = {}
    scheduler.checkpoints = NS(acquire_retained=lambda *a, **kw:None)
    scheduler.session_leases = {'logical': (1, 'known-hash')}
    scheduler._retire = lambda req, *, error:errors.append(error)
    def forbidden(*a, **kw):
        raise AssertionError('missing pair caused native prompt replay')
    scheduler.backend = NS(_start=forbidden)
    scheduler._schedule_waiting()
    assert len(errors) == 1 and 'prompt replay forbidden' in str(errors[0])


def test_session_outcome_is_published_only_by_scheduler_after_native_retirement(monkeypatch):
    pytest.importorskip('omlx'); pytest.importorskip('mlx.core')
    from omlx.request import Request, SamplingParams
    from ds41f_mlx.serving.production_mtp import ProductionScheduler
    from ds41f_mlx.serving.production_sessions import ProductionSessions
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    scheduler = object.__new__(ProductionScheduler)
    scheduler.waiting, scheduler.running, scheduler.requests = deque(), {}, {}
    scheduler.request_id_to_uid, scheduler.uid_to_request_id = {}, {}
    scheduler._pending_abort_ids, scheduler._completed = set(), []
    scheduler.batch_generator = None
    scheduler.total_prompt_tokens = scheduler.total_completion_tokens = 0
    scheduler.checkpoints = PairedCheckpointAuthority('identity', MX)
    scheduler.session_leases = {}
    stages = []
    backend = NS(fatal_error=None)
    sessions = backend.production_sessions = ProductionSessions(backend)
    logical = sessions.create('logical-only')
    body = {'messages':[{'role':'user','content':'Hi'}]}
    prepared = NS(token_ids=list(range(202)), protocol='chat_completions', image_sources=[], capacity={})
    appid = str(uuid4())
    sessions.reserve(logical.session_id, prepared, body, 0, appid, 'r1')
    def start(rec, prepared, tokenizer, trace, *, checkpoint_capture, batch_generator_factory):
        stages.append('Scheduler/P5')
        rec.cache, rec.rings, rec.canonical = pair(200)
        checkpoint_capture(rec.cache, rec.rings, rec.canonical)
        scheduler.batch_generator = NS()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    def advance(rec, trace):
        assert logical.request_count == 0 and logical.publication['last_turn'] is None
        stages.append('Scheduler/decode')
    def settle(rec, trace):
        stages.append('Scheduler/settle')
        rec.cache, rec.rings, rec.canonical = pair(210)
        rec.owner = None
        rec.certificate = {'representable':True,'executable_tools':False}
        trace.update(quiescence={}, canonical_frontier=210, target_offsets=[210]*40, dspark_offsets=[210]*3,
                     prompt_replay=0, full_cache_repack=0, response={'choices':[{
                         'finish_reason':'stop','message':{'role':'assistant','content':'OK'}}]})
    def retire(rec):
        stages.append('native retirement')
        assert logical.request_count == 0
        rec.cache = rec.rings = None
    native_publish = sessions.publish
    def publish(req, error):
        stages.append('canonical publication')
        assert stages[-2] == 'native retirement'
        native_publish(req, error)
    sessions.publish = publish
    backend._start, backend._advance_application, backend._settle, backend._retire = start, advance, settle, retire
    scheduler.backend = backend
    request = Request(request_id='r1', prompt=prepared.token_ids, sampling_params=SamplingParams())
    request.prompt_token_ids = prepared.token_ids
    request.execution = QualificationSession('r1')
    request.logical_session = logical
    request.prepared, request.recipe_tokenizer = prepared, None
    request.session_body, request.application_id = body, appid
    request.session_pair_retained = False
    request.trace = dict(t0=perf_counter())
    request.prompt_checkpoint, request.delivery_cursor = None, 0
    scheduler.requests['r1'] = request; scheduler.waiting.append(request)
    scheduler.step()
    assert logical.request_count == 0
    request.trace['application_terminal'] = True
    output = scheduler.step()
    assert output.outputs[-1].finished and output.outputs[-1].error is None
    assert stages[-3:] == ['Scheduler/settle','native retirement','canonical publication']
    assert logical.request_count == 1 and logical.canonical == tuple(range(210))
    assert logical.publication['last_turn']['request_id'] == 'r1'
    assert request.execution.cache is None and not request.execution.canonical
    assert scheduler.session_leases.keys() == {'logical-only'}
    assert scheduler.checkpoints.acquire(list(range(212)))[2] == list(range(210))
    scheduler.release_session('logical-only')
    assert not scheduler.session_leases
    scheduler.checkpoints.clear()
