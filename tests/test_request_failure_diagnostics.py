"""Deterministic ordinary-worker failures; no model, replay, or tool effects."""
from collections import deque
import json
import logging
import os
from time import perf_counter
from types import SimpleNamespace as NS
from urllib.request import urlopen

import pytest

from ds41f_mlx.observability import Projection
from ds41f_mlx.operator_dashboard import Dashboard
from ds41f_mlx.runtime_logging import History, RuntimeHandler

SECRET = 'PRIVATE_PROMPT tool-result data:image/base64 /Users/private/attachment.png'


@pytest.mark.parametrize('native,cleanup', [(False, False), (False, True), (True, False), (True, 'clear')])
def test_worker_failure_diagnostics(monkeypatch, native, cleanup):
    from omlx.request import Request, SamplingParams
    from ds41f_mlx.serving.production_mtp import ProductionScheduler, ProductionMTPBackend
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    from ds41f_mlx import execution_identity
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.server import create_app
    from fastapi.testclient import TestClient

    p = Projection()
    p.lifecycle('READY')
    p.begin('http-id')
    s = object.__new__(ProductionScheduler)
    s.waiting, s.running, s.requests = deque(), {}, {}
    s.request_id_to_uid, s.uid_to_request_id = {}, {}
    s._pending_abort_ids, s._completed = set(), []
    s.batch_generator = None
    s.session_leases = {}
    s.shutdown = lambda: None
    s.total_prompt_tokens = s.total_completion_tokens = 0
    s.checkpoints = PairedCheckpointAuthority('identity', NS(eval=lambda *a: None))
    calls = []
    def fail():
        raise ValueError(SECRET)
    def acquire(*a, **kw):
        calls.append('lookup')
        if native:
            return None
        kw['admit'](None)
    monkeypatch.setattr(s.checkpoints, 'acquire', acquire)
    monkeypatch.setattr(execution_identity, 'numerical_profile', fail)
    def start(*a, **kw):
        calls.append('native')
        raise RuntimeError(SECRET)
    def retire(rec):
        calls.append('retire')
        if cleanup is True:
            raise LookupError(SECRET)
    previous = dict(os.environ)
    b = ProductionMTPBackend(runtime_config=config(profile='mtp-serving-v1'))
    b.dependency_identity = 'identity'
    b.telemetry, b.scheduler, b._model = p, s, object()
    monkeypatch.setattr(b, '_start', start)
    monkeypatch.setattr(b, '_retire', retire)
    s.backend = b
    original_clear = s.checkpoints.clear
    if cleanup == 'clear':
        def bad_clear():
            raise LookupError(SECRET)
        monkeypatch.setattr(s.checkpoints, 'clear', bad_clear)
    mm = NS(spans=[NS(grid=(100000, 100000))], release_sources=lambda: None)
    r = Request(request_id='scheduler-id', prompt=[1, 2, 3], sampling_params=SamplingParams())
    r.prompt_token_ids = [1, 2, 3]
    r.execution = QualificationSession(r.request_id)
    r.trace = dict(t0=perf_counter(), diagnostic_id='http-id')
    r.prepared = NS(protocol='messages', multimodal=None if native else mm)
    r.recipe_tokenizer = None
    r.image_identities, r.native_started = (), False
    r.prompt_checkpoint, r.delivery_cursor = None, 0
    s.waiting.append(r)
    s.requests[r.request_id] = r
    try:
        app = create_app(profile='mtp-serving-v1', runtime_config=b.runtime_config, backend=b)
        result = s.step()
        assert result.outputs[-1].finished and result.outputs[-1].error
        assert calls == ['lookup', 'native', 'retire'] if native else calls == ['lookup', 'retire']
        snap = p.snapshot()
        f = snap['failures'][0]
        assert f['request_id'] == 'http-id' and f['scheduler_request_id'] == 'scheduler-id'
        assert f['stage'] == ('native_start' if native else 'admission')
        assert f['source']['module'] == 'ds41f_mlx.serving.production_mtp'
        assert f['source']['function'] == ('_schedule_waiting' if native else 'admit')
        assert f['source']['line'] > 0
        assert f['exception_type'] == ('RuntimeError' if native else 'ValueError')
        assert f['native_started'] == f['runtime_fatal'] == native
        assert not f['canonical_settled'] and not f['cache_published']
        if cleanup:
            assert snap['failures'][1]['secondary']
            assert snap['failures'][1]['exception_type'] == 'LookupError'
        assert not s.requests and not s.waiting and not s.running and not s.checkpoints._payloads
        assert r.execution.cache is None and r.execution.owner is None
        assert s.last_settlement.get('prompt_replay') is None
        assert s.last_settlement.get('full_cache_repack') is None
        monkeypatch.setattr(s.checkpoints, 'clear', original_clear)
        with TestClient(app, base_url='http://127.0.0.1:8000') as client:
            h = client.get('/health')
            assert h.status_code == (503 if native else 200)
            assert h.json()['status'] == ('unavailable' if native else 'ready')
        assert Dashboard(snap).state == ('FAILED' if native else 'READY')
        assert snap['last_request']['finish_reason'] == 'error'
        assert SECRET not in json.dumps(snap)
    finally:
        b._model = None
        b.close()
        os.environ.clear()
        os.environ.update(previous)


def test_local_api_retention_without_output(monkeypatch):
    from ds41f_mlx.operator_control import Control
    import socket
    import sys
    p, history = Projection(), History(capacity=8)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    with open(os.devnull, 'w') as null:
        monkeypatch.setattr(sys, 'stdout', null)
        monkeypatch.setattr(sys, 'stderr', null)
        handler = RuntimeHandler(history, null)
        logger = logging.getLogger('ds41f.operator.runtime')
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.INFO)
        control = Control(p, port=port, logs=history)
        try:
            for i in range(140):
                rid = str(i)
                p.begin(rid)
                p.correlate(rid, 'scheduler-'+rid)
                try:
                    raise ValueError(SECRET)
                except ValueError as exc:
                    p.failure(rid, exc, stage='admission')
                p.finish(rid)
            def get(path):
                with urlopen(f'http://127.0.0.1:{port}/ds41f/{path}') as response:
                    return json.load(response)
            status, logs = get('status'), get('logs?after=0&limit=8')
            assert len(status['failures']) == 128
            assert status['failures'][0]['request_id'] == '12'
            assert status['failures'][-1]['scheduler_request_id'] == 'scheduler-139'
            assert status['requests'] == []
            assert len(logs['records']) == 8 and logs['gap']
            assert any('request failure' in r['message'] for r in logs['records'])
            assert SECRET not in json.dumps([status, logs])
        finally:
            control.close()
            handler.drain()
            logger.removeHandler(handler)
            logger.setLevel(old_level)


def test_failure_projection_is_best_effort():
    from ds41f_mlx.observability import failure
    class Broken:
        def failure(self, *a, **kw):
            raise RuntimeError('projection unavailable')
    failure(NS(telemetry=Broken()), {}, ValueError(SECRET))
