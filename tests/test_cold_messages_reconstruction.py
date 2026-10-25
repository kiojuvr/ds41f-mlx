"""Step 2A: real Recipe/paired/Scheduler interfaces; no model weights."""
import base64
from collections import deque
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pytest

from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
from ds41f_mlx.serving.production_mtp import ProductionScheduler
from ds41f_mlx.serving.internal_mtp import QualificationSession
from tools import qualify_prefill_fp8_mlx as tool
from test_production_mtp import pair, MX

CHECKPOINT = Path('/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
DIGEST = '251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1'


def prepare_fixture():
    data = (CHECKPOINT/'inference/examples/images/corn.jpeg').read_bytes()
    assert sha256(data).hexdigest() == DIGEST
    body = dict(model='deepseek-v4.1-flash', max_tokens=1, temperature=0,
        messages=[dict(role='user', content=[dict(type='image', source=dict(type='base64',
            media_type='image/jpeg', data=base64.b64encode(data).decode())),
            dict(type='text', text='Identify this food in one short sentence.')]),
            dict(role='assistant', content='Corn.'),
            dict(role='user', content='State one visible feature.')])
    import sys
    tokenizer = load_v41_tokenizer(Path(sys.prefix)/'share/ds41f-mtp/recipe')
    prepared = prepare_request('messages', json.dumps(body).encode(), tokenizer=tokenizer,
                               checkpoint=CHECKPOINT, ordinary=True)
    return prepared, tokenizer, data, body


def idle_scheduler():
    scheduler = object.__new__(ProductionScheduler)
    scheduler.checkpoints = PairedCheckpointAuthority('identity', MX)
    scheduler.waiting, scheduler.running, scheduler.requests = deque(), {}, {}
    scheduler.session_leases = {}
    scheduler.request_id_to_uid, scheduler.uid_to_request_id = {}, {}
    scheduler._completed = []
    scheduler.batch_generator = None
    scheduler.total_prompt_tokens = scheduler.total_completion_tokens = 0
    scheduler.backend = SimpleNamespace(fatal_error=None, _retire=lambda rec: None)
    return scheduler


def publish(scheduler, prepared, C, images=None):
    target, rings, _ = pair(C)
    if C < 128:
        for ring in rings:
            ring.keys = ring.keys[:, :, :C]
    saved = scheduler.checkpoints.capture(target, rings, prepared.token_ids[:C],
        prepared.multimodal.identities() if images is None else images)
    scheduler.checkpoints.publish(saved)
    return saved


@pytest.mark.parametrize('fault', ['none', 'missing', 'modified', 'patch', 'pixel', 'pair', 'lease'])
def test_qualification_miss_admission(monkeypatch, fault):
    prepared, tokenizer, data, _ = prepare_fixture()
    scheduler = idle_scheduler()
    backend = SimpleNamespace(scheduler=scheduler, dependency_identity='identity')
    mm = prepared.multimodal
    if fault == 'missing':
        mm.release_sources()
    if fault == 'modified':
        mm._sources[mm.spans[0].start] = b'changed'
    if fault in ('patch', 'pixel'):
        field = dict(grid=(200, 200)) if fault == 'patch' else dict(source_pixels=17*1024**2)
        prepared.multimodal = replace(mm, spans=(replace(mm.spans[0], **field),))
    if fault == 'pair':
        publish(scheduler, prepared, len(prepared.token_ids)-2)
    if fault == 'lease':
        scheduler.session_leases['retained'] = (0, b'lease')
    calls = []
    def observe(backend, request, tokenizer, **kwargs):
        calls.append('existing observer')
        request.multimodal.validate_prefix(0, [])
        request.multimodal.validate_workload(0)
        assert kwargs['source_bytes'] == data
        return {}
    monkeypatch.setattr(tool, 'observe_case_b_n2', observe)
    reasons = dict(missing='source unavailable', modified='source unavailable', patch='patch budget',
                   pixel='pixel budget', pair='complete paired miss', lease='logical leases')
    if fault == 'none':
        result = tool.observe_cold_messages_n2(backend, prepared, tokenizer, source_bytes=data)
        assert result['cache_miss']['frontier'] == 0
        assert result['cache_miss']['hit_identity'] is None
        assert result['cache_miss']['fresh_patches'] == 1551
        assert result['cache_miss']['covered_images'] == 0
        assert calls == ['existing observer']
    else:
        with pytest.raises(ValueError, match=reasons[fault]):
            tool.observe_cold_messages_n2(backend, prepared, tokenizer, source_bytes=data)
        assert not calls
    assert not prepared.multimodal._sources
    assert len(scheduler.checkpoints._payloads) == (fault == 'pair')
    scheduler.checkpoints.clear()


def test_real_recipe_wire_age_does_not_change_numerical_representation():
    prepared, _, data, _ = prepare_fixture()
    mm = prepared.multimodal
    assert prepared.image_is_historical == (True,)
    # This is classification only: no message/history or token changes.
    prepared.image_is_historical = (False,)
    assert prepared.multimodal is mm
    assert prepared.token_ids == list(mm.token_ids)
    assert mm._sources[mm.spans[0].start] == data
    mm.validate_prefix(0, [])
    mm.validate_workload(0)
    mm.release_sources()


@pytest.mark.parametrize('case', ['complete', 'partial', 'changed', 'intersection'])
def test_genuine_pair_selection(case):
    prepared, _, _, _ = prepare_fixture()
    scheduler = idle_scheduler()
    mm = prepared.multimodal
    end = mm.spans[0].start + mm.spans[0].length
    if case == 'intersection':
        with pytest.raises(ValueError, match='intersects'):
            mm.validate_prefix(end-1, mm.identities())
        with pytest.raises(ValueError, match='intersects'):
            publish(scheduler, prepared, end-1)
        assert not scheduler.checkpoints._payloads
    else:
        C = len(prepared.token_ids)-2 if case == 'complete' else end
        saved = publish(scheduler, prepared, C)
        images = mm.identities()
        if case == 'changed':
            images = [dict(images[0], sha256='different')]
        restored = scheduler.checkpoints.acquire(prepared.token_ids, images=images)
        if case == 'changed':
            assert restored is None  # Same placeholder IDs are not an image hit.
            assert scheduler.checkpoints.acquire(prepared.token_ids, images=mm.identities()) is not None
        else:
            assert restored[2] == list(saved.tokens)
            mm.validate_prefix(C, list(saved.images))
            mm.validate_workload(C)
            calls = []
            mm.encode_new(SimpleNamespace(config=mm._config), C,
                          span_encoder=lambda *args: calls.append(args))
            assert not calls  # Wholly covered image; zero fresh Vision work.
    mm.release_sources()
    scheduler.checkpoints.clear()


@pytest.mark.parametrize('case', ['miss', 'discount_miss', 'discount_short_pair', 'retained_missing', 'cancel'])
def test_external_guard_and_premutation_retirement(case):
    from omlx.request import Request, SamplingParams
    prepared, _, _, _ = prepare_fixture()
    scheduler = idle_scheduler()
    mm = prepared.multimodal
    if case in ('discount_miss', 'discount_short_pair'):
        prepared.multimodal = replace(mm, spans=(replace(mm.spans[0], grid=(200, 200)),))
    if case == 'discount_short_pair':
        publish(scheduler, prepared, mm.spans[0].start)  # Text-only pair before image.
    request = Request(request_id='step2a', prompt=prepared.token_ids, sampling_params=SamplingParams())
    request.prompt_token_ids = prepared.token_ids
    request.execution = QualificationSession('step2a')
    request.trace = dict(t0=perf_counter())
    request.prepared, request.recipe_tokenizer = prepared, None
    request.image_identities = tuple(prepared.multimodal.identities())
    request.required_image_frontier = mm.spans[0].start+mm.spans[0].length
    request.native_started = False
    request.prompt_checkpoint, request.delivery_cursor = None, 0
    request.logical_session = None
    if case == 'retained_missing':
        # Exact retained integrity is checked before historical coverage.
        request.logical_session = SimpleNamespace(canonical=(1,), session_id='lost')
        scheduler.backend.production_sessions = SimpleNamespace(publish=lambda *args: None)
    scheduler.requests[request.request_id] = request
    scheduler.waiting.append(request)
    if case == 'cancel':
        assert scheduler._do_abort_request(request.request_id)
    else:
        scheduler._schedule_waiting()  # A missing _start would expose any accidental admission.
        error = scheduler._completed[-1].error
        assert ('not retained; no replay' if case == 'retained_missing' else
                'historical image pair unavailable') in error
    assert not request.native_started
    assert not prepared.multimodal._sources
    assert len(scheduler.checkpoints._payloads) == (case == 'discount_short_pair')
    assert not scheduler.requests and not scheduler.waiting and not scheduler.running
    assert scheduler.backend.fatal_error is None
    scheduler.checkpoints.clear()
