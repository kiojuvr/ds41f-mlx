"""Historical discount requires complete physical pairs, never wire age alone."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from ds41f_mlx.runtime import multimodal as mm
from ds41f_mlx.serving import paired_checkpoint as pc
from test_metadata_first_vision import source, scheduler_request
from test_production_mtp import MX, pair
from ds41f_mlx.model_execution.config import ModelConfig


def history(count=8):
    cfg = ModelConfig(vision_n_layers=1, max_seq_len=1_048_576)
    sources = [source(size=(1280, 720))] * count
    ids = [7] + [cfg.image_token_id] * count + [8, 9]
    return mm.prepare_multimodal(ids, sources, cfg, ordinary=True, historical=(True,) * count)


def test_historical_work_defers_not_cold_work_bytes_or_off(monkeypatch):
    inp = history()
    assert sum(s.grid[0]*s.grid[1] for s in inp.spans) > mm.MAX_REQUEST_PATCHES
    cfg = inp._config
    sources = [source(size=(1280, 720))]*8
    ids = [7] + [cfg.image_token_id]*8 + [8, 9]
    with pytest.raises(ValueError, match='patch budget'):
        mm.prepare_multimodal(ids, sources, cfg, ordinary=True)
    with pytest.raises(ValueError, match='origins'):
        mm.prepare_multimodal([7, cfg.image_token_id, 8], sources[:1], cfg, historical=(True,))
    monkeypatch.setattr(mm, 'MAX_REQUEST_PIXELS', 1)
    deferred = history()
    with pytest.raises(ValueError, match='pixel budget'):
        deferred.validate_workload(0)
    deferred.validate_prefix(len(deferred.token_ids)-2, deferred.identities())
    deferred.validate_workload(len(deferred.token_ids)-2)
    deferred.release_sources()
    with pytest.raises(ValueError, match='pixel budget'):
        mm.prepare_multimodal(ids, sources, cfg, ordinary=True)
    monkeypatch.setattr(mm, 'MAX_REQUEST_IMAGE_BYTES', 1)
    with pytest.raises(ValueError, match='byte budget'):
        history()
    inp.release_sources()


@pytest.mark.parametrize('case', ['hit', 'miss', 'changed', 'partial', 'pair_bytes', 'root_bytes', 'both_bytes', 'evicted_roots', 'fresh_work', 'profile', 'compressor', 'ring', 'missing_pair', 'projection', 'compacted'])
def test_discount_checks_before_clone_and_native_start(monkeypatch, case):
    monkeypatch.setattr('ds41f_mlx.execution_identity.numerical_profile',
                        lambda: 'unqualified' if case == 'profile' else 'flash-1017-pre-woa-fp8-v1')
    inp = history()
    C = len(inp.token_ids)-2
    images = inp.identities()
    authority = pc.PairedCheckpointAuthority('identity', MX)
    target, rings, _ = pair(C)
    saved = authority.capture(target, rings, inp.token_ids[:C], images)
    if case in ('pair_bytes', 'both_bytes'):
        import numpy as np
        # Supported singleton window layout; real logical arrays, not a
        # patched estimator or raised admission constant.
        saved = replace(saved, target=tuple(replace(layer,
            cache=(layer.cache[0], np.zeros((1, 128, 4096), dtype=np.uint8), *layer.cache[2:]))
            for layer in saved.target))
    if case == 'compressor':
        saved = replace(saved, target=(replace(saved.target[0], compress_ratio=4),) + saved.target[1:])
    if case == 'ring':
        import numpy as np
        saved = replace(saved, rings=(replace(saved.rings[0], keys=np.zeros((1, 1, 64, 1))),) + saved.rings[1:])
    if case != 'ring':
        authority.publish(saved)
    else:
        # Corrupted retained payload after a valid publication.
        authority.publish(authority.capture(target, rings, inp.token_ids[:C], images))
        key = next(iter(authority._payloads))
        authority._payloads[key] = saved
    if case in ('root_bytes', 'both_bytes', 'evicted_roots'):
        import numpy as np
        # Unrelated old root: large logical auxiliary array, no sharing credit.
        other = authority.capture(*pair(256))
        layer = other.target[0]
        other = replace(other, target=(replace(layer, cache=(*layer.cache[:6],
            np.zeros((1, 193 * 1024**2), dtype=np.uint8))),) + other.target[1:])
        authority.publish(other)
        if case == 'evicted_roots':
            authority.paged.evict(next(k for k, v in authority._payloads.items() if v is other))
    if case == 'missing_pair':
        authority._payloads.clear()
    if case == 'miss':
        authority.clear()
    if case == 'changed':
        inp = replace(inp, spans=(replace(inp.spans[0], digest='b'*64),) + inp.spans[1:])
    if case == 'partial':
        inp = replace(inp, spans=(replace(inp.spans[0], start=C-1, length=2),) + inp.spans[1:])
    scheduler, request = scheduler_request(inp, authority)
    request.required_image_frontier = max(s.start+s.length for s in inp.spans)
    if case == 'projection':
        request.prompt_token_ids[0] += 1
    if case == 'compacted':
        request.prompt_token_ids = request.prompt_token_ids[1:]
    if case == 'fresh_work':
        # A formerly cached suffix is not automatically fresh-admissible.
        request.required_image_frontier = 0
        monkeypatch.setattr(mm.MultimodalInput, 'validate_workload', lambda *a: (_ for _ in ()).throw(ValueError('fresh Vision workload')))
    errors, starts, restores = [], [], []
    original = pc.PairedCheckpoint.restore
    def restore(self, *args, **kw):
        restores.append(True)
        return original(self, *args, **kw)
    monkeypatch.setattr(pc.PairedCheckpoint, 'restore', restore)
    def start(rec, *args, **kwargs):
        starts.append(True)
        scheduler.batch_generator = object()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    from ds41f_mlx.observability import Projection
    projection = Projection()
    projection.begin('bounded-id')
    request.trace['diagnostic_id'] = 'bounded-id'
    scheduler.backend = NS(_start=start, request_max_tokens=lambda p: 64, telemetry=projection)
    scheduler._retire = lambda req, *, error: errors.append(error)
    scheduler._schedule_waiting()
    if case in ('hit', 'pair_bytes', 'root_bytes', 'both_bytes', 'evicted_roots'):
        assert starts == restores == [True] and not errors
        trace = request.trace
        assert trace['historical_patch_discount']
        assert trace['selected_pair_frontier'] == C
        assert trace['fresh_encoder_patches'] == 0
        assert trace['historical_planned_patches'] == 38272
        assert trace['legacy_pair_bytes_exceeded'] == (case in ('pair_bytes', 'both_bytes'))
        assert trace['legacy_root_bytes_exceeded'] == (case in ('root_bytes', 'both_bytes'))
        assert trace['selected_pair_bytes'] == pc.pair_bytes(saved)
        observed = projection.snapshot()['current_request']['metrics']
        assert observed['selected_pair_bytes'] == trace['selected_pair_bytes']
        assert observed['legacy_pair_bytes_exceeded'] == trace['legacy_pair_bytes_exceeded']
        assert all(isinstance(v, (int, float, bool)) for v in observed.values())
        assert trace['retained_pair_roots'] == (2 if case in ('root_bytes', 'both_bytes') else 1)
        if case in ('pair_bytes', 'root_bytes', 'both_bytes', 'evicted_roots'):
            import json
            from ds41f_mlx.observability import TRACE_FIELDS
            print(json.dumps(dict(case=case, **{k: trace[k] for k in TRACE_FIELDS if k in trace})))
    else:
        assert not starts and not restores and len(errors) == 1
        assert not request.native_started
    assert not inp._sources
    authority.clear()


def test_real_uncovered_work_still_exceeds_fresh_budget():
    # Wire age permits preparation deferral, never an encoder-budget bypass.
    inp = history(16)
    try:
        frontier = inp.spans[8].start
        assert sum(s.grid[0]*s.grid[1] for s in inp.spans if s.start >= frontier) == 38272
        with pytest.raises(ValueError, match='patch budget'):
            inp.validate_workload(frontier)
        inp.validate_workload(inp.spans[9].start)  # seven remaining are within budget
    finally:
        inp.release_sources()


def test_deferred_history_still_validates_every_container(monkeypatch):
    from ds41f_mlx.model_execution import processing
    calls = []
    original = mm.validate_encoded_image
    def validate(data, **kwargs):
        calls.append(len(data))
        return original(data, **kwargs)
    monkeypatch.setattr(mm, 'validate_encoded_image', validate)
    monkeypatch.setattr(processing, 'image_patch_array', lambda *a: pytest.fail('preparation normalized pixels'))
    inp = history()
    assert len(calls) == 8
    inp.release_sources()
    src = source(size=(1280, 720))
    bad = NS(kind='bytes', data=src.data[:40])
    with pytest.raises(ValueError, match='invalid image'):
        mm.prepare_multimodal([7]+[inp._config.image_token_id]*8+[8, 9], [src]*7+[bad],
                             inp._config, ordinary=True, historical=(True,)*8)


def test_projection_bounds_packed_growth_tails_and_output_without_allocating():
    from math import prod
    class Array:
        def __init__(self, shape, itemsize):
            self.shape, self.itemsize, self.ndim = shape, itemsize, len(shape)
            self.nbytes = prod(shape)*itemsize
    def checkpoint(C):
        layers = []
        for ratio in [0, 1, 2]:
            slots = [Array((1,), 4), Array((1, 128, 528), 1),
                     Array((1, C//ratio if ratio else 0, 288), 1),
                     Array((1, C//ratio if ratio else 0, 68), 1),
                     Array((1, C%ratio if ratio > 1 else 0, 512), 4),
                     Array((1, C%ratio if ratio > 1 else 0, 512), 4), Array((1, 3), 8)]
            layers.append(NS(cache=slots, compress_ratio=ratio, left_padding=None, lengths=None))
        return NS(tokens=range(C), target=layers, rings=[NS(keys=Array((1, 1, 128, 128), 2))]*3)
    old = checkpoint(256)
    for end in (257, 1024, 8193, 1048576):
        assert pc.projected_pair_bytes(old, end) >= pc.pair_bytes(checkpoint(end))
    assert pc.projected_pair_bytes(old, 1048576) > 12 * 1024**2
    old.target[-1].cache[4] = None
    with pytest.raises(ValueError, match='tail layout'):
        pc.projected_pair_bytes(old, 257)


def test_pre_restore_admission_also_applies_to_retained_lease():
    authority = pc.PairedCheckpointAuthority('identity', MX)
    saved = authority.capture(*pair())
    authority.publish(saved)
    lease = authority.retain(saved.tokens)
    def deny(checkpoint):
        assert checkpoint is saved
        raise ValueError('fallback')
    with pytest.raises(ValueError, match='fallback'):
        authority.acquire_retained(lease, list(saved.tokens)+[1, 2], admit=deny)
    # Admission failure releases the temporary lease, not the canonical pin.
    assert authority.acquire_retained(lease, list(saved.tokens)+[1, 2]) is not None
    authority.release(lease)
    authority.clear()
