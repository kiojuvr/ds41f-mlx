"""Patch deferral is conditional paired-byte admission, never historical replay."""
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


def test_only_historical_patches_defer_not_cold_pixels_bytes_or_off(monkeypatch):
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
    with pytest.raises(ValueError, match='pixel budget'):
        history()
    monkeypatch.setattr(mm, 'MAX_REQUEST_IMAGE_BYTES', 1)
    with pytest.raises(ValueError, match='byte budget'):
        history()
    inp.release_sources()


@pytest.mark.parametrize('case', ['hit', 'miss', 'changed', 'partial', 'pair_bytes', 'root_bytes', 'fresh_work', 'profile'])
def test_discount_checks_before_clone_and_native_start(monkeypatch, case):
    monkeypatch.setattr('ds41f_mlx.execution_identity.numerical_profile',
                        lambda: 'unqualified' if case == 'profile' else 'flash-1017-pre-woa-fp8-v1')
    inp = history()
    C = len(inp.token_ids)-2
    images = inp.identities()
    authority = pc.PairedCheckpointAuthority('identity', MX)
    target, rings, _ = pair(C)
    saved = authority.capture(target, rings, inp.token_ids[:C], images)
    authority.publish(saved)
    if case == 'miss':
        authority.clear()
    if case == 'changed':
        inp = replace(inp, spans=(replace(inp.spans[0], digest='b'*64),) + inp.spans[1:])
    if case == 'partial':
        inp = replace(inp, spans=(replace(inp.spans[0], start=C-1, length=2),) + inp.spans[1:])
    if case == 'pair_bytes':
        monkeypatch.setattr(pc, 'PATCH_DISCOUNT_PAIR_BYTES', 1)
    if case == 'root_bytes':
        monkeypatch.setattr(pc, 'PATCH_DISCOUNT_ROOT_BYTES', 1)
    scheduler, request = scheduler_request(inp, authority)
    request.required_image_frontier = max(s.start+s.length for s in inp.spans)
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
    scheduler.backend = NS(_start=start, request_max_tokens=lambda p: 64)
    scheduler._retire = lambda req, *, error: errors.append(error)
    scheduler._schedule_waiting()
    if case == 'hit':
        assert starts == restores == [True] and not errors
        assert request.trace['historical_patch_discount']
    else:
        assert not starts and not restores and len(errors) == 1
        assert not request.native_started
    assert not inp._sources
    authority.clear()


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
    assert pc.projected_pair_bytes(old, 1048576) > pc.PATCH_DISCOUNT_PAIR_BYTES
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
