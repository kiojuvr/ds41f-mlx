"""Harness key envelope and bounded physical histories, no model weights."""
import pytest

from test_metadata_first_vision import prepare, source, forbidden, scheduler_request
from ds41f_mlx.model_execution import processing
from ds41f_mlx.runtime import multimodal as mm
from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority, image_keys, image_ranges
from test_production_mtp import MX, pair
from types import SimpleNamespace as NS


@pytest.mark.parametrize('case', ['hit', 'incremental', 'miss', 'changed', 'offloaded'])
def test_larger_valid_paired_history(monkeypatch, case):
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    src = source(size=(1, 1))
    old = prepare([src] * 23, ordinary=True)
    incoming = [src] * (24 if case == 'incremental' else 23)
    if case == 'changed':
        incoming[0] = source(size=(1, 1), color='blue')
    # Official projection replaces oldest occurrences with text, changing all
    # subsequent absolute spans. No pretending the old pair covers this prompt.
    inp = prepare(incoming[20:] if case == 'offloaded' else incoming,
                  ordinary=True, prefix=20 if case == 'offloaded' else 1)
    C = len(old.token_ids)-2
    authority = PairedCheckpointAuthority('identity', MX)
    if case != 'miss':
        target, rings, _ = pair(C)
        authority.publish(authority.capture(target, rings, old.token_ids[:C], old.identities()))
    scheduler, request = scheduler_request(inp, authority)
    errors, starts = [], []
    def start(rec, prepared, *args, **kwargs):
        starts.append(len(rec.canonical))
        prepared.multimodal.validate_workload(len(rec.canonical))
        scheduler.batch_generator = object()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    scheduler.backend = NS(_start=start)
    scheduler._retire = lambda req, *, error: errors.append(error)
    scheduled, _ = scheduler._schedule_waiting()
    if case in ('hit', 'incremental'):
        assert scheduled == [request] and starts == [C] and not errors
        assert sum(s.start >= C for s in inp.spans) == (case == 'incremental')
    else:
        assert not scheduled and not starts
        assert 'historical image pair unavailable' in str(errors[0])
        assert not request.native_started
    assert not inp._sources
    authority.clear()
    assert not authority._payloads


def test_600_count_fresh_work_and_hash_compatibility(monkeypatch):
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    src = source(size=(1, 1))
    with pytest.raises(ValueError, match='patch budget'):
        prepare([src] * 600, ordinary=True)
    inp = prepare([src] * 24, ordinary=True)
    inp.validate_workload(inp.spans[-1].start)
    inp.validate_prefix(inp.spans[-1].start, inp.identities()[:-1])
    with pytest.raises(ValueError, match='intersects'):
        inp.validate_workload(inp.spans[-1].start+1)
    ids = [dict(start=1+i*184, length=184, sha256='0'*64, grid=[39,39]) for i in range(600)]
    assert image_ranges(ids) == [(s['start'], image_keys(ids[:i+1])) for i, s in enumerate(ids)]
    ranges = image_ranges(ids)
    assert ranges[0][1][0] is ranges[-1][1][0]
    with pytest.raises(ValueError, match='occurrence'):
        prepare([src] * 601, ordinary=True)
    # Lower the fresh budget to exercise rejection before native execution.
    # Production all-history guards are NOT raised for these tests.
    monkeypatch.setattr(mm, 'MAX_REQUEST_PATCHES', 1)
    # A cold fresh request rejects before native execution and drops all bytes.
    authority = PairedCheckpointAuthority('identity', MX)
    scheduler, request = scheduler_request(inp, authority)
    request.required_image_frontier = 0
    errors = []
    scheduler.backend = NS(_start=forbidden)
    scheduler._retire = lambda req, *, error: errors.append(error)
    assert scheduler._schedule_waiting()[0] == []
    assert 'patch budget' in str(errors[0]) and not request.native_started
    assert not inp._sources
    authority.clear()


def test_every_historical_occurrence_is_fully_validated(monkeypatch):
    original, calls = mm.validate_encoded_image, []
    def validate(data, **kwargs):
        calls.append(len(data))
        return original(data, **kwargs)
    monkeypatch.setattr(mm, 'validate_encoded_image', validate)
    src = source(size=(1, 1))
    inp = prepare([src] * 24, ordinary=True)
    assert len(calls) == len(inp.identities()) == 24
    inp.release_sources()
    calls.clear()
    with pytest.raises(ValueError, match='invalid image'):
        prepare([src] * 23 + [NS(kind='bytes', data=src.data[:40])], ordinary=True)
    assert len(calls) == 24
