"""Source history is not fresh work; real paired coverage is still mandatory."""
from types import SimpleNamespace as NS
import weakref

import pytest
from PIL import Image

from ds41f_mlx.model_execution.config import ModelConfig
from ds41f_mlx.runtime import multimodal as mm
from ds41f_mlx.serving import paired_checkpoint as pc
from tools.qualify_historical_image_admission import SIZES, fixture_images
from test_metadata_first_vision import scheduler_request, source
from test_production_mtp import MX, pair


@pytest.fixture(scope='module')
def incident_images():
    return fixture_images()


def history(images):
    cfg = ModelConfig(vision_n_layers=1, max_seq_len=1_048_576)
    return mm.prepare_multimodal([7]+[cfg.image_token_id]*len(images)+[8, 9],
        [NS(kind='bytes', data=data) for data in images], cfg,
        ordinary=True, historical=(True,)*len(images))


def test_twentieth_occurrence_validates_sequentially_without_retaining_pixels(incident_images, monkeypatch):
    assert sum(w*h for w,h in SIZES[:-1]) == 16251588
    assert sum(w*h for w,h in SIZES) == 16955588 > mm.MAX_REQUEST_PIXELS
    refs, calls = [], []
    original_open, original_validate = Image.open, mm.validate_encoded_image
    def opened(*args, **kwargs):
        image = original_open(*args, **kwargs)
        refs.append(weakref.ref(image))
        return image
    def validated(*args, **kwargs):
        assert not any(r() is not None for r in refs)
        calls.append(True)
        return original_validate(*args, **kwargs)
    monkeypatch.setattr(Image, 'open', opened)
    monkeypatch.setattr(mm, 'validate_encoded_image', validated)
    inp = history(incident_images)
    assert len(calls) == 20 and len(refs) == 40
    assert not any(r() is not None for r in refs)
    assert sum(s.source_pixels for s in inp.spans) == 16955588
    assert all(isinstance(v, bytes) for v in inp._sources.values())
    with pytest.raises(ValueError, match='patch budget'):
        inp.validate_workload(0)  # historical flags did not grant an executable hit
    inp.release_sources()


@pytest.mark.parametrize('case', ['hit', 'miss', 'projection', 'identity'])
def test_twenty_images_use_actual_pair_or_fail_before_native(incident_images, monkeypatch, case):
    inp = history(incident_images)
    C = len(inp.token_ids)-2
    authority = pc.PairedCheckpointAuthority('identity', MX)
    saved = authority.capture(*pair(C)[:2], inp.token_ids[:C], inp.identities())
    authority.publish(saved)
    if case == 'miss':
        authority.clear()
    elif case == 'identity':
        changed = list(incident_images)
        changed[0] = source(size=SIZES[0], color='blue').data
        inp.release_sources()
        inp = history(changed)
    scheduler, request = scheduler_request(inp, authority)
    request.prepared.protocol = 'messages'  # existing independent recovery path
    request.required_image_frontier = inp.spans[-1].start+inp.spans[-1].length
    if case == 'projection':
        request.prompt_token_ids[0] += 1
    starts, errors = [], []
    def start(rec, *args, **kwargs):
        starts.append(len(rec.canonical))
        assert tuple(rec.canonical) == saved.tokens  # never a fabricated shorter prefix
        result = inp.encode_new(NS(config=inp._config), len(rec.canonical),
            span_encoder=lambda *a: pytest.fail('covered image re-encoded'))
        assert result.rows == ()
        scheduler.batch_generator = object()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    scheduler.backend = NS(_start=start, request_max_tokens=lambda p: 16)
    scheduler._retire = lambda req, *, error: errors.append(error)
    scheduler._schedule_waiting()
    if case == 'hit':
        assert starts == [C] and not errors
        assert request.trace['historical_patch_discount']
    else:
        assert not starts and len(errors) == 1 and not request.native_started
        assert 'budget' in str(errors[0])
    assert not inp._sources
    authority.clear()


def test_above_former_pixels_real_miss_encodes_every_current_image():
    import mlx.core as mx
    data = source(size=(2048, 2048)).data
    inp = history([data]*4+[source(size=(512, 512)).data])
    assert sum(s.source_pixels for s in inp.spans) > mm.MAX_REQUEST_PIXELS
    assert sum(s.grid[0]*s.grid[1] for s in inp.spans) <= mm.MAX_REQUEST_PATCHES
    authority = pc.PairedCheckpointAuthority('identity', MX)
    scheduler, request = scheduler_request(inp, authority)
    request.prepared.protocol = 'messages'
    starts, encoded, errors = [], [], []
    def encoder(pixels, h, w, kinds):
        encoded.append((h, w))
        return mx.zeros((len(kinds), 4))
    def start(rec, *args, **kwargs):
        starts.append(len(rec.canonical))
        rows = inp.encode_new(NS(config=inp._config), len(rec.canonical), span_encoder=encoder)
        assert [origin for origin, _ in rows.rows] == [s.start for s in inp.spans]
        scheduler.batch_generator = object()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    scheduler.backend = NS(_start=start, request_max_tokens=lambda p: 16)
    scheduler._retire = lambda req, *, error: errors.append(error)
    scheduler._schedule_waiting()
    assert starts == [0] and len(encoded) == 5 and not errors
    assert not inp._sources
    authority.clear()


def test_pixel_only_deferral_uses_existing_whole_image_reconstruction():
    import mlx.core as mx
    inp = history([source(size=(2048, 2048)).data]*4+[source(size=(1, 1)).data])
    C = inp.spans[-1].start
    assert sum(s.source_pixels for s in inp.spans) == mm.MAX_REQUEST_PIXELS+1
    assert sum(s.grid[0]*s.grid[1] for s in inp.spans) <= mm.MAX_REQUEST_PATCHES
    authority = pc.PairedCheckpointAuthority('identity', MX)
    saved = authority.capture(*pair(C)[:2], inp.token_ids[:C], inp.identities())
    authority.publish(saved)
    scheduler, request = scheduler_request(inp, authority)
    request.prepared.protocol = 'messages'
    request.required_image_frontier = inp.spans[-1].start+inp.spans[-1].length
    starts, encoded, errors = [], [], []
    def encoder(pixels, h, w, kinds):
        encoded.append((h, w))
        return mx.zeros((len(kinds), 4))
    def start(rec, *args, **kwargs):
        starts.append(len(rec.canonical))
        rows = inp.encode_new(NS(config=inp._config), len(rec.canonical), span_encoder=encoder)
        assert [origin for origin, _ in rows.rows] == [C]
        scheduler.batch_generator = object()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    scheduler.backend = NS(_start=start, request_max_tokens=lambda p: 16)
    scheduler._retire = lambda req, *, error: errors.append(error)
    scheduler._schedule_waiting()
    assert starts == [C] and len(encoded) == 1 and not errors
    assert not inp._sources
    authority.clear()


def test_partial_image_and_identity_still_refuse_discount(incident_images):
    inp = history(incident_images)
    C = inp.spans[-1].start+1
    with pytest.raises(ValueError, match='intersects'):
        inp.validate_prefix(C, inp.identities())
    with pytest.raises(ValueError, match='intersects'):
        pc.PairedCheckpoint.capture('identity', *pair(C)[:2], inp.token_ids[:C], MX, inp.identities())
    identities = inp.identities()
    identities[0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='identity'):
        inp.validate_prefix(len(inp.token_ids)-2, identities)
    inp.release_sources()


def test_oversized_historical_image_and_encoded_bytes_still_reject(monkeypatch):
    with pytest.raises(ValueError, match='pixel limit'):
        history([source(size=(2049, 2048)).data])
    with pytest.raises(ValueError, match='invalid image'):
        history([b'not an image'])
    monkeypatch.setattr(mm, 'MAX_REQUEST_IMAGE_BYTES', 1)
    with pytest.raises(ValueError, match='byte budget'):
        history([source().data])


def test_file_inline_and_mixed_twenty_image_preparation_are_identical(http, incident_images):
    # Reuse the real HTTP/Recipe fixture, not a second parser or source adapter.
    from test_image_files import body, inline, ref, upload
    client, _, prepared = http
    keys = [upload(client, data=data, mime='image/jpeg' if i in (2,5,6,7,8,13,14,17) else 'image/png')['id']
            for i, data in enumerate(incident_images)]
    for mode in ('file', 'inline', 'mixed'):
        blocks = []
        for i, data in enumerate(incident_images):
            block = ref(keys[i]) if mode == 'file' or mode == 'mixed' and i % 2 else inline(data)
            if block['source']['type'] == 'base64' and i in (2,5,6,7,8,13,14,17):
                block['source']['media_type'] = 'image/jpeg'
            blocks.append(block)
        b = body(blocks)
        b['messages'] += [dict(role='assistant', content='OK'), dict(role='user', content='continue')]
        r = client.post('/v1/messages', json=b)
        assert r.status_code == 200, r.text
    a, b, c = prepared
    for item in (b, c):
        assert item.token_ids == a.token_ids
        assert item.multimodal.identities() == a.multimodal.identities()
        assert item.image_is_historical == a.image_is_historical == (True,)*20
    assert sum(s.source_pixels for s in a.multimodal.spans) == 16955588
    for item in prepared:
        item.multimodal.release_sources()


# Pytest fixtures defined in peer modules must be explicitly imported.
from test_image_files import http  # noqa: E402,F401
