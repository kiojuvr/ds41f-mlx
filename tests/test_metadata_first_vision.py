"""Geometry/cache/lifetime boundaries with real Pillow/MLX, no model weights."""
from collections import deque
from hashlib import sha256
from io import BytesIO
from time import perf_counter
from types import SimpleNamespace as NS
import weakref

import mlx.core as mx
import numpy as np
from PIL import Image
import pytest

from ds41f_mlx.model_execution.config import ModelConfig
from ds41f_mlx.model_execution import processing
from ds41f_mlx.runtime.multimodal import prepare_multimodal


def source(size=(400, 300), color='red', format='PNG'):
    stream = BytesIO()
    Image.new('RGB', size, color).save(stream, format=format)
    return NS(kind='bytes', data=stream.getvalue())


def prepare(sources, *, ordinary=False, prefix=1):
    config = ModelConfig(vision_n_layers=1, max_seq_len=1_048_576)
    ids = [7] * prefix
    for i in range(len(sources)):
        ids.extend([config.image_token_id, 100+i])
    ids.append(8)
    return prepare_multimodal(ids, sources, config, ordinary=ordinary)


def forbidden(*args, **kwargs):
    pytest.fail('cached/queued image reached patch preparation or Vision')


@pytest.mark.parametrize('format', ['PNG', 'JPEG', 'WEBP'])
@pytest.mark.parametrize('size', [(1, 1), (400, 300), (777, 1023),
                                  (2048, 2048), (1024, 2048), (2048, 1024)])
def test_metadata_matches_existing_patch_geometry(size, format, monkeypatch):
    config = ModelConfig(vision_n_layers=1)
    src = source(size=size, format=format)
    with Image.open(BytesIO(src.data)) as image:
        _, h, w, kinds = processing.image_patch_array(image, config)
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    # Even normalized float32 allocation is forbidden in request preparation.
    monkeypatch.setattr(processing.np, 'asarray', forbidden)
    inp = prepare([src])
    span, = inp.spans
    assert span.grid == (h, w) and span.kinds == tuple(kinds)
    assert span.identity() == dict(start=1, length=len(kinds),
                                   sha256=sha256(src.data).hexdigest(), grid=[h, w])
    assert inp.token_ids == (7,) + (config.image_token_id,)*len(kinds) + (100, 8)
    assert not hasattr(span, 'patches')
    assert inp._sources == {1: src.data}


def test_cached_images_do_not_resize_patch_or_encode(monkeypatch):
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    monkeypatch.setattr(Image.Image, 'resize', forbidden)
    inp = prepare([source(), source(color='blue')])
    identities = inp.identities()
    frontier = len(inp.token_ids)-2
    inp.validate_prefix(frontier, identities)
    rows = inp.encode_new(NS(config=ModelConfig(vision_n_layers=1)), frontier,
                          span_encoder=forbidden)
    assert rows.rows == ()
    assert not inp._sources and inp.identities() == identities


def test_mixed_cached_and_new_images_use_original_patches_in_absolute_order(monkeypatch):
    sources = [source(color=color) for color in ('red', 'blue', 'green', 'white', 'black', 'yellow')]
    inp = prepare(sources, ordinary=True, prefix=8200)
    frontier = inp.spans[2].start  # Two admitted historical images, two new.
    inp.validate_prefix(frontier, inp.identities()[:2])
    expected = []
    for src in sources[2:]:
        with Image.open(BytesIO(src.data)) as image:
            patches, h, w, kinds = processing.image_patch_array(image, inp._config)
        expected.append((mx.array(patches).astype(mx.bfloat16), h, w, tuple(kinds)))
    patch_refs, calls = [], []
    original = processing.image_patch_array
    def patch(image, config):
        result = original(image, config)
        patch_refs.append(weakref.ref(result[0]))
        return result
    def encode(pixels, h, w, kinds):
        # The preceding image's normalized CPU array must already be gone.
        assert all(ref() is None for ref in patch_refs[:-1])
        reference, eh, ew, ek = expected[len(calls)]
        assert (h, w, kinds) == (eh, ew, ek)
        np.testing.assert_array_equal(np.asarray(pixels.astype(mx.float32)),
                                      np.asarray(reference.astype(mx.float32)))
        calls.append((h, w, kinds))
        return mx.full((len(kinds), 4), len(calls), dtype=mx.bfloat16)
    monkeypatch.setattr(processing, 'image_patch_array', patch)
    embeddings = inp.encode_new(NS(config=inp._config), frontier, span_encoder=encode)
    assert [start for start, _ in embeddings.rows] == [s.start for s in inp.spans[2:]]
    assert len(calls) == len(patch_refs) == 4
    assert all(ref() is None for ref in patch_refs) and not inp._sources
    # Existing sparse merge keeps absolute positions and text between images.
    merged = embeddings.merge(mx.zeros((1, len(inp.token_ids), 4)), 0)
    for index, span in enumerate(inp.spans):
        np.testing.assert_array_equal(np.asarray(merged)[0, span.start:span.start+span.length],
                                      0 if index < 2 else index-1)
        np.testing.assert_array_equal(np.asarray(merged)[0, span.start+span.length], 0)


@pytest.mark.parametrize('stage', ['patch', 'encode', 'eval'])
def test_encoding_failure_releases_sources_and_normalized_arrays(monkeypatch, stage):
    inp = prepare([source(), source(color='blue')])
    refs = []
    original = processing.image_patch_array
    def patch(image, config):
        if stage == 'patch':
            raise RuntimeError('patch failure')
        result = original(image, config)
        refs.append(weakref.ref(result[0]))
        return result
    def encode(*args):
        if stage == 'encode':
            raise RuntimeError('encode failure')
        return mx.ones((inp.spans[0].length, 4))
    def fail_eval(*args):
        raise RuntimeError('eval failure')
    monkeypatch.setattr(processing, 'image_patch_array', patch)
    if stage == 'eval':
        monkeypatch.setattr(mx, 'eval', fail_eval)
    with pytest.raises(RuntimeError, match=stage) as error:
        inp.encode_new(NS(config=inp._config), span_encoder=encode)
    # Keep the exception/traceback alive: it must not keep normalized arrays.
    assert error.value is not None
    assert not inp._sources and all(ref() is None for ref in refs)


def test_metadata_does_not_accept_header_only_or_mismatched_source(monkeypatch):
    src = source(format='JPEG')
    # Pillow can read these header dimensions, but full pixel decoding fails.
    truncated = src.data[:-30]
    with Image.open(BytesIO(truncated)) as image:
        assert image.size == (400, 300)
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    with pytest.raises(ValueError, match='invalid image'):
        prepare([NS(kind='bytes', data=truncated)])
    inp = prepare([src])
    inp._sources[1] = source(color='blue').data
    with pytest.raises(ValueError, match='changed'):
        inp.encode_new(NS(config=inp._config), span_encoder=forbidden)
    assert not inp._sources


def test_existing_mtp_adapter_and_mlx_vision_match_eager_reference(monkeypatch):
    from omlx.patches.deepseek_v41 import model as native
    from omlx.patches.deepseek_v41.config import ModelConfig as NativeConfig
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    # Tiny random Vision weights, not checkpoint/model acceptance. Language
    # execution is excluded; the actual MTP Model/ViT/aligner adapter is used.
    config = NativeConfig(vision_n_layers=1, vision_dim=16, vision_n_heads=4,
                          vision_inter_dim=32, dim=8, vision_min_pixels=42*42,
                          preserve_mtp=True)
    monkeypatch.setattr(native, 'LanguageModel', lambda c:
                        NS(embed=lambda ids: mx.zeros((*ids.shape, c.dim))))
    model = native.Model(config)
    src = source(size=(42, 42))
    ids = [7, config.image_token_id, 8, 9]
    with Image.open(BytesIO(src.data)) as image:
        patches, h, w, kinds = processing.image_patch_array(image, config)
    eager = model.get_input_embeddings(mx.full((1, len(kinds)), config.image_token_id),
        pixel_values=mx.array(patches).astype(mx.bfloat16), image_grids=[(h, w)],
        image_spans=[(0, len(kinds))], image_types=[kinds]).inputs_embeds[0]
    mx.eval(eager)
    inp = prepare_multimodal(ids, [src], config)
    backend = NS(_model=model)
    result = ProductionMTPBackend._encode_new_images(backend, inp, 0)
    assert result.rows[0][0] == 1 and result.rows[0][1].shape == (len(kinds), config.dim)
    np.testing.assert_array_equal(np.asarray(result.rows[0][1]), np.asarray(eager))
    assert not inp._sources
    cached = prepare_multimodal(ids, [src], config)
    monkeypatch.setattr(native.ViT, '__call__', forbidden)
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    assert ProductionMTPBackend._encode_new_images(backend, cached, len(cached.token_ids)-2).rows == ()


def test_partial_frontier_and_missing_new_source_fail_closed(monkeypatch):
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    inp = prepare([source()])
    with pytest.raises(ValueError, match='intersects'):
        inp.encode_new(NS(config=inp._config), 2, span_encoder=forbidden)
    assert not inp._sources
    with pytest.raises(ValueError, match='unavailable'):
        inp.encode_new(NS(config=inp._config), span_encoder=forbidden)


def scheduler_request(inp, authority):
    """Use the real Scheduler admission and real upstream paired lookup."""
    from omlx.request import Request, SamplingParams
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    from ds41f_mlx.serving.production_mtp import ProductionScheduler
    scheduler = object.__new__(ProductionScheduler)
    scheduler.checkpoints = authority
    scheduler.waiting, scheduler.running, scheduler.requests = deque(), {}, {}
    scheduler.request_id_to_uid, scheduler.uid_to_request_id = {}, {}
    scheduler.total_prompt_tokens = 0
    scheduler.batch_generator = None
    request = Request(request_id='image', prompt=list(inp.token_ids), sampling_params=SamplingParams())
    request.prompt_token_ids = list(inp.token_ids)
    request.prepared, request.recipe_tokenizer = NS(protocol='chat_completions', multimodal=inp), None
    request.execution = QualificationSession('image')
    request.image_identities = tuple(inp.identities())
    request.required_image_frontier = inp.spans[0].start + inp.spans[0].length
    request.trace = dict(t0=perf_counter())
    request.native_started = False
    scheduler.waiting.append(request)
    scheduler.requests['image'] = request
    return scheduler, request


@pytest.mark.parametrize('count', [1, 6])
@pytest.mark.parametrize('case', ['hit', 'mixed', 'fresh_miss', 'miss', 'changed', 'target_only'])
def test_paired_lookup_precedes_patch_preparation(monkeypatch, case, count):
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    from test_production_mtp import MX, pair
    prefix = 8200 if count > 1 else 1
    old = prepare([source()] * count, ordinary=True, prefix=prefix)
    inp = prepare([source(color='blue' if case == 'changed' else 'red')] * count +
                  ([source(color='green')] if case == 'mixed' else []), ordinary=True, prefix=prefix)
    C = len(old.token_ids)-2
    assert C >= 128
    authority = PairedCheckpointAuthority('identity', MX)
    target, rings, _ = pair(C)
    if case not in ('miss', 'fresh_miss'):
        saved = authority.capture(target, rings, old.token_ids[:C], old.identities())
        authority.publish(saved)
        if case == 'target_only':
            # A target/index hint without its paired payload is NOT a cache hit.
            authority._payloads.clear()
    scheduler, request = scheduler_request(inp, authority)
    if case == 'fresh_miss':
        request.required_image_frontier = 0  # Image belongs to the new last message.
    expected_frontier = 0 if case == 'fresh_miss' else C
    errors, starts, patches = [], [], []
    original = processing.image_patch_array
    def patch(image, config):
        assert request.cached_tokens == expected_frontier
        patches.append(image.size)
        return original(image, config)
    def start(rec, prepared, tokenizer, trace, **kwargs):
        starts.append(len(rec.canonical))
        embeddings = prepared.multimodal.encode_new(NS(config=inp._config), len(rec.canonical),
            span_encoder=lambda pixels, h, w, kinds: mx.ones((len(kinds), 4)))
        expected_starts = ([inp.spans[-1].start] if case == 'mixed' else
                           [s.start for s in inp.spans] if case == 'fresh_miss' else [])
        assert [p for p, _ in embeddings.rows] == expected_starts
        scheduler.batch_generator = object()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    scheduler.backend = NS(_start=start)
    scheduler._retire = lambda req, *, error: errors.append(error)
    monkeypatch.setattr(processing, 'image_patch_array', patch)
    scheduled, _ = scheduler._schedule_waiting()
    if case in ('hit', 'mixed', 'fresh_miss'):
        assert not errors, errors
        assert scheduled == [request] and starts == [expected_frontier]
        assert len(patches) == (0 if case == 'hit' else count if case == 'fresh_miss' else 1)
    else:
        assert not scheduled and not starts and not patches
        assert len(errors) == 1 and 'historical image pair unavailable' in str(errors[0])
        assert not request.native_started
    assert not inp._sources
    authority.clear()


def test_queued_abort_discards_encoded_sources_without_patches(monkeypatch):
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    from test_production_mtp import MX
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    inp = prepare([source()])
    authority = PairedCheckpointAuthority('identity', MX)
    scheduler, request = scheduler_request(inp, authority)
    scheduler._completed = []
    scheduler.backend = NS()
    assert scheduler._do_abort_request(request.request_id)
    assert not inp._sources and not scheduler.waiting
    assert not scheduler.requests and scheduler._completed[0].finished
    authority.clear()


def test_fatal_retirement_releases_sources_of_unstarted_request(monkeypatch):
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
    from test_production_mtp import MX
    monkeypatch.setattr(processing, 'image_patch_array', forbidden)
    inp = prepare([source()])
    authority = PairedCheckpointAuthority('identity', MX)
    scheduler, request = scheduler_request(inp, authority)
    scheduler._completed = []
    scheduler.total_completion_tokens = 0
    scheduler.backend = NS(_retire=lambda rec: None, fatal_error='engine failed')
    request.prompt_checkpoint, request.delivery_cursor = None, 0
    scheduler._retire(request, error=RuntimeError('engine failed'))
    assert not inp._sources and not scheduler.waiting and not scheduler.requests
    assert scheduler._completed[0].error == 'engine failed'
    authority.clear()


def test_cancelled_encoding_keeps_inputs_until_worker_consumer_finishes(monkeypatch):
    import asyncio
    import threading
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from test_multimodal_cancellation import entered
    inp = prepare([source()])
    began, release = threading.Event(), threading.Event()
    refs = []
    original = processing.image_patch_array
    def patch(image, config):
        result = original(image, config)
        refs.append(weakref.ref(result[0]))
        return result
    def encode(pixels, h, w, kinds):
        began.set()
        assert release.wait(5)
        return mx.ones((len(kinds), 4))
    monkeypatch.setattr(processing, 'image_patch_array', patch)
    backend = DeepSeekRecipeRuntimeBackend()
    async def run():
        task = asyncio.create_task(backend._call(lambda:
            inp.encode_new(NS(config=inp._config), span_encoder=encode)))
        await entered(began)
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done() and refs[0]() is not None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert refs[0]() is None and not inp._sources
    try:
        asyncio.run(run())
    finally:
        release.set()
        backend.close()


def test_p6_seal_and_failure_do_not_retain_sparse_embedding_input(monkeypatch):
    from test_prefill_fp8_mlx_p1_p2 import FakeLanguageModel, full_ready_cache
    from test_prefill_fp8_mlx_p6 import fake_p6_app
    for fail in (False, True):
        app = fake_p6_app(FakeLanguageModel(), full_ready_cache(), list(range(16)))
        # Fixture supplies text-only tensors. Exercise input-reference lifetime
        # independently of image arithmetic (covered above with real MLX).
        original = app._make_segment_execution
        def setup(segment):
            rows, app.image_embeddings = app.image_embeddings, None
            try:
                return original(segment)
            finally:
                app.image_embeddings = rows
        monkeypatch.setattr(app, '_make_segment_execution', setup)
        class Rows:
            pass
        rows = Rows()
        ref = weakref.ref(rows)
        app.image_embeddings = rows
        del rows
        if fail:
            app.fail('injected failure')
        else:
            app.execute_all()
        assert app.image_embeddings is None and ref() is None
