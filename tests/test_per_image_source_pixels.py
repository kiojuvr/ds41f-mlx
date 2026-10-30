"""Bounded witnesses for retaining source decode protection; no model load."""
from io import BytesIO
from types import SimpleNamespace as NS
import weakref

from PIL import Image, ImageFile
import pytest

from ds41f_mlx.model_execution.config import ModelConfig
from ds41f_mlx.model_execution import processing
from ds41f_mlx.runtime import multimodal as mm
from test_image_files import http, form, body, inline  # real shared HTTP fixture


def encoded(size, mode='RGB', fmt='PNG'):
    out = BytesIO()
    with Image.new(mode, size) as image:
        image.save(out, format=fmt)
    return out.getvalue()


def prepare(data):
    config = ModelConfig(vision_n_layers=1, max_seq_len=1_048_576)
    return mm.prepare_multimodal([7, config.image_token_id, 8],
        [NS(kind='bytes', data=data)], config, ordinary=True)


@pytest.mark.parametrize('size', [(512, 512), (2600, 2000), (3000, 1600), (1600, 3000)])
def test_source_boundary_is_independent_of_processor_geometry(size):
    data = encoded(size)
    assert len(data) < mm.MAX_IMAGE_BYTES
    config = ModelConfig(vision_n_layers=1)
    nh, nw, h, w = processing.plan_image_grid(*size, config)
    assert len(processing.image_token_kinds(nh, nw)) <= config.vision_max_n_token
    assert h * w // config.vision_patch_size**2 <= mm.MAX_REQUEST_PATCHES
    if size[0] * size[1] > mm.MAX_PIXELS:
        with pytest.raises(mm.ImageConstraintError, match='pixel limit'):
            mm.validate_encoded_image(data)
        with pytest.raises(ValueError, match='pixel limit'):
            prepare(data)
    else:
        assert mm.validate_encoded_image(data)['width'] == size[0]
        inp = prepare(data)
        inp.validate_sources()
        inp.validate_prefix(inp.spans[0].start + inp.spans[0].length, inp.identities())
        with pytest.raises(ValueError, match='intersects'):
            inp.validate_prefix(inp.spans[0].start + 1, [])
        inp.release_sources()


def test_compressed_source_rejected_before_full_decode(monkeypatch):
    # 20MP RGBA: ~80MB core despite <100KB wire bytes. Never decode this input.
    data = encoded((5000, 4000), mode='RGBA')
    assert len(data) < 100_000
    def forbidden(*args, **kwargs):
        pytest.fail('oversized source reached full pixel decode')
    monkeypatch.setattr(ImageFile.ImageFile, 'load', forbidden)
    with pytest.raises(mm.ImageConstraintError, match='pixel limit'):
        mm.validate_encoded_image(data)


@pytest.mark.parametrize('size', [(512, 512), (2600, 2000)])
def test_files_and_inline_same_source_boundary(http, size):
    client, store, prepared = http
    data = encoded(size)
    upload = client.post('/v1/files', **form(data=data))
    request = client.post('/v1/messages', json=body([inline(data)]))
    if size[0] * size[1] > mm.MAX_PIXELS:
        assert upload.status_code == 400, upload.text
        assert request.status_code == 400, request.text
        assert 'pixel limit' in upload.text and 'pixel limit' in request.text
        assert not store._files and not prepared
    else:
        assert upload.status_code == request.status_code == 200
        assert store.snapshot(upload.json()['id']).data == data
        assert list(prepared[-1].multimodal._sources.values()) == [data]


@pytest.mark.parametrize('kind', ['malformed', 'truncated', 'mime', 'animated'])
def test_complete_validation_preserved(kind):
    data, mime = encoded((128, 128), fmt='JPEG'), 'image/jpeg'
    if kind == 'malformed':
        data = b'\x89PNG\r\n\x1a\ninvalid'
        mime = 'image/png'
    elif kind == 'truncated':
        data = data[:-30]
    elif kind == 'mime':
        mime = 'image/png'
    else:
        out = BytesIO()
        with Image.new('RGB', (32, 32), 'red') as first, Image.new('RGB', (32, 32), 'blue') as second:
            first.save(out, format='PNG', save_all=True, append_images=[second])
        data, mime = out.getvalue(), 'image/png'
    with pytest.raises((ValueError, OSError)):
        mm.validate_encoded_image(data, mime)


@pytest.mark.parametrize('threshold,error', [(1000, Image.DecompressionBombWarning),
                                             (100, Image.DecompressionBombError)])
def test_pillow_bomb_protection_preserved(monkeypatch, threshold, error):
    data = encoded((40, 40))
    # Exercise both existing branches with tiny bounded inputs, not giant rasters.
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', threshold)
    with pytest.raises(error):
        mm.validate_encoded_image(data)


def test_rgb_copy_and_source_overlap_before_resize(monkeypatch):
    data = encoded((1024, 768), mode='RGBA')
    convert, resize, asarray = Image.Image.convert, Image.Image.resize, processing.np.asarray
    refs, events = {}, []
    with Image.open(BytesIO(data)) as source:
        def conversion(image, *args, **kwargs):
            result = convert(image, *args, **kwargs)
            if image is source:
                assert source.im is not result.im
                refs['rgb'] = weakref.ref(result)
                events.append('full_source_and_rgb')
            return result
        def resizing(image, *args, **kwargs):
            if image is refs['rgb']():
                assert source.im is not None and source.size == (1024, 768)
                assert image.size == source.size and image.mode == 'RGB'
                events.append('resize_with_source_alive')
            return resize(image, *args, **kwargs)
        def normalization(image, *args, **kwargs):
            assert refs['rgb']() is None  # full RGB temporary released after resize
            assert source.im is not None  # caller still owns full decoded source
            events.append('normalize_with_source_alive')
            return asarray(image, *args, **kwargs)
        monkeypatch.setattr(Image.Image, 'convert', conversion)
        monkeypatch.setattr(Image.Image, 'resize', resizing)
        monkeypatch.setattr(processing.np, 'asarray', normalization)
        patches, h, w, _ = processing.image_patch_array(source, ModelConfig(vision_n_layers=1))
        assert patches.nbytes == h * w * 14 * 14 * 3 * 4
    assert events == ['full_source_and_rgb', 'resize_with_source_alive', 'normalize_with_source_alive']
    assert source.fp is None
    # Pillow's context manager closes the stream, not the decoded ImagingCore.
    assert source.getpixel((0, 0)) == (0, 0, 0, 0)


def test_validation_failure_closes_decoder(monkeypatch):
    data = encoded((64, 64), fmt='JPEG')
    images = []
    def fail(image, *args, **kwargs):
        images.append(image)
        raise OSError('controlled decode failure')
    monkeypatch.setattr(ImageFile.ImageFile, 'load', fail)
    with pytest.raises(OSError, match='controlled decode') as caught:
        mm.validate_encoded_image(data)
    assert caught.value is not None and images
    assert all(image.fp is None for image in images)


def test_conversion_failure_releases_worker_sources(monkeypatch):
    inp = prepare(encoded((64, 64)))
    images = []
    def fail(image, *args, **kwargs):
        image.load()
        images.append(image)
        raise RuntimeError('controlled RGB failure')
    monkeypatch.setattr(Image.Image, 'convert', fail)
    with pytest.raises(RuntimeError, match='controlled RGB') as caught:
        inp.encode_new(NS(config=inp._config), span_encoder=lambda *args: pytest.fail('encoder reached'))
    assert caught.value is not None and not inp._sources
    assert len(images) == 1 and images[0].fp is None
    # A retained conversion traceback still owns the source core, even with fp
    # closed. Once the exception and observer handles go, the image object goes.
    assert images[0].getpixel((0, 0)) == (0, 0, 0)
    ref = weakref.ref(images[0])
    images.clear()
    del caught
    import gc
    gc.collect()
    assert ref() is None
