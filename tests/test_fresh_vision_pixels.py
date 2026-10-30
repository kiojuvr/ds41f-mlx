"""Sequential pixel lifetime, not cumulative source-pixel admission."""
import asyncio
from types import SimpleNamespace as NS
import weakref

import mlx.core as mx
import pytest
from PIL import Image

from ds41f_mlx.model_execution import processing
from ds41f_mlx.runtime import multimodal as mm
from test_metadata_first_vision import prepare, source


def sources():
    return [source(size=(2048, 2048))]*4 + [source(size=(512, 512))]


def test_above_former_pixel_limit_decodes_sequentially(monkeypatch):
    inputs = sources()
    refs = []
    opened, validate = Image.open, mm.validate_encoded_image
    def open_image(*args, **kwargs):
        image = opened(*args, **kwargs)
        refs.append(weakref.ref(image))
        return image
    def validate_image(*args, **kwargs):
        assert all(r() is None for r in refs)
        assert kwargs['remaining_pixels'] is None
        return validate(*args, **kwargs)
    monkeypatch.setattr(Image, 'open', open_image)
    monkeypatch.setattr(mm, 'validate_encoded_image', validate_image)
    inp = prepare(inputs, ordinary=True)
    assert len(refs) == 10 and all(r() is None for r in refs)
    assert sum(s.source_pixels for s in inp.spans) == 17_039_360
    assert sum(s.grid[0]*s.grid[1] for s in inp.spans) == 36_117
    inp.validate_workload(0)
    inp.release_sources()


@pytest.mark.parametrize('fail', [False, True])
def test_above_former_pixel_limit_normalizes_sequentially_and_cleans_up(monkeypatch, fail):
    inp = prepare(sources(), ordinary=True)
    patch_refs, row_refs = [], []
    pack = processing.image_patch_array
    def patches(*args):
        assert all(r() is None for r in patch_refs)
        result = pack(*args)
        patch_refs.append(weakref.ref(result[0]))
        return result
    def encode(pixels, h, w, kinds):
        if fail and len(patch_refs) == 5:
            raise asyncio.CancelledError('during final image')
        row = mx.zeros((len(kinds), 4), dtype=mx.bfloat16)
        row_refs.append(weakref.ref(row))
        return row
    monkeypatch.setattr(processing, 'image_patch_array', patches)
    if fail:
        with pytest.raises(asyncio.CancelledError):
            inp.encode_new(NS(config=inp._config), span_encoder=encode)
        assert all(r() is None for r in row_refs)
    else:
        embeddings = inp.encode_new(NS(config=inp._config), span_encoder=encode)
        assert len(embeddings.rows) == 5
        assert all(r() is not None for r in row_refs)  # outputs really coexist
        del embeddings
        assert all(r() is None for r in row_refs)
    assert all(r() is None for r in patch_refs) and not inp._sources


def test_other_profiles_retain_pixel_guard(monkeypatch):
    monkeypatch.setattr('ds41f_mlx.execution_identity.numerical_profile', lambda: 'other-profile')
    with pytest.raises(ValueError, match='pixel budget'):
        prepare(sources(), ordinary=True)
    inp = prepare([source()], ordinary=True)
    assert not inp.sequential_pixel_validation
    inp.release_sources()


def test_remaining_pixels_is_still_a_legacy_validation_boundary():
    src = source(size=(32, 32))
    with pytest.raises(ValueError, match='pixel budget'):
        mm.validate_encoded_image(src.data, remaining_pixels=1023)
    assert mm.validate_encoded_image(src.data, remaining_pixels=1024)['width'] == 32
