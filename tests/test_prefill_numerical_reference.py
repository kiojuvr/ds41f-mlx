"""Model-free checks of test-only ordinary reference and physical collectors."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from tools.qualify_prefill_fp8_mlx import (
    ordinary_prefill_reference, paired_physical_metrics, physical_tensor_metrics,
)
from test_production_mtp import pair


def test_metrics_are_chunk_bounded_and_match_direct_reductions():
    a = np.arange(101, dtype=np.float32) / 10
    b = a + np.float32(.01)
    got = physical_tensor_metrics(a, b, chunk_elements=7)
    x, y = a.astype(np.float64), b.astype(np.float64)
    assert got['max_absolute'] == np.max(np.abs(x-y))
    assert got['rms_relative'] == pytest.approx(np.linalg.norm(x-y)/np.linalg.norm(y))
    assert got['cosine'] == pytest.approx(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)))
    assert got['mismatches'] == 101 and len(got['samples']) == 4
    assert got['samples'][0]['flat_index'] == 0
    assert 'status' not in got  # no invented numerical threshold
    json.dumps(got, allow_nan=False)


def test_metrics_do_not_treat_packed_bytes_as_float_values():
    a = np.array([0, 255], dtype=np.uint8)
    b = np.array([255, 0], dtype=np.uint8)
    got = physical_tensor_metrics(a, b, chunk_elements=1)
    assert got['mismatches'] == 2
    assert 'rms_relative' not in got
    assert not physical_tensor_metrics(a, b.astype(np.int32))['structure_equal']
    assert not physical_tensor_metrics(a, b.reshape(1, 2))['structure_equal']
    assert not physical_tensor_metrics(a, None)['structure_equal']
    assert physical_tensor_metrics(None, None)['structure_equal']
    with pytest.raises(ValueError):
        physical_tensor_metrics(a, b, chunk_elements=0)


def test_empty_zero_scalar_and_nonfinite_metrics_are_explicit():
    for a in (np.zeros(3, np.float32), np.empty(0, np.float32)):
        got = physical_tensor_metrics(a, a)
        assert got['max_absolute'] == 0 and got['rms_relative'] is None and got['cosine'] is None
    got = physical_tensor_metrics(np.array(2.), np.array(3.))
    assert got['elements'] == 1 and got['max_absolute'] == 1
    got = physical_tensor_metrics(np.array([np.nan, np.inf]), np.array([1., np.inf]))
    assert got['nonfinite_pairs'] == 2 and got['max_absolute'] is None
    json.dumps(got, allow_nan=False)


def test_mlx_bfloat_metrics_use_bounded_fp32_host_conversion():
    mx = pytest.importorskip('mlx.core')
    a = mx.array([0., 1., 2.], dtype=mx.bfloat16)
    b = mx.array([0., 1., 3.], dtype=mx.bfloat16)
    got = physical_tensor_metrics(a, b, chunk_elements=1)
    assert got['structure_equal'] and got['mismatches'] == 1
    assert got['max_absolute'] == 1 and got['rms_relative'] == pytest.approx(1/10**.5)


def test_collector_covers_every_layer_slot_and_preserves_ring_rotation():
    target, rings, _ = pair(193)
    other_target, other_rings, _ = pair(193)
    # Deliberately alter the last slot of the last layer and rotate only ring 2.
    other_target[39].cache[6][0, 1, 2] = -7
    other_rings[2].keys = np.roll(other_rings[2].keys, 1, axis=2)
    got = paired_physical_metrics(dict(target=target, rings=rings),
                                  dict(target=other_target, rings=other_rings), chunk_elements=5)
    assert got['status'] == 'OBSERVED_ONLY'
    assert [r['layer'] for r in got['target']] == list(range(40))
    assert all(len(r['slots']) == 7 for r in got['target'])
    assert [r['ring'] for r in got['rings']] == [0, 1, 2]
    assert got['target'][39]['slots'][6]['mismatches'] == 1
    assert got['rings'][2]['keys']['mismatches'] == 128
    assert got['rings'][0]['keys']['mismatches'] == 0
    assert got['rings'][2]['physical_positions'][0] == list(range(128, 193)) + list(range(65, 128))
    assert other_target[39].cache[6][0, 1, 2] == -7  # no mutation/normalization
    json.dumps(got, allow_nan=False)
    with pytest.raises(ValueError, match='incomplete 40-layer'):
        paired_physical_metrics(dict(target=target[:-1], rings=rings), dict(target=target, rings=rings))
    target[39].cache.pop()
    with pytest.raises(ValueError, match='seven-slot'):
        paired_physical_metrics(dict(target=target, rings=rings), dict(target=other_target, rings=other_rings))


def reference_model():
    calls = []
    class Cache:
        def __init__(self): self.end = 0
        def size(self): return self.end
    class LM:
        layers = [object() for _ in range(40)]
        _config = SimpleNamespace(dspark_target_layer_ids=(1, 14, 39))
        def make_cache(self): return [Cache() for _ in range(40)]
        def make_mtp_cache(self):
            return [SimpleNamespace(offset=0, keys=None) for _ in range(3)]
        def _forward(self, ids, *, cache, inputs_embeds, return_dspark_hidden):
            calls.append(('ordinary', inputs_embeds, return_dspark_hidden))
            for c in cache: c.end = ids.shape[1]
            return np.ones((1, ids.shape[1], 8)), np.ones((1, ids.shape[1], 12))
        def dspark_append_context(self, hidden, rings, *, start_offset):
            calls.append(('native', start_offset))
            for r in rings: r.offset, r.keys = hidden.shape[1], hidden
        def __call__(self, *a, **kw):
            raise AssertionError('implicit priming must be bypassed')
    class Model:
        language_model = LM()
        def get_input_embeddings(self, ids, **kwargs):
            calls.append(('embedding', kwargs))
            return SimpleNamespace(inputs_embeds=np.ones((1, ids.shape[1], 4)))
    return Model(), calls


def test_reference_invokes_existing_ordinary_interfaces_on_fresh_state():
    model, calls = reference_model()
    ids = np.array([[10, 11, 12]], np.int32)
    first = ordinary_prefill_reference(model, ids)
    second = ordinary_prefill_reference(model, ids)
    assert first['target'][0] is not second['target'][0]
    assert first['rings'][0] is not second['rings'][0]
    assert [c[0] for c in calls] == ['embedding', 'ordinary', 'native'] * 2
    assert calls[1] == ('ordinary', None, True)
    assert calls[2] == ('native', 0)
    with pytest.raises(TypeError):
        ordinary_prefill_reference(model, ids, cache=first['target'])
    with pytest.raises(TypeError):
        ordinary_prefill_reference(model, ids, inputs_embeds=first['embeddings'])


def test_reference_image_embedding_is_recomputed_and_input_bounds_fail_closed():
    model, calls = reference_model()
    ids = np.array([[10, 1, 1, 12]], np.int32)
    got = ordinary_prefill_reference(model, ids, pixel_values=np.ones((1, 2)),
                                     image_grids=[(1, 1)], image_spans=[(1, 2)], image_types=[[0, 3]])
    assert calls[1][1] is got['embeddings']
    for bad in (np.zeros((2, 4)), np.zeros((1, 4097)), np.zeros((1, 0))):
        with pytest.raises(ValueError): ordinary_prefill_reference(model, bad)
    with pytest.raises(ValueError, match='pixels/image'):
        ordinary_prefill_reference(model, ids, pixel_values=np.ones((1, 2)))
    with pytest.raises(ValueError, match='incomplete'):
        ordinary_prefill_reference(model, ids, pixel_values=np.ones((1, 2)),
                                   image_grids=[(1, 1)], image_spans=[(3, 2)], image_types=[[0, 3]])
    model.language_model.make_cache = lambda: [SimpleNamespace(size=lambda: 2)] * 40
    with pytest.raises(ValueError, match='fresh'):
        ordinary_prefill_reference(model, ids)
