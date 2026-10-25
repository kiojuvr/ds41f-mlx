"""Model-free checks of test-only ordinary reference and physical collectors."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from tools.qualify_prefill_fp8_mlx import (
    ordinary_prefill_reference, paired_physical_metrics, physical_tensor_metrics,
    meaningful_target_metrics, paired_boundary_metrics, observe_case_a_n2,
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


def native_meaningful_pair(end=193):
    import mlx.core as mx
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
    from omlx.patches.deepseek_v41.quantization import pack_activation
    from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache
    config = SimpleNamespace(window_size=128, compress_ratios=[4]*40,
                             kv_source_layers=(0, 39), index_source_layers=(0, 39),
                             engram_layer_ids=(0,), engram_max_ngram_size=3,
                             head_dim=64, index_head_dim=32)
    caches = []
    for layer in range(40):
        ratio = 4 if layer in config.kv_source_layers else 0
        c = DeepseekV41Cache(ratio)
        c[0] = mx.array([end], mx.int32)
        for slot, bits, group, scale, width, rows in (
                (1, 8, 32, False, 64, min(end, 128)),
                (2, 4, 16, True, 64, end//4 if ratio else 0),
                (3, 4, 32, False, 32, end//4 if ratio else 0)):
            # Explicit allocation padding, not meaningful committed rows.
            c[slot] = pack_activation(mx.ones((1, rows+3, width)), bits, group, scale)
        for slot in (4, 5):
            c[slot] = mx.ones((1, end%4 if ratio else 0, 64), mx.bfloat16)
        c[6] = mx.zeros((1, 2 if layer == 0 else 0), mx.int64)
        caches.append(c)
    rings = [DSparkContextCache(128) for _ in range(3)]
    for ring in rings:
        ring.append(mx.arange(end, dtype=mx.float32).reshape(1, 1, end, 1))
    return dict(target=caches, rings=rings, token_ids=tuple(range(end))), config


def test_meaningful_decoding_masks_padding_and_catches_final_source(monkeypatch):
    import mlx.core as mx
    import omlx.patches.deepseek_v41.quantization as quant
    a, cfg = native_meaningful_pair()
    b, _ = native_meaningful_pair()
    # Padding must differ physically without becoming meaningful numerical error.
    b['target'][39][2][:, -1] = 0
    # Last meaningful index row of last layer must NOT be hidden by padding.
    b['target'][39][3][:, 47] = quant.pack_activation(mx.full((1, 32), 2.), 4, 32)
    calls = []
    original = quant.unpack_activation
    def bounded(packed, *args, **kwargs):
        calls.append(packed.shape[1])
        assert packed.shape[1] <= 16
        return original(packed, *args, **kwargs)
    monkeypatch.setattr(quant, 'unpack_activation', bounded)
    got = paired_physical_metrics(a, b, config=cfg, chunk_elements=11)
    assert calls and max(calls) == 16
    assert len(got['target']) == 40 and len(got['rings']) == 3
    last = got['target'][39]
    assert last['slots'][2]['mismatches'] > 0
    meaningful = last['meaningful']
    assert meaningful['selectors_equal']
    assert meaningful['selectors'][0]['window_positions'] == [65, 193]
    assert meaningful['slots'][2]['meaningful_rows'] == 48
    assert all(c['metrics']['max_absolute'] == 0 for c in meaningful['slots'][2]['chunks'])
    assert meaningful['slots'][3]['chunks'][-1]['metrics']['max_absolute'] > 0
    assert meaningful['slots'][4]['meaningful_rows'] == 1
    assert got['target'][38]['meaningful']['slots'][2]['meaningful_rows'] == 0
    assert got['rings'][2]['mapping_equal']
    assert int(b['target'][39][2][0, -1, 0].item()) == 0
    json.dumps(got, allow_nan=False)


def test_meaningful_layout_frontier_and_final_ring_negatives():
    import mlx.core as mx
    a, cfg = native_meaningful_pair()
    b, _ = native_meaningful_pair()
    b['rings'][2].keys = mx.concatenate([b['rings'][2].keys[:, :, 1:], b['rings'][2].keys[:, :, :1]], axis=2)
    got = paired_physical_metrics(a, b, config=cfg)
    assert got['rings'][2]['keys']['mismatches'] == 128
    b['rings'][2].offset += 1
    assert not paired_physical_metrics(a, b)['rings'][2]['mapping_equal']
    b['rings'][2].keys = b['rings'][2].keys[:, :, :-1]
    with pytest.raises(ValueError, match='ring 2'):
        paired_physical_metrics(a, b)
    b['target'][39][3] = b['target'][39][3][:, :47]
    with pytest.raises(ValueError, match='slot 3'):
        meaningful_target_metrics(a['target'][39], b['target'][39], 39, cfg)
    b['target'][39][3] = a['target'][39][3]
    b['target'][39][0] = mx.array([192], mx.int32)
    assert not meaningful_target_metrics(a['target'][39], b['target'][39], 39, cfg)['selectors_equal']
    b['target'][39].left_padding = mx.array([1])
    with pytest.raises(ValueError, match='unpadded singleton'):
        meaningful_target_metrics(a['target'][39], b['target'][39], 39, cfg)


def test_boundary_guard_preserves_transient_frontiers_without_settlement():
    a, cfg = native_meaningful_pair(193)
    target_ref, _ = native_meaningful_pair(193)
    earlier, _ = native_meaningful_pair(192)
    # P5 transient: Target consumed terminal, native rings still own N-1.
    a['rings'] = earlier['rings']
    earlier_ref, _ = native_meaningful_pair(192)
    reference = dict(target=target_ref['target'], rings=earlier_ref['rings'], token_ids=target_ref['token_ids'])
    arrays = [tuple(c.cache) for c in a['target']]
    ring_keys = [r.keys for r in a['rings']]
    got = paired_boundary_metrics(a, target_ref, boundary='p5-terminal', config=cfg)
    assert not got['comparable'] and 'metrics' not in got
    got = paired_boundary_metrics(a, reference, boundary='p5-terminal', config=cfg)
    assert got['comparable'] and got['frontiers'] == [[193, 192], [193, 192]]
    assert not paired_boundary_metrics(a, reference, boundary='n-1-seal', config=cfg)['comparable']
    wrong = dict(reference, token_ids=(*reference['token_ids'][:-1], -1))
    assert not paired_boundary_metrics(a, wrong, boundary='p5-terminal', config=cfg)['comparable']
    with pytest.raises(ValueError, match='canonical consumed identity'):
        paired_boundary_metrics(a, dict(reference, token_ids=()), boundary='p5-terminal', config=cfg)
    for c, before in zip(a['target'], arrays):
        assert all(x is y for x, y in zip(c.cache, before))
    assert all(r.keys is k and r.offset == 192 for r, k in zip(a['rings'], ring_keys))


def test_case_a_adapter_uses_capture_then_fresh_reference_after_unwind(monkeypatch):
    import tools.qualify_prefill_fp8_mlx as tool
    candidate, config = native_meaningful_pair()
    independent, _ = native_meaningful_pair()
    # Actual P6 _set_cache_history retains the hasher's NumPy int64 array;
    # ordinary _forward explicitly converts history to MLX int64.
    candidate['target'][0][6] = np.zeros((1, 2), np.int64)
    original_layers = tuple(object() for _ in range(40))
    lm = SimpleNamespace(layers=list(original_layers), _config=config)
    events = []
    class Backend:
        _model = SimpleNamespace(language_model=lm)
        def _start(self, rec, request, tokenizer, trace, *, checkpoint_capture):
            assert rec.cache is None and rec.canonical == []
            lm.layers[0] = object()  # stand-in for native temporary tap wrapper
            try:
                events.append('candidate')
                checkpoint_capture(candidate['target'], candidate['rings'], request.token_ids[:-2])
                pytest.fail('adapter must stop before N-1/P5')
            finally:
                lm.layers[:] = original_layers
                events.append('unwound')
    def reference(model, input_ids):
        assert all(a is b for a, b in zip(lm.layers, original_layers))
        assert input_ids.tolist()[0] == list(range(193))
        assert events == ['candidate', 'unwound']
        events.append('reference')
        return independent
    monkeypatch.setattr(tool, 'ordinary_prefill_reference', reference)
    before = [tuple(c.cache) for c in candidate['target']]
    request = SimpleNamespace(token_ids=list(range(195)), multimodal=None)
    result = observe_case_a_n2(Backend(), request, None, identity='synthetic')
    assert result['comparable'] and len(result['metrics']['target']) == 40
    assert len(result['metrics']['rings']) == 3 and events[-1] == 'reference'
    history = result['metrics']['target'][0]['meaningful']['slots'][6]['exact']
    assert history['structure_equal'] and history['mismatches'] == 0
    assert not physical_tensor_metrics(candidate['target'][0][6].astype(np.int32),
                                       independent['target'][0][6])['structure_equal']
    assert all(x is y for c, old in zip(candidate['target'], before) for x, y in zip(c.cache, old))
    request.multimodal = object()
    with pytest.raises(ValueError, match='canonical text'):
        observe_case_a_n2(Backend(), request, None, identity='synthetic')


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
