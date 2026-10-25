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


def test_layer0_live_prefix_measurement_detects_one_changed_retained_byte():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
    from omlx.patches.deepseek_v41.quantization import pack_activation
    from tools.qualify_prefill_fp8_mlx import layer0_window_observation, layer0_prefix_metrics
    cfg = SimpleNamespace(window_size=128, head_dim=32)
    cache = DeepseekV41Cache(0)
    cache[0] = mx.array([11], mx.int32)
    cache[1] = pack_activation(mx.ones((1, 11, 32)))
    a = layer0_window_observation(cache, cfg)
    cache[0] = mx.array([16], mx.int32)
    cache[1] = mx.concatenate([cache[1], pack_activation(mx.ones((1, 5, 32)))], 1)
    b = layer0_window_observation(cache, cfg)
    assert layer0_prefix_metrics(a, b, 11)['packed']['mismatches'] == 0
    cache[1][0, 0, 11] = cache[1][0, 0, 11] ^ mx.array(1, mx.uint8)
    changed = layer0_window_observation(cache, cfg)
    got = layer0_prefix_metrics(a, changed, 11)
    assert got['packed']['mismatches'] == got['decoded']['mismatches'] == 1
    assert got['decoded']['samples'][0]['flat_index'] == 11
    assert a['packed'][0, 0, 11] != changed['packed'][0, 0, 11]
    changed['absolute_positions'][0] = 1
    with pytest.raises(ValueError, match='same absolute'):
        layer0_prefix_metrics(a, changed, 11)


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('has_engram', [False, True])
def test_layer0_hooks_preserve_values_and_restore_on_exception(fail, has_engram):
    import mlx.core as mx
    import mlx.nn as nn
    import omlx.patches.deepseek_v41.language as lang
    from tools.qualify_prefill_fp8_mlx import layer0_numerical_handles
    def projection(n, k):
        w, s = mx.quantize(mx.ones((n, k)), group_size=32, bits=8, mode='mxfp8')
        return lang.QuantizedProjection(w, s, 8, 'mxfp8')
    class Engram(nn.Module):
        def __init__(self):
            super().__init__()
            self.wkv = projection(32, 32)
        def __call__(self, h):
            return self.wkv(h)
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wkv = projection(32, 32)
            self.kv_norm = lang.RMSNorm(32, 1e-6)
        def __call__(self, x):
            return lang.pack_activation(self.kv_norm(self.wkv.project_quantized(lang.quantize_activation(x))))
    e, attn = Engram(), Attention()
    layer = SimpleNamespace(attn=attn)
    if has_engram:
        layer.engram = e
    lm = SimpleNamespace(layers=[layer])
    x = mx.ones((1, 11, 32), mx.bfloat16)
    baseline = attn(e(x) if has_engram else x)
    originals = (Engram.__call__, Attention.__call__, lang.QuantizedProjection.__call__,
                 lang.QuantizedProjection.project_quantized, lang.RMSNorm.__call__, lang.pack_activation)
    try:
        with layer0_numerical_handles(lm) as handles:
            out = attn(e(x) if has_engram else x)
            if fail:
                raise RuntimeError('intentional observer unwind')
    except RuntimeError as exc:
        assert fail and 'intentional' in str(exc)
    assert originals == (Engram.__call__, Attention.__call__, lang.QuantizedProjection.__call__,
                         lang.QuantizedProjection.project_quantized, lang.RMSNorm.__call__, lang.pack_activation)
    assert physical_tensor_metrics(out, baseline)['mismatches'] == 0
    prepack = handles['prepack']
    changed = prepack + mx.array(1, prepack.dtype)
    assert physical_tensor_metrics(prepack, changed)['mismatches'] == prepack.size
    if has_engram:
        assert handles['engram_input'] is x
    else:
        assert not any(name.startswith('engram') for name in handles)
        assert handles['attention_input'] is x
    assert handles['attention_wkv_qdq'].shape == (1, 11, 32)


def test_admitted_checkpoint_has_no_layer0_engram():
    # Metadata only: the full-model attempt exposed an assumption hidden by
    # the old all-Engram tiny fixture. No weights or model instance needed.
    from pathlib import Path
    from omlx.patches.deepseek_v41.config import ModelConfig
    config = ModelConfig.from_dict(json.loads(Path(
        '/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/config.json'
    ).read_text()))
    assert list(config.engram_layer_ids) == [1, 14]
    assert 0 not in config.engram_layer_ids


@pytest.mark.parametrize('fail', [False, True])
def test_installed_layer0_constructor_and_shared_projection_hooks(monkeypatch, fail):
    """Real constructor/dispatch contracts; replace allocation-heavy leaves only."""
    import mlx.core as mx
    import mlx.nn as nn
    import omlx.patches.deepseek_v41.language as lang
    from pathlib import Path
    from omlx.patches.deepseek_v41.config import ModelConfig
    from tools.qualify_prefill_fp8_mlx import layer0_numerical_handles
    config = ModelConfig.from_dict(json.loads(Path(
        '/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/config.json'
    ).read_text()))
    def linear(*args, **kwargs):
        w, s = mx.quantize(mx.ones((32, 32)), group_size=32, bits=8, mode='mxfp8')
        return lang.QuantizedProjection(w, s, 8, 'mxfp8')
    with monkeypatch.context() as m:
        m.setattr(lang.nn, 'Linear', linear)
        m.setattr(lang, 'MoE', lambda c: nn.Module())
        layer = lang.Block(config, 0)
    assert not any(hasattr(layer, name) for name in ('engram',))
    assert not any(hasattr(layer.attn, name) for name in ('compressor', 'indexer'))
    originals = (lang.Block.__call__, lang.Attention.__call__,
                 lang.QuantizedProjection.project_quantized, lang.pack_activation)
    x = mx.ones((1, 11, 32), mx.bfloat16)
    baseline = layer.attn._input_projections(x)
    try:
        with layer0_numerical_handles(SimpleNamespace(layers=[layer])) as handles:
            actual = layer.attn._input_projections(x)
            if fail:
                raise RuntimeError('intentional')
    except RuntimeError as exc:
        assert fail and str(exc) == 'intentional'
    assert originals == (lang.Block.__call__, lang.Attention.__call__,
                         lang.QuantizedProjection.project_quantized, lang.pack_activation)
    assert all(physical_tensor_metrics(a, b)['mismatches'] == 0
               for a, b in zip(actual, baseline))
    assert handles['attention_wkv_qdq'].shape == x.shape
    assert handles['attention_wkv_output'] is actual[1]


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


@pytest.mark.parametrize('patch_delta', [0., .25])
def test_case_b_production_vision_independent_inputs_and_unwind(monkeypatch, patch_delta):
    from io import BytesIO
    import mlx.core as mx
    from PIL import Image
    from omlx.patches.deepseek_v41 import model as native
    from omlx.patches.deepseek_v41.config import ModelConfig
    from ds41f_mlx.runtime.multimodal import prepare_multimodal
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.model_execution import processing
    import tools.qualify_prefill_fp8_mlx as tool
    config = ModelConfig(vision_n_layers=1, vision_dim=16, vision_n_heads=4,
        vision_inter_dim=32, dim=8, vision_min_pixels=42*42, preserve_mtp=True)
    buf = BytesIO()
    Image.new('RGB', (42, 42), (100, 80, 60)).save(buf, format='PNG')
    data = buf.getvalue()
    mm = prepare_multimodal([7, config.image_token_id, 8, 9],
        [SimpleNamespace(kind='bytes', data=data)], config)
    ids = list(mm.token_ids)
    candidate, state_config = native_meaningful_pair(len(ids)-2)
    independent, _ = native_meaningful_pair(len(ids)-2)
    original_layers = tuple(object() for _ in range(40))
    lm = SimpleNamespace(layers=list(original_layers), _config=state_config,
                        embed=lambda ids: mx.zeros((*ids.shape, config.dim)))
    monkeypatch.setattr(native, 'LanguageModel', lambda c: lm)
    model = native.Model(config)
    events = []
    class Backend:
        _model = model
        _encode_new_images = ProductionMTPBackend._encode_new_images
        def _start(self, rec, request, tokenizer, trace, *, checkpoint_capture):
            events.append('candidate')
            self._encode_new_images(mm, 0)
            lm.layers[0] = object()
            try:
                checkpoint_capture(candidate['target'], candidate['rings'], ids[:-2])
                pytest.fail('must stop at N-2')
            finally:
                lm.layers[:] = original_layers
                events.append('unwound')
    original_patch = processing.image_patch_array
    calls = []
    def patches(*args):
        p, h, w, kinds = original_patch(*args)
        calls.append(p)
        return (p + patch_delta if len(calls) == 2 else p), h, w, kinds
    monkeypatch.setattr(processing, 'image_patch_array', patches)
    def reference(model, input_ids, **kwargs):
        assert events == ['candidate', 'unwound']
        assert all(a is b for a, b in zip(lm.layers, original_layers))
        assert input_ids.tolist()[0] == ids[:-2]
        assert kwargs['image_spans'] == [(mm.spans[0].start, mm.spans[0].length)]
        independent['token_ids'] = tuple(ids[:-2])
        return dict(independent, embeddings=model.get_input_embeddings(input_ids, **kwargs).inputs_embeds)
    monkeypatch.setattr(tool, 'ordinary_prefill_reference', reference)
    request = SimpleNamespace(token_ids=ids, multimodal=mm)
    got = tool.observe_case_b_n2(Backend(), request, None, identity='synthetic', source_bytes=data)
    assert len(calls) == 2 and calls[0] is not calls[1]
    assert got['boundary']['comparable']
    assert len(got['boundary']['metrics']['target']) == 40
    assert len(got['boundary']['metrics']['rings']) == 3
    assert got['image']['patches']['geometry_equal']
    assert (got['image']['patches']['patches']['mismatches'] == 0) == (patch_delta == 0)
    assert (got['image']['embeddings']['max_absolute'] == 0) == (patch_delta == 0)
    assert [m['kind'] for m in got['image']['embeddings_by_kind']] == [0, 1, 2, 3]
    assert not mm._sources
    with pytest.raises(ValueError, match='source identity'):
        tool.observe_case_b_n2(Backend(), request, None, identity='synthetic', source_bytes=b'wrong')
    request.token_ids = ids[:mm.spans[0].start+mm.spans[0].length+1]
    with pytest.raises(ValueError, match='boundary mismatch'):
        tool.observe_case_b_n2(Backend(), request, None, identity='synthetic', source_bytes=data)


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


@pytest.mark.parametrize('changed', [False, True, 'restore'])
def test_case_c_layer0_lane_observes_live_restore_and_stops_on_modified_prefix(monkeypatch, changed):
    from contextlib import nullcontext
    import tools.qualify_prefill_fp8_mlx as tool
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpoint
    if changed == 'restore':
        restore = PairedCheckpoint.restore
        def modified_restore(*args, **kwargs):
            target, rings, tokens = restore(*args, **kwargs)
            target[0][1][0, 0, 0] = 0
            return target, rings, tokens
        monkeypatch.setattr(PairedCheckpoint, 'restore', modified_restore)
    seed, cfg = native_meaningful_pair(11)
    current, _ = native_meaningful_pair(16)
    independent, _ = native_meaningful_pair(16)
    for state in (seed, current, independent):
        c = state['target'][0]
        c[1] = c[1][:, :c.size()]
    lm = SimpleNamespace(layers=[object() for _ in range(40)], _config=cfg)
    backend = SimpleNamespace(_model=SimpleNamespace(language_model=lm))
    seed_request = SimpleNamespace(token_ids=list(range(13)), multimodal=None)
    request = SimpleNamespace(token_ids=list(range(18)), multimodal=None)
    events = []
    def start(rec, req, tokenizer, trace, *, checkpoint_capture):
        C, end = len(rec.canonical), len(req.token_ids)-2
        events.append((C, end))
        DeferredPrefillAppend.execute_all(SimpleNamespace(C=C, T=end))
        if not C:
            state = seed if end == 11 else current
            rec.cache, rec.rings = state['target'], state['rings']
        else:
            # Simulate a live append, never alter the immutable saved seed.
            rec.cache[:], rec.rings[:] = current['target'], current['rings']
            if changed:
                rec.cache[0][1][0, 0, 0] = 0
        checkpoint_capture(rec.cache, rec.rings, req.token_ids[:-2])
    backend._start = start
    monkeypatch.setattr(DeferredPrefillAppend, 'execute_all', lambda app: None)
    monkeypatch.setattr(tool, 'layer0_numerical_handles', lambda lm: nullcontext({}))
    def reference(model, ids):
        assert not changed and events == [(0, 11), (11, 16), (0, 16)]
        assert ids.tolist() == [list(range(16))]
        return independent
    monkeypatch.setattr(tool, 'ordinary_prefill_reference', reference)
    got = tool.observe_case_c_n2(backend, seed_request, request, None, identity='test', layer0=True)
    if changed == 'restore':
        assert got['comparisons']['A_B0']['packed']['mismatches'] == 1
        assert got['classifications'] == ['UNRESOLVED', 'PARTIAL']
        assert events == [(0, 11)]
        return
    assert got['comparisons']['A_B0']['packed']['mismatches'] == 0
    assert got['comparisons']['A_B1']['packed']['mismatches'] == int(changed)
    if changed:
        assert got['classifications'] == ['PREFIX_MODIFIED', 'PARTIAL']
        assert events == [(0, 11), (11, 16)]
        assert 'C' not in got['observations']
    else:
        assert got['classifications'] == ['PREFIX_PRESERVED']
        assert set(got['observations']) == {'A', 'B0', 'B1', 'C', 'D'}
        assert got['executions']['C'] == [dict(start=0, end=16, rows=16)]
        assert got['comparisons']['C_D']['decoded']['mismatches'] == 0


@pytest.mark.parametrize('fault', [None, 'prefix', 'suffix', 'frontier', 'mutation'])
def test_case_c_real_restore_adapter_boundaries_and_snapshot(monkeypatch, fault):
    import tools.qualify_prefill_fp8_mlx as tool
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpoint
    seed, config = native_meaningful_pair(11)
    current, _ = native_meaningful_pair(17)
    independent, _ = native_meaningful_pair(17)
    layers = tuple(object() for _ in range(40))
    lm = SimpleNamespace(layers=list(layers), _config=config)
    seed_request = SimpleNamespace(token_ids=list(range(13)), multimodal=None)
    request = SimpleNamespace(token_ids=list(range(19)), multimodal=None)
    if fault == 'prefix':
        request.token_ids[0] = -1
    if fault == 'suffix':
        request.token_ids = seed_request.token_ids
    events, retained = [], []
    restore = PairedCheckpoint.restore
    def tracked_restore(saved, *args, **kwargs):
        retained.append(saved)
        target, rings, tokens = restore(saved, *args, **kwargs)
        if fault == 'frontier':
            tokens.pop()
        return target, rings, tokens
    monkeypatch.setattr(PairedCheckpoint, 'restore', tracked_restore)
    monkeypatch.setattr(DeferredPrefillAppend, 'execute_all', lambda app: None)
    class Backend:
        _model = SimpleNamespace(language_model=lm)
        def _start(self, rec, req, tokenizer, trace, *, checkpoint_capture):
            start, end = len(rec.canonical), len(req.token_ids)-2
            events.append((start, end))
            if start:
                assert start == 11 and all(c.size() == 11 for c in rec.cache)
                assert all(r.offset == 11 for r in rec.rings)
                if fault == 'mutation':
                    retained[0].target[0].cache[1][0, 0, 0] = 255
            DeferredPrefillAppend.execute_all(SimpleNamespace(C=start, T=end))
            state = seed if start == 0 else current
            checkpoint_capture(state['target'], state['rings'], req.token_ids[:-2])
    def reference(model, input_ids):
        assert events == [(0, 11), (11, 17)]
        assert input_ids.tolist() == [list(range(17))]
        return independent
    monkeypatch.setattr(tool, 'ordinary_prefill_reference', reference)
    if fault:
        with pytest.raises(ValueError):
            tool.observe_case_c_n2(Backend(), seed_request, request, None, identity='test')
        if fault in ('prefix', 'suffix'):
            assert events == []
    else:
        got = tool.observe_case_c_n2(Backend(), seed_request, request, None, identity='test')
        assert got['comparable'] and got['frontiers'] == [[17, 17], [17, 17]]
        assert got['restore']['snapshot_unchanged']
        assert got['restore']['distinct_mutable_containers']
        assert got['restore']['append_executions'] == [dict(start=11, end=17, rows=6)]
