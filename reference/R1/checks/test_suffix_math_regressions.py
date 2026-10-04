"""Operation recording exercises the actual adapter, without MLX/weights."""
from types import SimpleNamespace as NS
import tempfile
import unittest
from pathlib import Path

from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import *
from test_prefill_fp8_mlx_p1_p2 import FakeLanguageModel, FakeMx, FakeRowTensor
from ds41f_mlx.prefill_fp8_mlx.omlx_suffix_math import OmlxV41SuffixMath, SuffixMathError


class Tensor:
    def __init__(self, rows, width=16, values=None, shape=None):
        self.shape = shape or (1, rows, width)
        self.dtype = 'bf16'
        self.values = list(range(rows)) if values is None else list(values)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) > 1 and isinstance(key[1], slice):
            values = self.values[key[1]]
            return Tensor(len(values), self.shape[-1], values)
        return self

    def reshape(self, *shape):
        shape = tuple(16 if x == -1 else x for x in shape)
        return Tensor(shape[1], values=self.values, shape=shape)

    def flatten(self, _):
        return Tensor(self.shape[1], values=self.values)

    def astype(self, _):
        return self

    def concat_rows(self, values):
        return Tensor(sum(v.shape[1] for v in values), values=[row for v in values for row in v.values])

    def __add__(self, _): return self
    def __sub__(self, _): return self
    def __mul__(self, _): return self
    def __ge__(self, _): return self
    def __le__(self, _): return self
    def __gt__(self, _): return self
    def __and__(self, _): return self
    def __neg__(self): return self


class Ops:
    """Records semantic operations and row identities, not marker flags."""
    float32 = 'fp32'
    uint8 = 'uint8'
    int32 = 'int32'

    def __init__(self):
        self.events = []
        self.attended = []

    def arange(self, start, end=None):
        if end is None: start, end = 0, start
        return Tensor(end-start, values=range(start, end))

    def zeros(self, shape, _dtype): return Tensor(shape[1], shape[-1], values=[], shape=shape)
    def concatenate(self, arrays, _axis):
        return Tensor(sum(a.shape[1] for a in arrays), values=[v for a in arrays for v in a.values])
    def maximum(self, a, _): return a
    def where(self, _, a, b): return a if isinstance(a, Tensor) else b
    def einsum(self, _, a, b): return a
    def sort(self, a, axis=-1): return a
    def argsort(self, a, axis=-1): return a
    def take_along_axis(self, a, _, axis=-1): return a

    def hc_pre_norm(self, h, pre, weight, eps):
        self.events.append(('hc_pre_norm', h.shape[1]))
        return h
    def hc_mixes(self, h, *args): return h, h, h
    def hc_post(self, out, h, *args): return out
    def rope(self, a, pos, c, compressed, inverse=False):
        self.events.append(('rope', tuple(pos.values), compressed, inverse))
        return a
    def pack_activation(self, a, **kw):
        self.events.append(('pack', dict(kw), a.shape[1]))
        return a
    def quantize_activation(self, a, **kw): return a
    def packed_index_topk(self, q, key, weights, start, ratio, topk, **kw):
        self.events.append(('topk', start, q.shape[1], key.shape[1]))
        return Tensor(q.shape[1]), Tensor(q.shape[1], 2)
    def packed_index_scores(self, q, key, weights, start, ratio, candidates):
        self.events.append(('candidate_scores', start, q.shape[1]))
        return Tensor(q.shape[1])
    def packed_sparse_attention(self, q, kv, pooled, idx, ci, sink, scale):
        self.attended.append((kv, pooled, ci))
        return q


class RecordingMath(OmlxV41SuffixMath):
    def _ops(self, layer): return self.ops, self.ops


def recording_model():
    ops = Ops()
    c = NS(window_size=128, compress_ratios={i: 1 for i in range(40)}, kv_source_layers=(20,), index_source_layers=(20, 24, 28, 32, 36), candidate_source_layer=20, candidate_block_size=16, candidate_topk_blocks=2, index_topk=2, head_dim=16, index_head_dim=16, index_n_heads=1, n_heads=1, o_groups=1, o_lora_rank=16)
    def projection(name):
        def call(x, *args):
            ops.events.append((name, x.shape[1]))
            return x
        return call
    def inputs(x):
        ops.events.append(('input_projections', x.shape[1]))
        return x, x
    indexer = NS(wk=projection('wk'), k_norm=projection('k_norm'), wq_b=projection('index_q'), weights_proj=projection('index_weights'))
    attn = NS(_input_projections=inputs, compressor=projection('compressor'), indexer=indexer, q_norm=projection('q_norm'), kv_norm=projection('kv_norm'), wq_b=projection('query_projection'), attn_sink=None, wo_a=NS(weight=Tensor(16)), wo_b=projection('output_projection'))
    norm = NS(weight=None, eps=1e-6)
    layer = NS(attn=attn, attn_norm=norm, ffn_norm=norm, hc_attn_fn=None, hc_attn_scale=None, hc_attn_base=None, hc_ffn_fn=None, hc_ffn_scale=None, hc_ffn_base=None, ffn=lambda x, mask: projection('official_moe')(x))
    model = NS(layers=[layer] * 40, _config=c)
    math = RecordingMath(model)
    math.ops = ops
    return math, ops


class CommandRecordingMath(RecordingMath):
    """Bridge arena fake carry into the operation recorder, preserving row spans."""
    def publish_full_source(self, **kw):
        kw['h_full'] = Tensor(kw['rows'])
        kw['pre_full'] = Tensor(kw['rows'])
        for slot in (2, 3):
            if isinstance(kw['cache'][slot], list):
                kw['cache'][slot] = Tensor(0)
        return super().publish_full_source(**kw)

    def prepare_local_window(self, **kw):
        kw['h_rows'] = Tensor(kw['rows'], values=range(kw['absolute_start'], kw['absolute_start'] + kw['rows']))
        kw['pre_rows'] = Tensor(kw['rows'])
        return super().prepare_local_window(**kw)

    def execute_suffix_query(self, **kw):
        original_h, original_pre = kw['h_chunk'], kw['pre_chunk']
        count = original_h.shape[1]
        kw['h_chunk'] = Tensor(count, values=range(kw['absolute_start'], kw['absolute_start'] + count))
        kw['pre_chunk'] = Tensor(count)
        source2, source3 = kw['cache'][2:4]
        super().execute_suffix_query(**kw)
        if kw['layer_id'] == 20:
            assert kw['cache'][2] is source2 and kw['cache'][3] is source3
        return original_h, original_pre


class SuffixMathRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.planner = SweepPlanner(load_native_prefill_library(compile_native_prefill_library(Path(cls.tmp.name))))
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_no_ordinary_block_fallback(self):
        calls = []
        class Unknown:
            def __call__(self, *a): calls.append(a)
        math = OmlxV41SuffixMath(NS(layers=[Unknown()]))
        with self.assertRaises(SuffixMathError):
            math.execute_suffix_query(layer_id=0, h_chunk=None, pre_chunk=None, cache=None, shared={}, absolute_start=0, image_mask=None)
        self.assertEqual(calls, [])

    def test_stale_window_replaced_and_first_query_has_127_preceding_rows(self):
        math, ops = recording_model()
        cache = [500, Tensor(128, values=[-999]*128), None, None, None, None, None]
        rows = Tensor(127, values=range(500, 627))
        math.prepare_local_window(layer_id=21, h_rows=rows, pre_rows=rows, cache=cache, absolute_start=500, rows=127)
        self.assertEqual(cache[0], 500)
        self.assertEqual(cache[1].values, list(range(500, 627)))
        self.assertNotIn(-999, cache[1].values)
        self.assertIn(('rope', tuple(range(500, 627)), True, False), ops.events)
        self.assertEqual([event[0] for event in ops.events], ['hc_pre_norm', 'input_projections', 'kv_norm', 'rope', 'pack'])
        shared = {'kv': Tensor(1000), 'idx': Tensor(1)}
        math.execute_suffix_query(layer_id=21, h_chunk=Tensor(1, values=[627]), pre_chunk=Tensor(1), cache=cache, shared=shared, absolute_start=627, image_mask=None)
        self.assertEqual(math.query_old_lengths, [(21, 627, 127)])
        self.assertEqual(ops.attended[-1][0].values, list(range(500, 628)))
        self.assertEqual(ops.attended[-1][0].shape[1], 128)
        self.assertIn(('official_moe', 1), ops.events)

    def test_source_generation_once_and_queries_never_regenerate_source(self):
        math, ops = recording_model()
        cache = [100, None, Tensor(100), Tensor(100), 'pending4', 'pending5', None]
        shared = {}
        math.publish_full_source(layer_id=20, h_full=Tensor(8192), pre_full=Tensor(8192), cache=cache, shared=shared, absolute_start=100, rows=8192)
        self.assertEqual(set(shared), {'kv', 'index_k'})
        self.assertFalse(any(e[0] in ('topk', 'index_q', 'q_norm', 'candidate_scores') for e in ops.events))
        self.assertEqual(cache[0], 100)
        source2, source3 = cache[2], cache[3]
        self.assertEqual(source2.shape[1], 8292)
        self.assertEqual(source3.shape[1], 8292)
        self.assertEqual(cache[4:6], ['pending4', 'pending5'])
        math.prepare_local_window(layer_id=20, h_rows=Tensor(127), pre_rows=Tensor(127), cache=cache, absolute_start=5700, rows=127)
        for start, count in ((5827, 1), (5828, 127), (5955, 100)):
            math.execute_suffix_query(layer_id=20, h_chunk=Tensor(count), pre_chunk=Tensor(count), cache=cache, shared=shared, absolute_start=start, image_mask=None)
            self.assertIs(cache[2], source2)
            self.assertIs(cache[3], source3)
            self.assertEqual(shared['idx'].shape[1], count)
            self.assertEqual(shared['candidates'].shape[1], count)
        self.assertEqual(math.full_source_generation_count, 1)
        self.assertEqual([e for e in ops.events if e[0] == 'compressor'], [('compressor', 8192)])
        self.assertEqual([e for e in ops.events if e[0] == 'wk'], [('wk', 8192)])
        self.assertEqual([(e[1], e[2]) for e in ops.events if e[0] == 'topk'], [(5827, 1), (5828, 127), (5955, 100)])

    def test_refresh_requires_index_k_not_previous_idx(self):
        topo = PublicationTopology.v41_expected()
        self.assertEqual(topo.consumes_by_layer[24], ('kv', 'index_k', 'candidates'))
        self.assertEqual(topo.consumes_by_layer[25], ('kv', 'idx', 'candidates'))
        self.assertEqual(topo.consumes_by_layer[20], ())
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        manager = PublicationManager(RequestArena.from_plan(plan, token_ids=[]))
        manager.capture_layer_outputs(20, {'kv': Tensor(8192), 'index_k': Tensor(8192), 'candidates': Tensor(10)}, command_index=1, offset=8182, rows=10)
        manager.publish_frontier(next(c for c in plan.publication_commands if c.layer == 20))
        shared = manager.shared_for_span(24, 8182, 10, require_keys=topo.consumes_by_layer[24])
        self.assertIsNone(shared['idx'])
        math, ops = recording_model()
        math.execute_suffix_query(layer_id=24, h_chunk=Tensor(10), pre_chunk=Tensor(10), cache=[0, Tensor(127), None, None, None, None, None], shared=shared, absolute_start=8182, image_mask=None)
        self.assertEqual(shared['idx'].shape[1], 10)
        self.assertIn(('candidate_scores', 8182, 10), ops.events)

    def test_producer_private_source_until_planner_frontier(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        manager = PublicationManager(RequestArena.from_plan(plan, token_ids=[]))
        private = object()
        manager.capture_layer_outputs(20, {'kv': private, 'index_k': private}, command_index=1, keys=('kv', 'index_k'))
        self.assertIs(manager.producer_shared_for_span(20, 6000, 1)['kv'], private)
        self.assertIsNone(manager.shared_for_span(21, 6000, 1)['kv'])
        with self.assertRaises(PublicationError):
            manager.shared_for_span(21, 6000, 1, require_keys=('kv',))
        manager.publish_frontier(next(c for c in plan.publication_commands if c.layer == 20))
        self.assertIs(manager.shared_for_span(21, 6000, 1, require_keys=('kv',))['kv'], private)
        unavailable = PublicationManager(RequestArena.from_plan(plan, token_ids=[]))
        unavailable.capture_layer_outputs(20, {'kv': None}, command_index=1, keys=('kv',))
        unavailable.publish_frontier(next(c for c in plan.publication_commands if c.layer == 20))
        with self.assertRaises(PublicationError):
            unavailable.shared_for_span(21, 6000, 1, require_keys=('kv',))

    def test_three_live_wide_sweeps_all_40_frontiers(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        lm = FakeLanguageModel()
        live = None
        for sweep in range(3):
            setup = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=FakeMx()).prepare(plan, [1]*8192, continuation=live)
            math, ops = recording_model()
            recorder = CommandRecordingMath(math.language_model)
            recorder.ops = ops
            setup.block_runner.suffix_math = recorder
            setup.block_runner.execute_batch(plan.commands, setup.arena)
            caches = setup.block_runner.working_cache
            self.assertEqual(caches[20][2].shape[1], (sweep+1)*8192)
            self.assertEqual(caches[20][3].shape[1], (sweep+1)*8192)
            self.assertEqual([e for e in ops.events if e[0] == 'compressor'], [('compressor', 8192)])
            expected_spans = [(sweep*8192+c.offset, c.rows) for c in plan.encode_commands if c.layer == 20]
            self.assertEqual([(e[1], e[2]) for e in ops.events if e[0] == 'topk'], expected_spans)
            self.assertTrue(all(length == 127 for layer, start, length in recorder.query_old_lengths if start == sweep*8192 + next(c.offset for c in plan.encode_commands if c.layer == layer)))
            self.assertEqual([c[0] for c in caches], [(sweep+1)*8192]*40)
            self.assertEqual(setup.block_runner.suffix_math.full_source_generation_count, 1)
            self.assertEqual(setup.block_runner.full_cache_repack_count, 0)
            self.assertFalse(setup.block_runner.prefill_continuation_exported)
            live = LivePrefillContinuation.from_cache(caches)



if __name__ == '__main__': unittest.main()
