from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from ds41f_mlx.prefill_fp8_mlx import (
    DeferredPrefillAppend,
    LivePrefillContinuation,
    LivePrefillResult,
    validate_committed_cache,
    OfficialFP8MLXBlockRunner,
    P6AppendError,
    P6AppendPlanner,
    PublicationManager,
    RequestArena,
    SegmentMode,
    SweepCommandKind,
)
from ds41f_mlx.prefill_fp8_mlx.p6_append import _sweep_shell
from test_prefill_fp8_mlx_p1_p2 import FakeCache, FakeLanguageModel, FakeTensor, full_ready_cache


def fake_p6_app(lm, cache, tokens, *, committed_frontier=0, mx=None):
    return DeferredPrefillAppend.create(lm, cache, tokens, committed_frontier=committed_frontier, mx=mx if mx is not None else RecordingMx())


class P6DeferredAppendStructuralTests(unittest.TestCase):
    def modes(self, plan):
        return [(s.start, s.count, s.mode) for s in plan.segments]

    def test_pinned_geometry_fixtures(self):
        p = P6AppendPlanner()
        self.assertEqual(self.modes(p.fixture('tiny_16385')), [
            (0, 16384, SegmentMode.FINAL_ENCODER_DECODER),
            (16384, 1, SegmentMode.ORDINARY_COMPLETE_RANGE),
        ])
        self.assertEqual(self.modes(p.fixture('A_24577')), [
            (0, 16384, SegmentMode.ENCODER_SOURCE_ONLY),
            (16384, 8192, SegmentMode.FINAL_ENCODER_DECODER),
            (24576, 1, SegmentMode.ORDINARY_COMPLETE_RANGE),
        ])
        self.assertEqual(self.modes(p.fixture('B_fresh_49155')), [
            (0, 16384, SegmentMode.ENCODER_SOURCE_ONLY),
            (16384, 16384, SegmentMode.ENCODER_SOURCE_ONLY),
            (32768, 16384, SegmentMode.FINAL_ENCODER_DECODER),
            (49152, 1, SegmentMode.ORDINARY_COMPLETE_RANGE),
            (49153, 1, SegmentMode.ORDINARY_COMPLETE_RANGE),
            (49154, 1, SegmentMode.ORDINARY_COMPLETE_RANGE),
        ])
        self.assertEqual(self.modes(p.fixture('B_continued_C24578_T49155')), [
            (24578, 16384, SegmentMode.ENCODER_SOURCE_ONLY),
            (40962, 8192, SegmentMode.FINAL_ENCODER_DECODER),
            (49154, 1, SegmentMode.ORDINARY_COMPLETE_RANGE),
        ])

    def test_matched_geometry_non_deferred_control_uses_same_counts(self):
        p = P6AppendPlanner()
        candidate = p.fixture('B_fresh_49155')
        control = p.matched_control_plan(C=0, T=49155)
        self.assertEqual([s.count for s in control.segments], [s.count for s in candidate.segments])
        self.assertFalse(any(s.mode is SegmentMode.ENCODER_SOURCE_ONLY for s in control.segments))
        self.assertEqual([s.mode for s in control.segments[:3]], [SegmentMode.FINAL_ENCODER_DECODER] * 3)

    def test_source_only_zero_decoder_work_public_slot_frozen_and_once_per_range(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        app = fake_p6_app(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        first = app.execute_segment(app.plan.segments[0])
        self.assertEqual((app.E, app.D), (16384, 0))
        self.assertEqual({c[0] for c in [(cache[i][0],) for i in range(40)]}, {0})
        recs = first.runner.records
        self.assertEqual(sum(1 for r in recs if r.layer == 20 and r.invoked_full_source_publish), 1)
        self.assertEqual(sum(1 for r in recs if r.layer == 20 and r.invoked_block), 0)
        self.assertEqual(sum(1 for r in recs if r.layer is not None and 21 <= r.layer <= 39 and r.invoked_block), 0)
        source_only_commands = app.plan.segments[0].commands
        self.assertFalse([c for c in source_only_commands if c.layer is not None and c.layer >= 20 and c.kind.value == 'encode_rows'])

    def test_fake_p6_execution_uses_explicit_tensor_adapter(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        mx = RecordingMx()
        app = fake_p6_app(lm, cache, list(range(16384)), committed_frontier=0, mx=mx)
        app.execute_all()
        self.assertTrue(mx.repeat_calls)
        self.assertIsInstance(app.final_execution.arena.carry.current.value, FakeTensor)
        self.assertNotEqual(type(app.final_execution.arena.carry.current.value).__module__.split('.')[0], 'mlx')

    def test_source_boundary_eval_and_failure_are_real_boundaries(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        for layer in range(20):
            cache[layer][1] = FakeTensor(f'window{layer}', rows=128)
        for layer in (2, 8, 14, 20):
            for slot in (2, 3, 4, 5):
                cache[layer][slot] = FakeTensor(f'L{layer}s{slot}', rows=4)
        mx = RecordingMx()
        app = DeferredPrefillAppend.create(lm, cache, list(range(24577)), committed_frontier=0, mx=mx)
        app.begin()
        self.assertFalse(app.segment_records)
        app.execute_segment(app.plan.segments[0])
        self.assertTrue(mx.eval_calls)
        event = app.segment_records[0]
        self.assertTrue(event.materialized)
        self.assertTrue(app.segment_records[0].retired)

        bad_cache = full_ready_cache(0)
        bad_cache[0][1] = FakeTensor('must_eval', rows=128)
        bad = DeferredPrefillAppend.create(lm, bad_cache, list(range(24577)), committed_frontier=0, mx=RecordingMx(fail_eval=True))
        bad.begin()
        with self.assertRaises(Exception):
            bad.execute_segment(bad.plan.segments[0])
        self.assertEqual(bad.state.value, 'failed')

    def test_decoder_cone_bounded_before_and_through_all_swaps(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        app = fake_p6_app(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        app.execute_segment(app.plan.segments[0])
        segment = app.plan.segments[1]
        execn = app._make_segment_execution(segment)
        saw_detach = False
        for command in segment.commands:
            execn.runner.execute_command(command, execn.arena)
            if command.kind is SweepCommandKind.P6_SOURCE_COMPLETE_AND_DETACH_CONE:
                saw_detach = True
                self.assertLessEqual(execn.arena.carry.current.rows, 2541)
                self.assertLessEqual(execn.arena.carry.next.rows, 2541)
                self.assertLessEqual(execn.arena.carry.pre.rows, 2541)
                self.assertIsNone(execn.arena.encoder_final_h)
                self.assertIsNone(execn.arena.encoder_final_pre)
            if saw_detach and command.kind is SweepCommandKind.SWAP_HC_AFTER_LAYER and command.layer is not None and command.layer >= 20:
                for slot in (execn.arena.carry.current, execn.arena.carry.next, execn.arena.carry.pre):
                    self.assertIsNotNone(slot.rows)
                    self.assertLessEqual(slot.rows, 2541)
                    self.assertLessEqual(command.offset - slot.row_origin, 2541)
        self.assertTrue(saw_detach)

    def test_real_copy_fail_closed_but_fake_adapter_is_explicit(self):
        lm = FakeLanguageModel()
        app = DeferredPrefillAppend.create(lm, full_ready_cache(0), list(range(16385)), committed_frontier=0, mx=RecordingMx(fail_copy=True))
        with self.assertRaises(Exception):
            app.execute_all()
        self.assertEqual(app.state.value, 'failed')

        ok = fake_p6_app(lm, p6_ready_cache(lm, 0, shape_frontier=16384), list(range(16384)), committed_frontier=0)
        ok.execute_all()
        self.assertTrue(ok.final_execution.arena.p6_final_cone_detached)

    def test_final_decoder_cone_is_bounded_and_not_full_parent_view(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        app = fake_p6_app(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        app.execute_segment(app.plan.segments[0])
        app.execute_segment(app.plan.segments[1])
        arena = app.final_execution.arena
        self.assertTrue(arena.p6_final_cone_detached)
        self.assertEqual(arena.p6_final_cone_rows, 2541)
        self.assertEqual(arena.p6_final_cone_origin, 22035)
        self.assertIsNone(arena.encoder_final_h)
        self.assertIsNone(arena.encoder_final_pre)
        q_by_layer = {layer: sum(c.rows for c in app.plan.segments[1].commands if c.layer == layer and c.kind.value == 'encode_rows') for layer in range(20, 40)}
        self.assertEqual(q_by_layer[20], 2414)
        self.assertEqual(q_by_layer[39], 1)

    def test_final_seal_all40_only_after_readiness_and_stale_admission_fenced(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        old = LivePrefillContinuation.from_cache(cache)
        app = fake_p6_app(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        app.execute_segment(app.plan.segments[0])
        with self.assertRaises(Exception):
            _ = old.frontier
        self.assertFalse(app.readiness())
        self.assertTrue(all(c[0] == 0 for c in cache))
        with self.assertRaises(P6AppendError):
            app.final_seal()
        self.assertEqual(app.state.value, 'failed')
        contract_lm = ContractFrontierLanguageModel()
        fresh = fake_p6_app(contract_lm, p6_ready_cache(contract_lm, 0, shape_frontier=24577), list(range(24577)), committed_frontier=0)
        fresh.execute_all()
        for c in fresh.live_cache:
            self.assertEqual(offset_value(c[0]), 24577)
            self.assertEqual(tuple(c[0].shape), (1,))
            self.assertIn(str(c[0].dtype), ('int32', 'mlx.core.int32'))
        self.assertEqual(fresh.live_setup.arena.plan.count, 1)
        self.assertEqual(fresh.commit_certificate.T, 24577)
        self.assertEqual(tuple(fresh.live_setup.arena.tokens), tuple(range(24576, 24577)))

    def test_private_engram_history_feeds_second_segment_and_not_public_slot6(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        cache[0][6] = 'history_C'
        app = fake_p6_app(lm, cache, list(range(49155)), committed_frontier=0)
        app.begin()
        app.execute_segment(app.plan.segments[0])
        cache[0][6] = 'polluted_public_history_C'
        app.execute_segment(app.plan.segments[1])
        self.assertEqual(lm.hasher_calls[0], 'history_C')
        self.assertEqual(lm.hasher_calls[1], 'history_after_history_C')
        self.assertEqual(app.private_history_at_E, 'history_after_history_after_history_C')

    def test_stale_runners_revoked_and_historical_records_lightweight(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        shell = _sweep_shell(1, (), encoder_only=False)
        pre_arena = RequestArena.from_plan(shell, token_ids=[0])
        pre_runner = OfficialFP8MLXBlockRunner(lm, PublicationManager(pre_arena), working_cache=cache)
        app = fake_p6_app(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        with self.assertRaises(Exception):
            pre_runner.execute_command(app.plan.segments[0].commands[0], pre_arena)
        first = app.execute_segment(app.plan.segments[0])
        self.assertTrue(first.runner.closed)
        with self.assertRaises(Exception):
            first.runner.execute_command(app.plan.segments[0].commands[0], first.arena)
        self.assertEqual(len(app.segment_records), 1)
        self.assertFalse(hasattr(app.segment_records[0], 'runner'))
        self.assertFalse(hasattr(app.segment_records[0], 'arena'))
        self.assertIsNone(first.arena.carry.current.value)
        self.assertIsNone(first.arena.carry.pre.value)
        self.assertTrue(first.arena.p6_source_materialized)

    def test_p5_accepts_real_p6_commit_without_fake_sweep_metadata(self):
        lm = FakeLanguageModel()
        app = fake_p6_app(lm, p6_ready_cache(lm, 0, shape_frontier=16385), list(range(16385)), committed_frontier=0)
        app.execute_all()
        normalize_p6_fake_cache(lm, app.live_cache, 16385)
        self.assertEqual(app.live_setup.arena.plan.count, 1)
        result = LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=list(range(16385)))
        self.assertEqual(result.frontier, 16385)
        self.assertIs(result.live_cache, app.live_cache)
        self.assertEqual(validate_committed_cache(result._setup, list(range(16385))), 16385)
        with self.assertRaises(Exception):
            app.final_execution.runner.execute_command(app.plan.segments[-1].commands[0], app.final_execution.arena)

    def test_p6_commit_uses_full_p5_cache_structure_validation(self):
        lm = FakeLanguageModel()
        def committed():
            app = fake_p6_app(lm, p6_ready_cache(lm, 0, shape_frontier=16385), list(range(16385)), committed_frontier=0)
            app.execute_all()
            normalize_p6_fake_cache(lm, app.live_cache, 16385)
            return app
        app = committed()
        self.assertEqual(validate_committed_cache(app.commit_certificate, list(range(16385))), 16385)
        cases = [
            ('slot1', lambda c: c[0].__setitem__(1, ArrayMetadata((1, 127, lm._config.head_dim + lm._config.head_dim // 32)))),
            ('source_kv', lambda c: c[20].__setitem__(2, ArrayMetadata((1, 0, lm._config.head_dim // 2 + lm._config.head_dim // 16)))),
            ('index_k', lambda c: c[20].__setitem__(3, ArrayMetadata((1, 0, lm._config.index_head_dim // 2 + lm._config.index_head_dim // 32)))),
            ('pending', lambda c: c[2].__setitem__(4, ArrayMetadata((1, 99, lm._config.head_dim), 'bfloat16'))),
            ('engram', lambda c: c[0].__setitem__(6, ArrayMetadata((1, 1), 'int64'))),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                bad = committed()
                mutate(bad.live_cache)
                with self.assertRaises(Exception):
                    validate_committed_cache(bad.commit_certificate, list(range(16385)))

    def test_generation_p6_fence_precedes_frontier_mismatch_in_source(self):
        src = (ROOT / 'ds41f_mlx/runtime/omlx_generation.py').read_text()
        fn = src[src.index('    def from_prefilled_cache'):src.index('    @classmethod', src.index('    def from_prefilled_cache') + 1)]
        self.assertLess(fn.index('P6 cache is not sealed/admissible for generation'), fn.index('np.asarray'))

    def test_p6_smoke_zero_decoder_uses_runner_records_not_model_calls(self):
        text = (ROOT / 'tools/run_prefill_fp8_mlx_p6_smoke.py').read_text()
        self.assertIn('first.runner.records', text)
        self.assertNotIn('lm.layers[20].calls', text)

    def test_failed_final_seal_marks_cache_inadmissible(self):
        class BadCache(P6Cache):
            def __setitem__(self, key, value):
                if key == 0 and getattr(self, 'fail_slot0_install', False):
                    raise RuntimeError('slot0 install failure')
                return super().__setitem__(key, value)
        lm = FakeLanguageModel()
        cache = p6_ready_cache(lm, 0, shape_frontier=16385)
        bad = BadCache()
        for i in range(7):
            bad[i] = cache[3][i]
        bad.compress_ratio = cache[3].compress_ratio
        bad.fail_slot0_install = True
        cache[3] = bad
        app = fake_p6_app(lm, cache, list(range(16385)), committed_frontier=0)
        with self.assertRaises(Exception):
            app.execute_all()
        self.assertEqual(app.state.value, 'failed')
        self.assertTrue(any(getattr(c, '_p6_append_failed', False) for c in cache))
        with self.assertRaises(Exception):
            LivePrefillContinuation.from_cache(cache)

    def test_failed_append_rebuilds_with_fresh_cache_not_rewind(self):
        lm = ContractFrontierLanguageModel()
        cache = full_ready_cache(0)
        app = fake_p6_app(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        app.execute_segment(app.plan.segments[0])
        with self.assertRaises(P6AppendError):
            app.execute_segment(app.plan.segments[0])
        self.assertEqual(app.state.value, 'failed')
        rebuilt = app.rebuild_with_tokens(list(range(129)))
        self.assertIsNot(rebuilt.live_cache, cache)
        self.assertEqual((rebuilt.C, rebuilt.E, rebuilt.D, rebuilt.T), (0, 0, 0, 129))
        for c in rebuilt.live_cache:
            self.assertEqual(offset_value(c[0]), 0)
            self.assertEqual(tuple(c[0].shape), (1,))
            self.assertIn(str(c[0].dtype), ('int32', 'mlx.core.int32'))


class ContractFrontierLanguageModel(FakeLanguageModel):
    def cache_offset(self, value):
        return ArrayMetadata((1,), 'int32', int(value))


class ArrayMetadata:
    def __init__(self, shape, dtype='uint8', value=None):
        self.shape, self.dtype, self.value = shape, dtype, value
        self.allow_fake_eval = True
    def item(self):
        return self.value


def offset_value(offset):
    if hasattr(offset, 'item'):
        return int(offset.item())
    if hasattr(offset, 'value'):
        return int(offset.value)
    return int(offset)


class P6Cache(FakeCache):
    def size(self):
        return offset_value(self[0])


def normalize_p6_fake_cache(lm, cache, frontier: int):
    fresh = p6_ready_cache(lm, frontier, shape_frontier=frontier)
    for dst, src in zip(cache, fresh):
        flags = {name: getattr(dst, name) for name in ('_p6_append_invalid', '_p6_append_pending', '_p6_append_failed', '_p6_append_sealed', '_p6_owner_token') if hasattr(dst, name)}
        ratio = getattr(dst, 'compress_ratio', getattr(src, 'compress_ratio', None))
        for i in range(7):
            dst[i] = src[i]
        dst.compress_ratio = ratio
        for name, value in flags.items():
            setattr(dst, name, value)


def p6_ready_cache(lm, frontier: int, *, shape_frontier: int | None = None):
    c = lm._config
    c.window_size = getattr(c, 'window_size', 128)
    c.head_dim = getattr(c, 'head_dim', 512)
    c.index_head_dim = getattr(c, 'index_head_dim', 128)
    c.engram_max_ngram_size = getattr(c, 'engram_max_ngram_size', 4)
    c.vocab_size = getattr(c, 'vocab_size', 129280)
    c.compress_ratios[20] = 1
    physical = frontier if shape_frontier is None else int(shape_frontier)
    cache = []
    for layer in range(40):
        item = P6Cache()
        ratio = c.compress_ratios[layer] if layer in c.kv_source_layers else 0
        item.compress_ratio = ratio
        item[0] = ArrayMetadata((1,), 'int32', frontier)
        item[1] = ArrayMetadata((1, min(physical, c.window_size), c.head_dim + c.head_dim // 32))
        item[2] = ArrayMetadata((1, physical // ratio if ratio else 0, c.head_dim // 2 + c.head_dim // 16))
        item[3] = ArrayMetadata((1, physical // ratio if (ratio and layer in c.index_source_layers) else 0, c.index_head_dim // 2 + c.index_head_dim // 32))
        pending = physical % ratio if ratio > 1 else 0
        item[4] = ArrayMetadata((1, pending, c.head_dim), 'bfloat16')
        item[5] = ArrayMetadata((1, pending, c.head_dim), 'bfloat16')
        item[6] = ArrayMetadata((1, c.engram_max_ngram_size - 1), 'int64') if layer == 0 else ArrayMetadata((1, 0), 'int64')
        cache.append(item)
    return cache


class RecordingMx:
    int64 = 'int64'
    float32 = 'float32'

    def __init__(self, *, fail_eval=False, fail_copy=False):
        self.fail_eval = fail_eval
        self.fail_copy = fail_copy
        self.eval_calls = []
        self.copy_calls = []
        self.repeat_calls = []

    def array(self, ids, dtype=None):
        return _FakeInput(ids)

    def repeat(self, value, repeats, axis):
        self.repeat_calls.append((value, repeats, axis))
        return FakeTensor('hc', rows=value.shape[1])

    def zeros_like(self, value):
        return FakeTensor('zeros', rows=value.shape[1])

    def arange(self, n):
        return _FakeArray(range(n))

    def broadcast_to(self, value, shape):
        return FakeTensor('pre', rows=shape[1] if len(shape) > 1 else 1)

    def copy(self, value):
        if self.fail_copy:
            raise RuntimeError('copy failed')
        rows = value.shape[1] if hasattr(value, 'shape') and len(value.shape) > 1 else 1
        out = FakeTensor('compact_copy', rows=rows)
        out.allow_fake_compact = False
        self.copy_calls.append(value)
        return out

    def eval(self, *values):
        if self.fail_eval:
            raise RuntimeError('eval failed')
        self.eval_calls.append(values)


class _FakeInput:
    def __init__(self, ids):
        self.ids = list(ids)
        self.shape = (1, len(self.ids))
    def __getitem__(self, item):
        if item is None:
            return self
        return self.ids[item]


class _FakeArray(list):
    def astype(self, dtype):
        return self
    def __eq__(self, other):
        return _FakeArray([x == other for x in self])


if __name__ == '__main__':
    unittest.main()
