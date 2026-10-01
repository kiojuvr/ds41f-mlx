from __future__ import annotations

import unittest

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
)
from ds41f_mlx.prefill_fp8_mlx.p6_append import _sweep_shell
from test_prefill_fp8_mlx_p1_p2 import FakeLanguageModel, full_ready_cache


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
        app = DeferredPrefillAppend.create(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        app.execute_segment(app.plan.segments[0])
        self.assertEqual((app.E, app.D), (16384, 0))
        self.assertEqual({c[0] for c in [(cache[i][0],) for i in range(40)]}, {0})
        self.assertEqual(len(lm.layers[20].full_source_publishes), 1)
        self.assertEqual(lm.layers[20].calls, [])
        for layer in range(21, 40):
            self.assertEqual(lm.layers[layer].calls, [])
        source_only_commands = app.plan.segments[0].commands
        self.assertFalse([c for c in source_only_commands if c.layer is not None and c.layer >= 20 and c.kind.value == 'encode_rows'])

    def test_final_decoder_cone_is_bounded_and_not_full_parent_view(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        app = DeferredPrefillAppend.create(lm, cache, list(range(24577)), committed_frontier=0)
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
        app = DeferredPrefillAppend.create(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        app.execute_segment(app.plan.segments[0])
        with self.assertRaises(Exception):
            _ = old.frontier
        self.assertFalse(app.readiness())
        self.assertTrue(all(c[0] == 0 for c in cache))
        with self.assertRaises(P6AppendError):
            app.final_seal()
        self.assertEqual(app.state.value, 'failed')
        fresh = DeferredPrefillAppend.create(lm, full_ready_cache(0), list(range(24577)), committed_frontier=0)
        fresh.execute_all()
        self.assertTrue(all(c[0] == 24577 for c in fresh.live_cache))
        self.assertEqual(fresh.live_setup.arena.plan.count, 1)
        self.assertEqual(fresh.commit_certificate.T, 24577)
        self.assertEqual(tuple(fresh.live_setup.arena.tokens), tuple(range(24576, 24577)))

    def test_private_engram_history_feeds_second_segment_and_not_public_slot6(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        cache[0][6] = 'history_C'
        app = DeferredPrefillAppend.create(lm, cache, list(range(49155)), committed_frontier=0)
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
        app = DeferredPrefillAppend.create(lm, cache, list(range(24577)), committed_frontier=0)
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
        app = DeferredPrefillAppend.create(lm, full_ready_cache(0), list(range(16385)), committed_frontier=0)
        app.execute_all()
        self.assertEqual(app.live_setup.arena.plan.count, 1)
        result = LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=list(range(16385)))
        self.assertEqual(result.frontier, 16385)
        self.assertIs(result.live_cache, app.live_cache)
        self.assertEqual(validate_committed_cache(result._setup, list(range(16385))), 16385)

    def test_failed_final_seal_marks_cache_inadmissible(self):
        class BadFrontierCache(type(full_ready_cache(0)[0])):
            def __setitem__(self, item, value):
                if item == 0 and getattr(self, 'fail_slot0', False):
                    raise RuntimeError('slot0 install failure')
                return super().__setitem__(item, value)
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        bad = BadFrontierCache()
        for i in range(7):
            bad[i] = cache[3][i]
        bad.fail_slot0 = True
        cache[3] = bad
        app = DeferredPrefillAppend.create(lm, cache, list(range(16385)), committed_frontier=0)
        with self.assertRaises(Exception):
            app.execute_all()
        self.assertEqual(app.state.value, 'failed')
        self.assertTrue(any(getattr(c, '_p6_append_failed', False) for c in cache))
        with self.assertRaises(Exception):
            LivePrefillContinuation.from_cache(cache)

    def test_failed_append_rebuilds_with_fresh_cache_not_rewind(self):
        lm = FakeLanguageModel()
        cache = full_ready_cache(0)
        app = DeferredPrefillAppend.create(lm, cache, list(range(24577)), committed_frontier=0)
        app.begin()
        app.execute_segment(app.plan.segments[0])
        with self.assertRaises(P6AppendError):
            app.execute_segment(app.plan.segments[0])
        self.assertEqual(app.state.value, 'failed')
        rebuilt = app.rebuild_with_tokens(list(range(129)))
        self.assertIsNot(rebuilt.live_cache, cache)
        self.assertEqual((rebuilt.C, rebuilt.E, rebuilt.D, rebuilt.T), (0, 0, 0, 129))


if __name__ == '__main__':
    unittest.main()
