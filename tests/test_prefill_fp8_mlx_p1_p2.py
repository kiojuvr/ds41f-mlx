from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import (
    FORBIDDEN_HOT_PATH_MODULES,
    OfficialFP8MLXBlockRunner,
    PublicationError,
    PublicationManager,
    RequestArena,
    ResumeUnavailableError,
    SemanticBoundary,
    SweepCommandKind,
    SweepPhase,
    SweepPlan,
    SweepPlanner,
    SweepTopologyError,
    TensorOwnership,
    UnsupportedTopologyError,
    check_no_reference_hot_path,
)


class PrefillFP8MLXP1P2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        lib_path = compile_native_prefill_library(Path(cls._tmp.name))
        cls.native = load_native_prefill_library(lib_path)
        cls.planner = SweepPlanner(cls.native)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def assert_all_commands_known(self, plan):
        self.assertTrue(plan.commands)
        for command in plan.commands:
            self.assertIsNot(command.kind, SweepCommandKind.UNKNOWN)
            self.assertIsNot(command.phase, SweepPhase.UNKNOWN)

    def test_short_non_wide_sweep_is_strictly_known(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        self.assertFalse(plan.wide)
        self.assertFalse(plan.decoder_suffix)
        self.assert_all_commands_known(plan)
        self.assertEqual(len(plan.publication_commands), 40)
        begin_layers = {c.layer for c in plan.commands_by_kind(SweepCommandKind.BEGIN_LAYER)}
        self.assertEqual(begin_layers, set(range(40)))

    def test_wide_decoder_suffix_sweep_is_strictly_known(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        self.assertTrue(plan.wide)
        self.assertTrue(plan.decoder_suffix)
        self.assertTrue(plan.commands_by_kind(SweepCommandKind.DECODER_PREPARE_SUFFIX))
        suffix_encodes = [c for c in plan.encode_commands if c.phase is SweepPhase.DECODER_SUFFIX]
        self.assertTrue(suffix_encodes)
        self.assert_all_commands_known(plan)

    def test_encoder_only_plan_stops_invalid_after_encoder(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192, encoder_only=True)
        self.assertTrue(plan.encoder_only)
        self.assertFalse(plan.checkpoint_valid_after_sweep)
        self.assertEqual({c.layer for c in plan.commands_by_kind(SweepCommandKind.BEGIN_LAYER)}, set(range(20)))
        self.assertTrue(plan.commands_by_kind(SweepCommandKind.ENCODER_ONLY_COMPLETE_INVALID))
        self.assert_all_commands_known(plan)

    def test_resume_is_explicitly_unavailable_until_p6(self):
        with self.assertRaises(ResumeUnavailableError):
            self.planner.build_static(ctx=32768, remaining=8192, resume_encoder=True)

    def test_strict_plan_rejects_unknown_command_or_phase(self):
        base = self.planner.build_static(ctx=32768, remaining=2048).to_json()
        bad_command = dict(base)
        bad_commands = [dict(c) for c in base["commands"]]
        bad_commands[0]["kind"] = "new_native_command"
        bad_commands[0]["raw_kind"] = 999
        bad_command["commands"] = bad_commands
        with self.assertRaises(SweepTopologyError):
            SweepPlan.from_native_json(bad_command, strict=True)
        diagnostic = SweepPlan.from_native_json(bad_command, strict=False)
        self.assertIs(diagnostic.commands[0].kind, SweepCommandKind.UNKNOWN)

        bad_phase = dict(base)
        phase_commands = [dict(c) for c in base["commands"]]
        phase_commands[0]["phase"] = "new_phase"
        phase_commands[0]["raw_phase"] = 99
        bad_phase["commands"] = phase_commands
        with self.assertRaises(SweepTopologyError):
            SweepPlan.from_native_json(bad_phase, strict=True)

    def test_v41_topology_mismatch_fails_closed(self):
        cfg = SimpleNamespace(
            n_layers=39,
            hc_mult=4,
            engram_layer_ids=(1, 14),
            kv_source_layers=(),
            index_source_layers=(),
            compress_ratios={},
            vocab_size=129280,
            dim=5120,
        )
        lm = SimpleNamespace(_config=cfg, layers=[object()] * 39, make_cache=lambda: [])
        with self.assertRaises(UnsupportedTopologyError):
            self.planner.build_from_model(lm, ctx=32768, remaining=2048)

    def test_missing_or_wrong_engram_topology_fails_closed(self):
        for engram_layers in ((), (1,), (1, 13)):
            cfg = SimpleNamespace(
                n_layers=40,
                hc_mult=4,
                engram_layer_ids=engram_layers,
                kv_source_layers=(),
                index_source_layers=(),
                compress_ratios={},
                vocab_size=129280,
                dim=5120,
            )
            lm = SimpleNamespace(_config=cfg, layers=[object()] * 40, make_cache=lambda: [])
            with self.assertRaises(UnsupportedTopologyError):
                self.planner.build_from_model(lm, ctx=32768, remaining=2048)

    def test_malformed_per_layer_command_topology_fails_closed(self):
        base = self.planner.build_static(ctx=32768, remaining=2048).to_json()
        malformed = dict(base)
        malformed["commands"] = [c for c in base["commands"] if not (c["layer"] == 0 and c["kind"] == "publish_state_frontier")]
        with self.assertRaises(SweepTopologyError):
            SweepPlan.from_native_json(malformed, strict=True)

    def test_semantic_boundaries_do_not_imply_mlx_materialization(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        boundaries = plan.semantic_boundaries
        self.assertTrue(boundaries)
        self.assertIn(SemanticBoundary.TRANSACTION_BEGIN_INVALID, {c.semantic_boundary for c in boundaries})
        self.assertTrue(all(not c.requires_mlx_materialization for c in boundaries))

    def test_arena_ownership_aliases_and_transaction_follow_plan(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], input_ids="ids", h_current="h0", h_next="h1", pre="pre")
        self.assertIn("batch_cur_hc", arena.allocations)
        self.assertEqual(arena.carry.current.alias_reuse_class, "hc_pingpong")
        self.assertEqual(arena.carry.next.alias_reuse_class, "hc_pingpong")
        self.assertEqual(arena.carry.current.ownership, TensorOwnership.OWNED)
        begin = plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0]
        arena.apply_command(begin)
        self.assertFalse(arena.transaction.valid)
        self.assertFalse(arena.cache_materialized)
        encode = plan.encode_commands[0]
        arena.apply_command(encode)
        self.assertEqual(arena.active_chunk_views[-1].base_role, arena.carry.current.role)
        self.assertEqual(arena.active_chunk_views[-1].rows, encode.rows)
        swap = plan.commands_by_kind(SweepCommandKind.SWAP_HC_AFTER_LAYER)[0]
        old_current = arena.carry.current.role
        arena.apply_command(swap)
        self.assertNotEqual(arena.carry.current.role, old_current)
        publish = plan.publication_commands[0]
        arena.apply_command(publish)
        self.assertEqual(len(arena.publications.pending_events), 1)
        self.assertFalse(arena.prefill_continuation_exported)

    def test_arena_records_suffix_views_without_full_cache_materialization(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3])
        for command in plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE) + plan.commands_by_kind(SweepCommandKind.DECODER_PREPARE_SUFFIX)[:2]:
            arena.apply_command(command)
        self.assertTrue(arena.suffix_views)
        self.assertGreater(arena.retained_suffix_rows(), 0)
        self.assertFalse(arena.cache_materialized)

    def test_allocation_lifetime_and_transient_views_retire(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current="h0", h_next="h1", pre="pre")
        begin = plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0]
        encode = plan.encode_commands[0]
        swap = plan.commands_by_kind(SweepCommandKind.SWAP_HC_AFTER_LAYER)[0]
        arena.apply_command(begin)
        arena.apply_command(encode)
        self.assertTrue(arena.active_allocation_roles())
        self.assertEqual(len(arena.active_chunk_views), 1)
        arena.apply_command(swap)
        self.assertEqual(len(arena.active_chunk_views), 0)
        self.assertEqual(arena.retired_chunk_view_count, 1)

    def test_publication_visibility_and_failure_are_transaction_local(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3])
        manager = PublicationManager(arena)
        manager.begin_transaction()
        manager.capture_layer_outputs(2, {"kv": "kv", "index_k": "ik", "idx": "idx"}, command_index=10)
        with self.assertRaises(PublicationError):
            manager.consume("kv", layer=3)
        publish = [c for c in plan.publication_commands if c.layer == 2][0]
        arena.apply_command(publish)
        manager.publish_frontier(publish)
        self.assertEqual(manager.consume("kv", layer=3), "kv")
        manager.fail()
        with self.assertRaises(PublicationError):
            manager.consume("kv", layer=3)
        self.assertEqual(manager.committed_visible, {})

    def test_block_runner_is_command_driven_and_receives_offsets(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h"), h_next=FakeTensor("n"), pre=FakeTensor("p"))
        manager = PublicationManager(arena)
        lm = FakeLanguageModel()
        runner = OfficialFP8MLXBlockRunner(lm, manager)
        begin = plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0]
        encode = [c for c in plan.encode_commands if c.layer == 2][0]
        runner.execute_command(begin, arena)
        runner.execute_command(encode, arena)
        self.assertEqual(lm.layers[2].calls, [(encode.offset, encode.rows, encode.offset)])
        self.assertEqual(len(runner.records), 2)
        self.assertFalse(runner.prefill_continuation_exported)
        self.assertEqual(runner.full_cache_repack_count, 0)

    def test_connected_publication_and_serving_logits_policy_preserve_commit(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h"), h_next=FakeTensor("n"), pre=FakeTensor("p"))
        manager = PublicationManager(arena)
        runner = OfficialFP8MLXBlockRunner(FakeLanguageModel(), manager)
        commands = [plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0]]
        commands.append([c for c in plan.encode_commands if c.layer == 2][0])
        commands.append([c for c in plan.publication_commands if c.layer == 2][0])
        commands.extend(plan.commands_by_kind(SweepCommandKind.ENCODE_OUTPUT_HEAD))
        commands.extend(plan.commands_by_kind(SweepCommandKind.READ_LOGITS))
        commands.extend(plan.commands_by_kind(SweepCommandKind.CHECKPOINT_MAY_COMMIT))
        runner.execute_batch(commands, arena)
        self.assertTrue(runner.final_logits_suppressed)
        self.assertTrue(arena.transaction.valid)
        self.assertIn("kv", manager.committed_visible)
        self.assertTrue(any(r.skipped_final_logits for r in runner.records))

    def test_import_does_not_load_forbidden_reference_modules(self):
        loaded = [name for name in FORBIDDEN_HOT_PATH_MODULES if name in sys.modules]
        self.assertEqual(loaded, [])
        self.assertTrue(check_no_reference_hot_path().ok)


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.last_slice = None

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], slice):
            sl = item[1]
            start = 0 if sl.start is None else int(sl.start)
            stop = start if sl.stop is None else int(sl.stop)
            out = FakeTensor(f"{self.name}[{start}:{stop}]")
            out.last_slice = (start, stop - start)
            return out
        return self

    def __setitem__(self, item, value):
        return None


class FakeLayer:
    def __init__(self):
        self.calls = []

    def __call__(self, h, pre, cache, shared, start, image_mask):
        rows = 0
        if getattr(h, "last_slice", None) is not None:
            _offset, rows = h.last_slice
        self.calls.append((start, rows, start))
        shared["kv"] = "kv"
        shared["index_k"] = "ik"
        shared["idx"] = "idx"
        shared["candidates"] = "cand"
        return h, pre


class FakeLanguageModel:
    def __init__(self):
        self.layers = [FakeLayer() for _ in range(40)]
        self._config = SimpleNamespace(engram_layer_ids=(1, 14))

    def make_cache(self):
        return [{} for _ in range(40)]


if __name__ == "__main__":
    unittest.main()
