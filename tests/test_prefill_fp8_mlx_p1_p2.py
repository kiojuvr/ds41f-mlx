from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import (
    FORBIDDEN_HOT_PATH_MODULES,
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
        self.assertEqual(max(c.layer for c in plan.commands if c.layer is not None and c.kind is SweepCommandKind.BEGIN_LAYER), 19)
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

    def test_import_does_not_load_forbidden_reference_modules(self):
        loaded = [name for name in FORBIDDEN_HOT_PATH_MODULES if name in sys.modules]
        self.assertEqual(loaded, [])
        self.assertTrue(check_no_reference_hot_path().ok)


if __name__ == "__main__":
    unittest.main()
