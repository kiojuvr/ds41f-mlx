from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import (
    FORBIDDEN_HOT_PATH_MODULES,
    DwarfStarFP8MLXPrefillExecutorSetup,
    LivePrefillContinuation,
    OfficialFP8MLXBlockRunner,
    PublicationError,
    PublicationManager,
    PublicationTopology,
    PrefillSetupError,
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
        lm = FakeLanguageModel()
        runner = OfficialFP8MLXBlockRunner(lm, manager)
        runner.working_cache = full_ready_cache(frontier=plan.count)
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

    def test_two_chunk_row_span_idx_and_candidates_publication(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3])
        manager = PublicationManager(arena)
        manager.begin_transaction()
        manager.capture_layer_outputs(20, {"kv": "kv0", "index_k": "ik0", "idx": FakeRowTensor("idx0", 0, 4096), "candidates": FakeRowTensor("cand0", 0, 4096)}, command_index=1, offset=0, rows=4096)
        manager.capture_layer_outputs(20, {"kv": "kv1", "index_k": "ik1", "idx": FakeRowTensor("idx1", 4096, 4096), "candidates": FakeRowTensor("cand1", 4096, 4096)}, command_index=2, offset=4096, rows=4096)
        publish = [c for c in plan.publication_commands if c.layer == 20][0]
        arena.apply_command(publish)
        manager.publish_frontier(publish)
        self.assertEqual(manager.consume_span("idx", layer=24, offset=0, rows=4096).name, "idx0")
        self.assertEqual(manager.consume_span("candidates", layer=24, offset=4096, rows=4096).name, "cand1")
        both = manager.consume_span("idx", layer=24, offset=0, rows=8192)
        self.assertEqual([x.name for x in both.parts], ["idx0", "idx1"])
        self.assertEqual(manager.consume("kv", layer=24), "kv1")
        with self.assertRaises(Exception):
            manager.consume_span("candidates", layer=24, offset=8192, rows=1)

    def test_block_runner_preserves_two_chunk_row_span_publication(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h", rows=8192), h_next=FakeTensor("n", rows=8192), pre=FakeTensor("p", rows=8192))
        manager = PublicationManager(arena)
        runner = OfficialFP8MLXBlockRunner(FakeLanguageModel(), manager)
        runner.execute_command(plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0], arena)
        for command in [c for c in plan.encode_commands if c.layer == 2][:2]:
            runner.execute_command(command, arena)
        publish = [c for c in plan.publication_commands if c.layer == 2][0]
        arena.apply_command(publish)
        manager.publish_frontier(publish)
        self.assertEqual(manager.consume_span("idx", layer=3, offset=0, rows=4096).name, "idx@0")
        self.assertEqual(manager.consume_span("idx", layer=3, offset=4096, rows=4096).name, "idx@4096")
        self.assertEqual(manager.consume("kv", layer=3), "kv@4096")

    def test_base_frontier_absolute_start_reaches_layer(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h", rows=8192), h_next=FakeTensor("n", rows=8192), pre=FakeTensor("p", rows=8192), base_frontier=100)
        manager = PublicationManager(arena)
        lm = FakeLanguageModel()
        runner = OfficialFP8MLXBlockRunner(lm, manager)
        runner.execute_command(plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0], arena)
        encodes = [c for c in plan.encode_commands if c.layer == 0]
        runner.execute_command(encodes[0], arena)
        runner.execute_command(encodes[1], arena)
        self.assertEqual(lm.layers[0].calls[0][0], 100)
        self.assertEqual(lm.layers[0].calls[1][0], 4196)

    def test_decoder_prepare_has_effect_and_suffix_cone_is_exact(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h", rows=8192), h_next=FakeTensor("n", rows=8192), pre=FakeTensor("p", rows=8192))
        manager = PublicationManager(arena)
        lm = FakeLanguageModel()
        runner = OfficialFP8MLXBlockRunner(lm, manager)
        runner.execute_command(plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0], arena)
        arena.encoder_final_h = "full_encoder_final"
        arena.encoder_final_pre = "encoder_final_pre"
        prepares20 = [c for c in plan.commands_by_kind(SweepCommandKind.DECODER_PREPARE_SUFFIX) if c.layer == 20]
        runner.execute_command(prepares20[0], arena)
        self.assertEqual(len(lm.layers[20].full_source_publishes), 1)
        runner.execute_command(prepares20[1], arena)
        self.assertIn(20, arena.decoder_prepared_by_layer)
        self.assertTrue(lm.layers[20].prepared_windows)
        enc20s = [c for c in plan.encode_commands if c.layer == 20]
        self.assertEqual(sum(c.rows for c in enc20s), 1 + (39 - 20) * 127)
        for enc20 in enc20s:
            runner.execute_command(enc20, arena)
        self.assertEqual(len(lm.layers[20].full_source_publishes), 1)
        self.assertEqual(lm.layers[20].full_source_publishes[-1][0], "full_encoder_final")
        self.assertEqual(lm.layers[20].full_source_publishes[-1][1], "encoder_final_pre")
        self.assertEqual(lm.layers[20].shared_seen[-1].get("kv"), "full_source_kv")
        for layer in range(21, 40):
            encs = [c for c in plan.encode_commands if c.layer == layer]
            self.assertEqual(sum(c.rows for c in encs), 1 + (39 - layer) * 127)

    def test_suffix_encode_without_prepare_fails(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h", rows=8192), h_next=FakeTensor("n", rows=8192), pre=FakeTensor("p", rows=8192))
        runner = OfficialFP8MLXBlockRunner(FakeLanguageModel(), PublicationManager(arena))
        runner.execute_command(plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0], arena)
        with self.assertRaises(Exception):
            runner.execute_command([c for c in plan.encode_commands if c.layer == 20][0], arena)

    def test_working_cache_frontier_and_engram_history_progress(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h", rows=2048), h_next=FakeTensor("n", rows=2048), pre=FakeTensor("p", rows=2048), engram_history="history", base_frontier=7)
        manager = PublicationManager(arena)
        runner = OfficialFP8MLXBlockRunner(FakeLanguageModel(), manager)
        runner.execute_command(plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0], arena)
        encode = [c for c in plan.encode_commands if c.layer == 0][0]
        runner.execute_command(encode, arena)
        cache0 = runner.working_cache[0]
        self.assertEqual(cache0[0], 7 + encode.rows)
        self.assertEqual(cache0[6], "history")
        self.assertTrue(all(cache0[i] is not None for i in range(7)))

    def test_carry_writes_preserve_multiple_chunks_and_fail_closed(self):
        plan = self.planner.build_static(ctx=32768, remaining=8192)
        arena = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h", rows=8192), h_next=FakeTensor("n", rows=8192), pre=FakeTensor("p", rows=8192))
        manager = PublicationManager(arena)
        runner = OfficialFP8MLXBlockRunner(FakeLanguageModel(), manager)
        runner.execute_command(plan.commands_by_kind(SweepCommandKind.BEGIN_INVALIDATE)[0], arena)
        for command in [c for c in plan.encode_commands if c.layer == 0][:2]:
            runner.execute_command(command, arena)
        self.assertEqual(arena.carry.next.value.shape[1], 8192)
        self.assertEqual([w[0] for w in arena.carry.next.value.writes], [0, 4096])
        bad = RequestArena.from_plan(plan, token_ids=[0, 3], h_current=FakeTensor("h", rows=1), h_next=FakeTensor("n", rows=1), pre=FakeTensor("p", rows=1))
        with self.assertRaises(Exception):
            runner.execute_command([c for c in plan.encode_commands if c.layer == 0][0], bad)

    def test_executor_setup_rejects_bad_frontier_and_token_count(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        with self.assertRaises(PrefillSetupError):
            DwarfStarFP8MLXPrefillExecutorSetup(FakeLanguageModel(), mx=FakeMx()).prepare(plan, [0, 3], base_frontier=55)
        with self.assertRaises(PrefillSetupError):
            DwarfStarFP8MLXPrefillExecutorSetup(FakeLanguageModel(), mx=FakeMx()).prepare(plan, [0, 3])

    def test_live_continuation_derives_frontier_reuses_cache_and_hash_history(self):
        plan = self.planner.build_static(ctx=32768, remaining=2048)
        lm = FakeLanguageModel()
        live = full_ready_cache(frontier=77)
        live[0][6] = "prior_history"
        setup = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=FakeMx()).prepare(plan, list(range(2048)), continuation=live)
        self.assertEqual(setup.arena.base_frontier, 77)
        self.assertIs(setup.block_runner.working_cache, live)
        self.assertEqual(lm.hasher_calls[-1], "prior_history")
        self.assertEqual(setup.block_runner.working_cache[0][6], "history_after_prior_history")
        with self.assertRaises(PrefillSetupError):
            DwarfStarFP8MLXPrefillExecutorSetup(FakeLanguageModel(), mx=FakeMx()).prepare(plan, list(range(2048)), continuation=live, base_frontier=78)
        cont = LivePrefillContinuation.from_cache(live)
        for cache in live:
            cache[0] = 99
        setup2 = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=FakeMx()).prepare(plan, list(range(2048)), continuation=cont)
        self.assertEqual(setup2.arena.base_frontier, 99)
        for cache in live:
            cache[0] = 123
        setup3 = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=FakeMx()).prepare(plan, list(range(2048)), continuation=cont)
        self.assertEqual(setup3.arena.base_frontier, 123)
        divergent = full_ready_cache(frontier=77)
        divergent[3][0] = 76
        with self.assertRaises(PrefillSetupError):
            LivePrefillContinuation.from_cache(divergent)

    def test_model_derived_publication_topology_and_mismatch(self):
        topo = PublicationTopology.from_model_config(FakeLanguageModel()._config)
        self.assertEqual(topo.publishes_by_layer[20], ("kv", "index_k", "idx", "candidates"))
        bad = SimpleNamespace(engram_layer_ids=(1, 14), kv_source_layers=(2,), index_source_layers=(), candidate_source_layer=-1, compress_ratios={})
        with self.assertRaises(PublicationError):
            PublicationTopology.from_model_config(bad)

    def test_no_independent_whole_model_layer_loop_or_intermediate_export(self):
        import inspect
        import ds41f_mlx.prefill_fp8_mlx.block_runner as br
        source = inspect.getsource(br.OfficialFP8MLXBlockRunner)
        self.assertNotIn("for layer in self.language_model.layers", source)
        self.assertNotIn("for layer in lm.layers", source)
        runner = OfficialFP8MLXBlockRunner(FakeLanguageModel(), PublicationManager(RequestArena.from_plan(self.planner.build_static(ctx=32768, remaining=2048), token_ids=[0, 3])))
        self.assertFalse(runner.prefill_continuation_exported)
        self.assertEqual(runner.full_cache_repack_count, 0)

    def test_import_does_not_load_forbidden_reference_modules(self):
        loaded = [name for name in FORBIDDEN_HOT_PATH_MODULES if name in sys.modules]
        self.assertEqual(loaded, [])
        self.assertTrue(check_no_reference_hot_path().ok)


class FakeMx:
    int64 = "int64"
    float32 = "float32"

    def array(self, ids, dtype=None):
        return FakeInput(ids)

    def repeat(self, value, repeats, axis):
        rows = value.shape[1] if hasattr(value, "shape") else len(value[0])
        return FakeTensor("hc", rows=rows)

    def zeros_like(self, value):
        return FakeTensor("zeros", rows=value.shape[1])

    def arange(self, n):
        return FakeArray(list(range(n)))

    def broadcast_to(self, value, shape):
        return FakeTensor("pre", rows=shape[1] if len(shape) > 1 else 1)


class FakeInput:
    def __init__(self, ids):
        self.ids = list(ids)
        self.shape = (1, len(self.ids))

    def __getitem__(self, item):
        if item is None:
            return self
        return self.ids[item]

    def __len__(self):
        return len(self.ids)


class FakeArray(list):
    def astype(self, dtype):
        return self

    def __eq__(self, other):
        return FakeArray([x == other for x in self])


class FakeTensor:
    def __init__(self, name, rows=2048):
        self.name = name
        self.shape = (1, rows, 4, 8)
        self.last_slice = None
        self.writes = []

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], slice):
            sl = item[1]
            start = 0 if sl.start is None else int(sl.start)
            stop = start if sl.stop is None else int(sl.stop)
            if stop > self.shape[1]:
                raise IndexError("slice out of range")
            out = FakeTensor(f"{self.name}[{start}:{stop}]", rows=stop - start)
            out.last_slice = (start, stop - start)
            return out
        return self

    def __setitem__(self, item, value):
        if isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], slice):
            sl = item[1]
            start = 0 if sl.start is None else int(sl.start)
            stop = start if sl.stop is None else int(sl.stop)
            if stop > self.shape[1]:
                raise IndexError("write out of range")
            self.writes.append((start, stop - start, value))
            return None
        raise IndexError("unsupported write")

    def assign_rows(self, offset, rows, update):
        if offset + rows > self.shape[1]:
            raise IndexError("write out of range")
        self.writes.append((offset, rows, update))


class FakeRowTensor:
    def __init__(self, name, offset, rows):
        self.name = name
        self.offset = offset
        self.rows = rows
        self.shape = (1, rows, 1)

    def __getitem__(self, key):
        span = key[1]
        start, stop, _ = span.indices(self.rows)
        return FakeRowTensor(self.name, self.offset + start, stop - start)

    def concat_rows(self, values):
        return values[0] if len(values) == 1 else FakeConcatTensor(values)


class FakeConcatTensor:
    def __init__(self, parts):
        self.parts = parts


class FakeCache:
    def __init__(self):
        self.cache = [None] * 7

    def __getitem__(self, item):
        return self.cache[item]

    def __setitem__(self, item, value):
        self.cache[item] = value


class FakeLayer:
    def __init__(self):
        self.calls = []
        self.shared_seen = []
        self.prepared_windows = []
        self.full_source_publishes = []
        self.attn = FakeAttention(self)

    def __call__(self, h, pre, cache, shared, start, image_mask):
        rows = 0
        if getattr(h, "last_slice", None) is not None:
            _offset, rows = h.last_slice
        self.calls.append((start, rows, start))
        self.shared_seen.append(dict(shared))
        if shared.get("kv") is None:
            shared["kv"] = f"kv@{start}"
        if shared.get("index_k") is None:
            shared["index_k"] = f"ik@{start}"
        shared["idx"] = FakeRowTensor(f"idx@{start}", start, rows)
        shared["candidates"] = FakeRowTensor(f"cand@{start}", start, rows)
        return h, pre


    # Explicit test-only math hook; production must never fall back to __call__.
    def execute_suffix_query_math(self, h, pre, cache, shared, start, image_mask):
        return self(h, pre, cache, shared, start, image_mask)


class FakeAttention:
    def __init__(self, parent):
        self.parent = parent
        self.query_full_source_calls = 0

    def prepare_decoder_local_window_math(self, rows, pre, cache, absolute_start, count):
        self.parent.prepared_windows.append((absolute_start, count, rows))
        cache[1] = f"prepared_window@{absolute_start}:{count}"

    def publish_full_encoder_source_math(self, full_source, pre, cache, shared, absolute_start, count):
        self.parent.full_source_publishes.append((full_source, pre, absolute_start, count))
        shared["kv"] = "full_source_kv"
        shared["index_k"] = "full_source_index_k"


class FakeLanguageModel:
    def __init__(self):
        self.layers = [FakeLayer() for _ in range(40)]
        self._config = SimpleNamespace(
            engram_layer_ids=(1, 14),
            kv_source_layers=(2, 8, 14, 20),
            index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
            candidate_source_layer=20,
            compress_ratios={i: (4 if i >= 2 else 0) for i in range(40)},
            hc_mult=4,
        )
        self.hasher_calls = []
        self._hasher = self._hash

    def embed(self, input_ids):
        return FakeTensor("emb", rows=input_ids.shape[1] if hasattr(input_ids, "shape") else 8192)

    def _hash(self, input_ids, prior_history, image_mask):
        self.hasher_calls.append(prior_history)
        return FakeTensor("hash", rows=8192), f"history_after_{prior_history}"

    def make_cache(self):
        return [FakeCache() for _ in range(40)]

    def empty_cache_slot(self, slot):
        return []

    def cache_offset(self, value):
        return int(value)


def full_ready_cache(frontier=0):
    caches = [FakeCache() for _ in range(40)]
    for cache in caches:
        for i in range(7):
            cache[i] = frontier if i == 0 else []
    return caches


if __name__ == "__main__":
    unittest.main()
