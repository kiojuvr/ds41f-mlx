from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import json
import unittest
from unittest.mock import patch

from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import (
    BlockExecutionError, LiveCacheHandoffError, LivePrefillResult,
    OfficialFP8MLXBlockRunner, PrefillExecutionSetup, PublicationManager,
    RequestArena, SweepPlanner, handoff_to_generation,
)
from test_prefill_fp8_mlx_p1_p2 import FakeCache, FakeLanguageModel


class ArrayMetadata:
    def __init__(self, shape, dtype='uint8', value=None):
        self.shape, self.dtype, self.value = shape, dtype, value
    def item(self): return self.value


class Cache(FakeCache):
    def size(self): return self[0].value


class RecordingGenerationSession:
    @classmethod
    def from_prefilled_cache(cls, model, cache, ids, config, **kw):
        obj = cls()
        obj.initial_cache = cache
        obj.admitted_cache = cache
        obj.prefix_tokens = list(ids)
        obj.prompt_replay_count = 0
        obj.forwarded = []
        obj.started = False
        return obj

    def start(self, terminal):
        if self.started: raise RuntimeError('start already attempted')
        self.started = True
        self.inserted_prompt = [terminal]
        self.forwarded.append(terminal)
        for c in self.admitted_cache: c[0].value += 1
        self.scheduler_cache = self.admitted_cache
        self.initial_cache = []

    def active_cache_offsets(self): return tuple(c.size() for c in self.scheduler_cache)
    def close(self): pass


class P5HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.planner = SweepPlanner(load_native_prefill_library(compile_native_prefill_library(Path(cls.tmp.name))))
        cls.plan = cls.planner.build_static(ctx=32768, remaining=2048)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def setup_ready(self, frontier=2048):
        lm = FakeLanguageModel()
        c = lm._config
        c.window_size, c.head_dim, c.index_head_dim = 128, 512, 128
        c.engram_max_ngram_size, c.vocab_size = 4, 129280
        c.compress_ratios[20] = 1
        arena = RequestArena.from_plan(self.plan, token_ids=[1]*2048, base_frontier=frontier-2048)
        manager = PublicationManager(arena)
        manager.begin_transaction()
        arena.transaction.begin_invalid()
        arena.transaction.commit()
        manager.commit()
        runner = OfficialFP8MLXBlockRunner(lm, manager)
        runner.final_logits_suppressed = True
        cache = []
        for layer in range(40):
            item = Cache()
            ratio = c.compress_ratios[layer] if layer in c.kv_source_layers else 0
            item.compress_ratio = ratio
            item[0] = ArrayMetadata((1,), 'int32', frontier)
            item[1] = ArrayMetadata((1, 128, 528))
            item[2] = ArrayMetadata((1, frontier//ratio if ratio else 0, 288))
            item[3] = ArrayMetadata((1, frontier//ratio if ratio else 0, 68))
            item[4] = ArrayMetadata((1, frontier % ratio if ratio > 1 else 0, 512), 'bfloat16')
            item[5] = ArrayMetadata(item[4].shape, 'bfloat16')
            item[6] = ArrayMetadata((1, 3), 'int64') if layer == 0 else ArrayMetadata((1, 0), 'int64')
            cache.append(item)
        runner.working_cache = cache
        return PrefillExecutionSetup(arena, manager, runner), lm

    def result(self, setup, count=2048):
        return LivePrefillResult.from_committed(setup, prefix_token_ids=[1]*count)

    def test_prefix_length_mismatch_fails(self):
        setup, _ = self.setup_ready()
        with self.assertRaises(LiveCacheHandoffError): self.result(setup, 2049)
        self.assertFalse(setup.handoff_claimed)

    def test_divergent_frontier_fails(self):
        setup, _ = self.setup_ready()
        setup.block_runner.working_cache[39][0].value -= 1
        with self.assertRaises(LiveCacheHandoffError): self.result(setup)

    def test_uncommitted_or_failed_transactions_fail(self):
        for attr in ('begun', 'committed', 'valid', 'failed', 'publication_failed'):
            with self.subTest(attr=attr):
                setup, _ = self.setup_ready()
                if attr == 'publication_failed': setup.publication_manager.failed = True
                else: setattr(setup.arena.transaction, attr, attr == 'failed')
                with self.assertRaises(LiveCacheHandoffError): self.result(setup)

    def test_export_and_repack_fail(self):
        for owner, attr in [('runner', 'full_cache_repack_count'), ('runner', 'prefill_continuation_exported'), ('arena', 'prefill_continuation_exported')]:
            with self.subTest(attr=attr, owner=owner):
                setup, _ = self.setup_ready()
                setattr(setup.block_runner if owner == 'runner' else setup.arena, attr, 1)
                with self.assertRaises(LiveCacheHandoffError): self.result(setup)

    def test_missing_engram_history_fails(self):
        setup, _ = self.setup_ready()
        setup.block_runner.working_cache[0][6] = None
        with self.assertRaises(LiveCacheHandoffError): self.result(setup)

    def test_source_and_window_structure_fail_closed(self):
        for layer, slot, value in [(0, 1, ArrayMetadata((1, 127, 528))), (20, 2, ArrayMetadata((1, 0, 288))), (20, 3, None), (20, 4, ArrayMetadata((1, 1, 512), 'float32'))]:
            with self.subTest(layer=layer, slot=slot):
                setup, _ = self.setup_ready()
                setup.block_runner.working_cache[layer][slot] = value
                with self.assertRaises(LiveCacheHandoffError): self.result(setup)

    def test_prefix_logits_must_be_suppressed(self):
        setup, _ = self.setup_ready()
        setup.block_runner.final_logits_suppressed = False
        with self.assertRaises(LiveCacheHandoffError): self.result(setup)

    def test_complete_history_not_last_arena_tokens(self):
        setup, _ = self.setup_ready(frontier=4096)
        self.assertEqual(len(setup.arena.tokens), 2048)
        with self.assertRaises(LiveCacheHandoffError): self.result(setup, 2048)
        result = self.result(setup, 4096)
        self.assertEqual(result.frontier, 4096)

    def test_same_cache_terminal_once_zero_replay_and_revoked_authority(self):
        setup, lm = self.setup_ready()
        cache = setup.block_runner.working_cache
        representatives = [(cache[i], cache[i][2]) for i in (0, 20, 39)]
        result = self.result(setup)
        self.assertIs(result.live_cache, cache)
        with self.assertRaises(LiveCacheHandoffError): self.result(setup)
        with self.assertRaises(BlockExecutionError): setup.block_runner.execute_command(self.plan.commands[0], setup.arena)
        with patch('ds41f_mlx.prefill_fp8_mlx.handoff._generation_session_type', return_value=RecordingGenerationSession):
            session = handoff_to_generation(result, lm, terminal_prompt_token=3, max_tokens=4)
        self.assertIs(session.admitted_cache, cache)
        for i, (item, tensor) in zip((0, 20, 39), representatives):
            self.assertIs(cache[i], item)
            self.assertIs(cache[i][2], tensor)
        self.assertNotIn(3, session.prefix_tokens)
        self.assertEqual(session.inserted_prompt, [3])
        self.assertEqual(session.forwarded, [3])
        self.assertEqual(session.prompt_replay_count, 0)
        self.assertEqual(session.active_cache_offsets(), (2049,)*40)
        self.assertEqual(result.handoff_count, 1)
        self.assertEqual(result.cache_authority_owner, 'BatchGenerator/GenerationBatch')
        self.assertIsNone(setup.block_runner.working_cache)
        with self.assertRaises(LiveCacheHandoffError): result.live_cache
        with self.assertRaises(LiveCacheHandoffError): handoff_to_generation(result, lm, terminal_prompt_token=3)
        with self.assertRaises(RuntimeError): session.start(3)
        with self.assertRaises(BlockExecutionError): setup.block_runner.execute_command(self.plan.commands[0], setup.arena)

    def test_failed_bootstrap_burns_result(self):
        class Bad(RecordingGenerationSession):
            def start(self, terminal):
                super().start(terminal)
                self.scheduler_cache[0][0].value -= 1
        setup, lm = self.setup_ready()
        result = self.result(setup)
        with patch('ds41f_mlx.prefill_fp8_mlx.handoff._generation_session_type', return_value=Bad):
            with self.assertRaises(LiveCacheHandoffError): handoff_to_generation(result, lm, terminal_prompt_token=3)
        with self.assertRaises(LiveCacheHandoffError): handoff_to_generation(result, lm, terminal_prompt_token=3)
        self.assertIsNone(setup.block_runner.working_cache)

    def test_real_session_start_is_once_even_if_bootstrap_fails(self):
        try:
            from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
        except ModuleNotFoundError as exc:
            raise
        session = object.__new__(OMLXGenerationSession)
        session._start_attempted, session._inserted = False, False
        session.admission_report = None
        setup, _ = self.setup_ready()
        session.initial_cache = setup.block_runner.working_cache
        session.admitted_frontier, session.max_tokens = 2048, 4
        session.prefix_tokens, session.sampler = [1]*2048, None
        session.stream, session.mx = None, NS(synchronize=lambda s: None)
        session._bg = NS(insert=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('injected insert failure')))
        with self.assertRaisesRegex(RuntimeError, 'injected'): session.start(3)
        with self.assertRaisesRegex(RuntimeError, 'already attempted'): session.start(3)

    def test_replay_during_bootstrap_fails_closed(self):
        class Replaying(RecordingGenerationSession):
            def start(self, terminal):
                super().start(terminal)
                self.prompt_replay_count = 1
        setup, lm = self.setup_ready()
        result = self.result(setup)
        with patch('ds41f_mlx.prefill_fp8_mlx.handoff._generation_session_type', return_value=Replaying):
            with self.assertRaises(LiveCacheHandoffError): handoff_to_generation(result, lm, terminal_prompt_token=3)
        self.assertEqual(result.cache_authority_owner, 'invalidated')

    def test_pending_publication_fails(self):
        setup, _ = self.setup_ready()
        setup.publication_manager.pending_cumulative_by_layer[20] = {'kv': None}
        with self.assertRaises(LiveCacheHandoffError): self.result(setup)

    def test_real_session_successful_start_seed_and_once(self):
        try:
            from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
        except ModuleNotFoundError as exc:
            raise
        setup, _ = self.setup_ready()
        cache = setup.block_runner.working_cache
        class Generator:
            def insert(obj, prompts, **kwargs):
                obj.prompts, obj.kwargs = prompts, kwargs
                obj._generation_batch = NS(prompt_cache=kwargs['caches'][0])
                return [0]
            def next(obj):
                for item in obj._generation_batch.prompt_cache: item[0].value += 1
                return [NS(end_of_prompt=True, progress=(0, 1))], []
        session = object.__new__(OMLXGenerationSession)
        session._start_attempted, session._inserted = False, False
        session.admission_report, session._final_cache = None, None
        session.initial_cache, session.prefix_tokens = cache, [1]*2048
        session.admitted_frontier, session.max_tokens = 2048, 4
        session.stream, session.sampler = None, None
        session.mx, session._bg = NS(synchronize=lambda s: None), Generator()
        session.start(3)
        self.assertIs(session._bg.kwargs['caches'][0], cache)
        self.assertEqual(session._bg.kwargs['all_tokens'], [[1]*2048])
        self.assertEqual(session._bg.prompts, [[3]])
        self.assertEqual(session.bootstrap_inserted_prompt, (3,))
        self.assertEqual(session.prompt_replay_count, 0)
        self.assertEqual(session.active_cache_offsets(), (2049,)*40)
        self.assertEqual(session.initial_cache, [])
        with self.assertRaises(RuntimeError): session.start(3)






if __name__ == '__main__': unittest.main()
