import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import ds41f_mlx.runtime.dwarfstar_prefill as dp
from ds41f_mlx.prefill_fp8_mlx.telemetry import summarize_p7_engram_events
from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend, RequestTrace


class FakeLanguageModel:
    def __init__(self):
        self._p7_enable_overlap = False
        self.configured_mtp = None
    def configure_mtp(self, enabled, value):
        self.configured_mtp = (enabled, value)
    def make_cache(self):
        return [object() for _ in range(40)]


class FakeModel:
    def __init__(self):
        self.language_model = FakeLanguageModel()


class FakeAppend:
    created = []
    def __init__(self, lm, cache, ids, committed_frontier, mx):
        self.language_model = lm
        self.live_cache = cache
        self.ids = tuple(ids)
        self.commit_certificate = None
        self.segment_records = []
    @classmethod
    def create(cls, lm, cache, ids, *, committed_frontier, mx):
        inst = cls(lm, cache, ids, committed_frontier, mx)
        cls.created.append((lm, cache, tuple(ids), committed_frontier, mx))
        return inst
    def execute_all(self):
        self.commit_certificate = SimpleNamespace(final_setup=SimpleNamespace())


class FakeLiveResult:
    def __init__(self, ids):
        self.prefix_token_ids = tuple(ids)
        self.frontier = len(ids)
        self.handoff_count = 0
        self._cache = [object()]
    @property
    def live_cache(self):
        return self._cache


class ProductionSelectorTests(unittest.TestCase):
    def test_facade_defaults_to_dense_p0_p7_and_not_legacy_one_chunk(self):
        model = FakeModel()
        made = []
        def fake_from_committed(commit, *, prefix_token_ids):
            made.append((commit, tuple(prefix_token_ids)))
            return FakeLiveResult(prefix_token_ids)
        with patch.object(dp, "DeferredPrefillAppend", FakeAppend), patch.object(dp.LivePrefillResult, "from_committed", side_effect=fake_from_committed):
            session = dp.DwarfStarMLXPrefillSession(model, mx=object())
            result = session.prefill([29, 30, 31])
        self.assertEqual(result.production_prefill_selector, "DENSE_P0_P7")
        self.assertEqual(result.frontier, 3)
        self.assertTrue(model.language_model._p7_enable_overlap)
        self.assertEqual(model.language_model.configured_mtp, (False, 1))
        self.assertIsInstance(session._delegate, dp.DenseP0P7PrefillSession)
        self.assertIsNot(session._delegate.__class__, dp.LegacyOneChunkMLXPrefillSession)
        self.assertEqual(FakeAppend.created[-1][2], (29, 30, 31))
        self.assertEqual(made[-1][1], (29, 30, 31))

    def test_tile_native_rejected_candidate_fails_closed(self):
        old = os.environ.get("DS41F_P8_TILE_NATIVE_CARRY")
        os.environ["DS41F_P8_TILE_NATIVE_CARRY"] = "1"
        try:
            with self.assertRaises(RuntimeError):
                dp.DwarfStarMLXPrefillSession(FakeModel(), mx=object())
        finally:
            if old is None:
                os.environ.pop("DS41F_P8_TILE_NATIVE_CARRY", None)
            else:
                os.environ["DS41F_P8_TILE_NATIVE_CARRY"] = old

    def test_p7_engram_fallback_uses_m6_logical_match_definition(self):
        summary = summarize_p7_engram_events([
            {"event": "engram_consume", "logical_match": True, "donor_issue_observed": True},
            {"event": "engram_consume", "logical_match": False, "donor_issue_observed": False},
            {"event": "engram_prefetch_submit"},
        ])
        self.assertEqual(summary["foreground_engram_fallback"], 1)
        self.assertEqual(summary["logical_match_consumptions"], 1)
        self.assertEqual(summary["prefetch_submissions"], 1)

    def test_trace_retention_is_bounded_deque(self):
        with patch.dict(os.environ, {"DS41F_TRACE_HISTORY_LIMIT": "2"}):
            backend = DeepSeekRecipeRuntimeBackend()
        for i in range(3):
            backend.traces.append(RequestTrace(str(i), "chat_completions", 2, 1, 7))
        self.assertEqual([t.request_id for t in backend.traces], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
