from __future__ import annotations

import unittest

from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, P8ExecutionOptimizer
from test_prefill_fp8_mlx_p1_p2 import FakeLanguageModel, FakeTensor, full_ready_cache
from test_prefill_fp8_mlx_p6 import RecordingMx, p6_ready_cache


def _contains_fake_tensor(obj, seen=None):
    seen = seen or set()
    oid = id(obj)
    if oid in seen:
        return False
    seen.add(oid)
    if isinstance(obj, FakeTensor):
        return True
    if isinstance(obj, dict):
        return any(_contains_fake_tensor(k, seen) or _contains_fake_tensor(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return any(_contains_fake_tensor(v, seen) for v in obj)
    return False


class P8EvidenceTelemetryTests(unittest.TestCase):
    def test_detach_substep_telemetry_does_not_change_execution(self):
        lm = FakeLanguageModel()
        mx = RecordingMx()
        p8 = P8ExecutionOptimizer(enabled=True, mx=mx)
        lm._p8_optimizer = p8
        app = DeferredPrefillAppend.create(lm, p6_ready_cache(lm, 0, shape_frontier=16384), list(range(16384)), committed_frontier=0, mx=mx)
        app.execute_all()
        names = {b["boundary"] for b in p8.telemetry.materialization_boundaries}
        self.assertIn("persistent_source_eval", names)
        self.assertIn("owned_h_copy", names)
        self.assertIn("owned_next_copy", names)
        self.assertIn("owned_pre_copy", names)
        self.assertIn("arena_detach_rebind", names)
        self.assertIn("materialize_boundary_bookkeeping", names)
        self.assertTrue(app.segment_records[0].materialized)

    def test_lineage_records_retain_no_tensors_and_write_chain_accounting(self):
        lm = FakeLanguageModel()
        mx = RecordingMx()
        p8 = P8ExecutionOptimizer(enabled=True, mx=mx)
        lm._p8_optimizer = p8
        app = DeferredPrefillAppend.create(lm, p6_ready_cache(lm, 0, shape_frontier=16384), list(range(16384)), committed_frontier=0, mx=mx)
        app.execute_all()
        payload = p8.to_json()
        self.assertFalse(_contains_fake_tensor(payload))
        writes = payload["shape_registry"]["write_rows"]
        self.assertTrue(writes)
        self.assertTrue(all("event_id" in w for w in writes))
        self.assertEqual(sum(v["write_count"] for v in payload["shape_registry"]["write_chain_summary"].values()), len(writes))

    def test_diagnostic_barriers_default_off_and_no_extra_eval(self):
        lm = FakeLanguageModel()
        mx = RecordingMx()
        p8 = P8ExecutionOptimizer(enabled=True, mx=mx)
        lm._p8_optimizer = p8
        self.assertEqual(p8.diagnostic_barrier, "off")
        app = DeferredPrefillAppend.create(lm, p6_ready_cache(lm, 0, shape_frontier=16384), list(range(16384)), committed_frontier=0, mx=mx)
        app.execute_all()
        self.assertFalse(p8.telemetry.diagnostic_probes)
        self.assertEqual(len(mx.eval_calls), 4)  # one persistent eval + three existing owned-copy evals in fake mx.copy branch


if __name__ == "__main__":
    unittest.main()
