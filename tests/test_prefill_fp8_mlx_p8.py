import os
import sys
import time
import types
import unittest
from unittest.mock import patch

from ds41f_mlx.prefill_fp8_mlx.p8_optimizer import (
    ExistingKernelFastPathVerifier,
    P8ExecutionOptimizer,
    PerformanceTelemetry,
    ShapeClassRegistry,
)
from tools.run_p8_same_process_cold_warm import deterministic_tokens, _staircase_should_cut, _staircase_bundle_metadata
from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommandKind, SweepPhase


class FakeTensor:
    def __init__(self, shape=(1, 8192, 4, 5120), dtype="bfloat16", device="gpu"):
        self.shape = shape
        self.dtype = dtype
        self.device = device


class FakeKind:
    value = "ENCODE_ROWS"


class FakePhase:
    name = "ENCODER"


class FakeCommand:
    kind = FakeKind()
    phase = FakePhase()
    layer = 0
    rows = 8192


class P8OptimizerTests(unittest.TestCase):
    def test_shape_signature_collapsing_and_tensor_free(self):
        reg = ShapeClassRegistry()
        t = FakeTensor()
        cmd = FakeCommand()
        reg.record_command(region="ENCODE_ROWS", command=cmd, value=t, absolute_start=0, hc_multiplicity=4)
        reg.record_command(region="ENCODE_ROWS", command=cmd, value=t, absolute_start=0, hc_multiplicity=4)
        out = reg.to_json()
        self.assertTrue(out["tensor_free"])
        self.assertEqual(len(out["shape_classes"]), 1)
        self.assertEqual(out["shape_classes"][0]["call_count"], 2)
        dumped = repr(out)
        self.assertNotIn("FakeTensor object", dumped)

    def test_verifier_accounting_direct_record(self):
        ver = ExistingKernelFastPathVerifier()
        ver.record("HC", "fast", "fused_hc_pre_norm", (FakeTensor(),), observation="DIRECT_CALL_OBSERVATION")
        table = ver.required_table()
        self.assertEqual(table[0]["component"], "HC")
        self.assertEqual(table[0]["fast calls"], 1)
        self.assertEqual(table[0]["fallback calls"], 0)

    def test_verifier_wrapper_does_not_change_result(self):
        mod = types.ModuleType("omlx.patches.deepseek_v41.language")
        def fused_hc_pre_norm(x):
            return ("ok", x)
        mod.fused_hc_pre_norm = fused_hc_pre_norm
        with patch.dict(sys.modules, {"omlx": types.ModuleType("omlx"), "omlx.patches": types.ModuleType("patches"), "omlx.patches.deepseek_v41": types.ModuleType("dsv41"), "omlx.patches.deepseek_v41.language": mod}):
            ver = ExistingKernelFastPathVerifier()
            ver.TARGETS = (("omlx.patches.deepseek_v41.language", "fused_hc_pre_norm", "HC", "fast", "DIRECT_CALL_OBSERVATION"),)
            with ver.instrument():
                self.assertEqual(mod.fused_hc_pre_norm(FakeTensor())[0], "ok")
            self.assertEqual(ver.report()[0]["calls"], 1)

    def test_nested_timing_metadata(self):
        tel = PerformanceTelemetry()
        with tel.time_region("Engram micro-pipeline", nesting=("ENCODE_ROWS",), inclusive_overlapping=True):
            time.sleep(0)
        out = tel.to_json()
        self.assertEqual(out["timing_events"][0]["nesting"], ["ENCODE_ROWS"])
        self.assertTrue(out["timing_events"][0]["inclusive_overlapping"])
        self.assertIn("do not sum", out["overlap_warning"])

    def test_instrumentation_disabled_by_default(self):
        old = os.environ.pop("DS41F_P8_OPTIMIZER", None)
        try:
            self.assertIsNone(P8ExecutionOptimizer.from_env())
        finally:
            if old is not None:
                os.environ["DS41F_P8_OPTIMIZER"] = old

    def test_cold_warm_harness_deterministic_tokens_and_fresh_cache_contract(self):
        a = deterministic_tokens(5)
        b = deterministic_tokens(5)
        self.assertEqual(a, b)
        self.assertIsNot(a, b)

    def test_upstream_staircase_cut_matching_and_default_off(self):
        opt = P8ExecutionOptimizer(enabled=True)
        self.assertEqual(opt.diagnostic_barrier, os.environ.get("DS41F_P8_DIAGNOSTIC_BARRIER", "off").lower())
        state = {"completed": set()}
        cmd = types.SimpleNamespace(kind=SweepCommandKind.SWAP_HC_AFTER_LAYER, layer=4, phase=SweepPhase.ENCODER)
        self.assertEqual(_staircase_should_cut(cmd, state), "CUT_A_layers_0_4")
        state["completed"].add("CUT_A_layers_0_4")
        self.assertIsNone(_staircase_should_cut(cmd, state))
        e = types.SimpleNamespace(kind=SweepCommandKind.PUBLISH_FRONTIER, layer=20, phase=SweepPhase.DECODER_SUFFIX)
        self.assertEqual(_staircase_should_cut(e, state), "CUT_E_layer20_source_publication")

    def test_upstream_staircase_bundle_metadata_retains_no_tensors(self):
        t = FakeTensor(shape=(1, 16, 4))
        slot_value = FakeTensor(shape=(1, 2, 4), dtype="uint8")
        carry = types.SimpleNamespace(
            current=types.SimpleNamespace(value=t),
            next=types.SimpleNamespace(value=t),
            pre=types.SimpleNamespace(value=FakeTensor(shape=(1, 16, 4))),
        )
        arena = types.SimpleNamespace(carry=carry)
        class Runner:
            def _p6_persistent_source_values(self, arena):
                return [slot_value], [{"slot_class": "window_KV", "logical_bytes": 8}]
        values, meta = _staircase_bundle_metadata(Runner(), arena)
        self.assertEqual(len(values), 3)  # current and next dedup by identity
        self.assertEqual(meta["persistent_value_count"], 1)
        self.assertEqual(meta["persistent_logical_bytes_known"], 8)
        self.assertEqual(meta["carry_shapes"]["current"], (1, 16, 4))
        self.assertNotIn("values", meta)


if __name__ == "__main__":
    unittest.main()
