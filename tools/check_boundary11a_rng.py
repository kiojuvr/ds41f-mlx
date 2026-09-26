#!/usr/bin/env python3
"""Check Boundary 11a MLX Exp(1) RNG validation artifact."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "mlx-exp1-rng-validation.json"
BASE = "c9c8ab1de87c69e1af7e8cb74b2bdff528ec75a1"
PRIMARY_N = 1_048_576
PRIMARY_SHAPE = [1, 129280]


def main() -> int:
    rec = json.loads(ART.read_text())
    assert rec["schema"] == "ds41f.boundary11a.mlx-exp1-rng-validation.v1"
    assert rec["base_commit"] == BASE
    assert rec["ok"] is True
    assert rec["not_omlx_derived"] is True
    assert rec["official_source_semantics"]["pytorch_bitstream_reproduction_required"] is False
    assert "same bit sequence as PyTorch" in rec["not_semantic_requirement"]
    assert rec["runtime_inventory"]["mlx_version"]
    assert rec["runtime_inventory"]["target_stream_device"] == "Metal GPU / mx.gpu"
    assert rec["runtime_inventory"]["apis"]["key"]["present"] is True
    assert rec["runtime_inventory"]["apis"]["split"]["present"] is True
    assert rec["runtime_inventory"]["apis"]["uniform"]["present"] is True
    assert rec["runtime_inventory"]["apis"]["randint"]["present"] is True
    assert rec["runtime_inventory"]["native_exponential_present"] is False
    assert rec["exp1_construction"]["case"].startswith("B:")
    assert "2^-24" in rec["exp1_construction"]["formula"]
    assert rec["exp1_construction"]["output_dtype"] == "FP32"
    assert rec["rng_state_policy"]["production_contract"].startswith("sampling session owns an explicit MLX PRNG key")

    stat = rec["predeclared_statistical_contract"]
    assert stat["N"] >= PRIMARY_N
    assert stat["alpha"] == 1e-6
    expected_eps = math.sqrt(math.log(2.0 / stat["alpha"]) / (2.0 * stat["N"]))
    assert abs(stat["dkw_epsilon"] - expected_eps) < 1e-18
    assert abs(stat["mean_gate_abs_error_lte"] - 6.0 / math.sqrt(stat["N"])) < 1e-18
    assert rec["theoretical_authority"]["distribution"] == "Exp(rate=1)"

    runs = rec["statistical_fixture"]["runs"]
    assert len(runs) >= 2
    for run in runs:
        assert run["shape"] == [stat["N"]]
        assert run["dtype"] == "mlx.core.float32"
        assert run["all_finite"] is True
        assert run["all_positive"] is True
        assert run["ks"] <= stat["dkw_epsilon"]
        assert run["ks_pass"] is True
        assert abs(run["mean"] - 1.0) <= stat["mean_gate_abs_error_lte"]
        assert run["mean_gate_pass"] is True
        assert set(run["quantiles"].keys()) >= {"0.001", "0.01", "0.1", "0.5", "0.9", "0.99", "0.999"}

    prod = rec["production_shape_fixture"]
    assert prod["shape"] == PRIMARY_SHAPE
    assert prod["dtype"] == "mlx.core.float32"
    assert prod["digest"]
    assert prod["all_positive"] is True and prod["all_finite"] is True
    assert prod["key_before"] and prod["key_after"]

    rep = rec["reproducibility"]
    assert rep["same_key_same_shape_exact_same_output"] is True
    assert rep["same_key_same_shape_exact_same_next_key"] is True
    assert rep["different_key_different_output"] is True

    assert rec["gates"]["boundary9_closeout_remains_pass"] is True
    assert rec["gates"]["boundary10_closeout_remains_pass"] is True
    assert all(rec["gates"].values())
    assert "no actual sampled-token authority" in rec["non_claims"]
    assert "no PyTorch/MLX RNG bitwise parity" in rec["non_claims"]
    print(f"Boundary11a RNG check PASS: {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
