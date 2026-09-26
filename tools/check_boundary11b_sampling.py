#!/usr/bin/env python3
"""Check Boundary 11b connected MLX stochastic sampling artifact."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "native-connected-mlx-stochastic-sampling-validation.json"

EXPECTED_LOGITS = "b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d"
EXPECTED_NOISE = "38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4"
EXPECTED_INITIAL = "39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630"
EXPECTED_DRAW = "4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1"
EXPECTED_NEXT = "175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4"


def main() -> int:
    rec = json.loads(ART.read_text())
    assert rec["schema"] == "ds41f.boundary11b.connected-mlx-stochastic-sampling-validation.v1"
    assert rec["base_commit"] == "cb286d836d910d256c8b76a2d04f0e1fcc415586"
    assert rec["ok"] is True
    assert all(rec["gates"].values())
    assert rec["not_omlx_derived"] is True

    assert rec["runtime"]["mlx_version"] == "0.32.2"
    assert rec["runtime"]["target_stream_device"] == "Metal GPU / mx.gpu"
    assert rec["runtime"]["softmax_api_used"].startswith("mx.softmax")

    assert rec["model_execution"]["scope"]["tokens"] == [[0, 3]]
    assert rec["model_execution"]["scope"]["no_logits_artifact_injection"] is True
    assert rec["model_execution"]["logits"]["digest"] == EXPECTED_LOGITS
    assert rec["model_execution"]["logits"]["shape"] == [1, 129280]
    assert rec["model_execution"]["logits"]["dtype"] == "FP32"

    seam = rec["logits_host_to_mlx_seam"]
    assert seam["producer_native_logits_digest"] == EXPECTED_LOGITS
    assert seam["mlx_to_host_roundtrip_digest"] == EXPECTED_LOGITS
    assert seam["exact_fp32_bytes_preserved"] is True
    assert seam["mlx_consumer_tensor"]["dtype"] == "mlx.core.float32"
    assert seam["mlx_consumer_tensor"]["shape"] == [1, 129280]

    temp = rec["temperature"]
    assert temp["temperature"] == 1.0
    assert temp["effective_temperature"] == 1.0
    assert temp["temperature_floor_reviewed"] is True
    assert temp["temperature_floor_exercised"] is False

    rng = rec["rng"]
    assert rng["seed"] == 289513473
    assert rng["session_key_before"]["values"] == [0, 289513473]
    assert rng["session_key_before"]["digest"] == EXPECTED_INITIAL
    assert rng["draw_key"]["values"] == [2308264947, 2363689365]
    assert rng["draw_key"]["digest"] == EXPECTED_DRAW
    assert rng["next_session_key"]["values"] == [613528892, 572711044]
    assert rng["next_session_key"]["digest"] == EXPECTED_NEXT

    noise = rec["noise"]
    assert noise["digest"] == EXPECTED_NOISE
    assert noise["shape"] == [1, 129280]
    assert noise["dtype"] == "float32"
    assert noise["all_positive"] is True
    assert noise["all_finite"] is True
    assert "not a mathematically exact continuous Exp(1)" in noise["not_strengthened_claim"]

    ar = rec["sampling_arithmetic"]
    assert ar["target_mlx_sampled_token_id"] == ar["independent_log_domain_token"]
    assert ar["boundary10b_separation"]["boundary10b_conditional_token"] == 795
    assert ar["boundary10b_separation"]["not_expected_value_for_boundary11b"] is True
    soft = ar["probs"]
    assert soft["max_abs_vs_independent"] <= soft["predeclared_tolerance"]["softmax_max_abs_lte"]
    assert soft["sum_abs_error_vs_independent"] <= soft["predeclared_tolerance"]["softmax_sum_abs_error_lte"]
    assert ar["winner_evidence"]["runtime_score_domain"]["tie_count"] == 1
    assert ar["winner_evidence"]["independent_log_domain"]["tie_count"] == 1

    rep = rec["reproducibility"]
    assert rep["same_key_whole_path"]["same_logits"] is True
    assert rep["same_key_whole_path"]["same_generated_noise"] is True
    assert rep["same_key_whole_path"]["same_next_session_key"] is True
    assert rep["same_key_whole_path"]["same_sampled_token"] is True
    assert rep["secondary_key"]["noise_differs_from_primary"] is True
    assert rep["secondary_key"]["token_difference_not_required"] is True

    assert rec["pytorch_parity"]["pytorch_noise_bitstream_used"] is False
    assert rec["pytorch_parity"]["pytorch_seed_parity_required"] is False
    assert rec["pytorch_parity"]["pytorch_sampled_token_parity_required"] is False
    assert rec["stop_boundary"]["next_source_operation_recorded_not_executed"] == "main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None"
    assert "no PyTorch/MLX RNG bitwise parity" in rec["non_claims"]
    assert "no main_hidden assembly" in rec["non_claims"]
    print(f"Boundary11b sampling check PASS: {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
