#!/usr/bin/env python3
"""Check Boundary 11 closeout artifact."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "boundary11-closeout.json"

EXPECTED_LOGITS = "b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d"
EXPECTED_NOISE = "38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4"
EXPECTED_PROBS = "e08896fb5461b28e4019cdf978f6e91db521b94fee3b91e757fdf25bb8366ea6"
EXPECTED_INITIAL = "39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630"
EXPECTED_DRAW = "4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1"
EXPECTED_NEXT = "175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4"
FORBIDDEN_BROAD = [
    "official sampled token is 9468",
    "DeepSeek canonical token is 9468",
    "PyTorch sampling matched",
    "full stochastic model validated",
    "full Transformer.forward validated",
    "full model validated",
]


def main() -> int:
    rec = json.loads(ART.read_text())
    b9 = json.loads((ROOT / "artifacts" / "boundary9-closeout.json").read_text())
    b10 = json.loads((ROOT / "artifacts" / "boundary10-closeout.json").read_text())
    b11a = json.loads((ROOT / "artifacts" / "mlx-exp1-rng-validation.json").read_text())
    b11b = json.loads((ROOT / "artifacts" / "native-connected-mlx-stochastic-sampling-validation.json").read_text())

    assert rec["schema"] == "ds41f.boundary11-closeout.v1"
    assert rec["ok"] is True
    assert rec["not_omlx_derived"] is True
    assert b9["ok"] is True
    assert b10["ok"] is True
    assert b11a["ok"] is True
    assert b11b["ok"] is True
    assert subprocess.run([sys.executable, str(ROOT / "tools" / "check_boundary11a_rng.py")], cwd=ROOT).returncode == 0
    assert subprocess.run([sys.executable, str(ROOT / "tools" / "check_boundary11b_sampling.py")], cwd=ROOT).returncode == 0

    rt = rec["runtime"]
    assert rt["mlx_version"] == "0.32.2"
    assert rt["target_device"] == "Metal GPU / mx.gpu"

    ah = rec["authority_hierarchy"]
    assert ah["Boundary9"]["authority"] == "current integrated deterministic model-forward authority through final-position logits"
    assert ah["Boundary9"].get("not_stochastic_model_forward_authority") is True
    assert ah["Boundary11b"]["supersedes_prior_boundaries"] is False

    seam = rec["boundary11b"]["host_to_mlx_logits_seam"]
    assert seam["native_logits_digest"] == EXPECTED_LOGITS
    assert seam["mlx_roundtrip_digest"] == EXPECTED_LOGITS
    assert seam["exact_fp32_bytes_preserved"] is True
    assert seam["mlx_dtype"] == "mlx.core.float32"
    assert seam["mlx_shape"] == [1, 129280]

    assert rec["boundary11b"]["boundary9_regression"]["logits_digest"] == EXPECTED_LOGITS
    assert rec["boundary11b"]["boundary9_regression"]["artifact_tensor_injection"] is False

    rng = rec["boundary11b"]["primary_rng_state"]
    assert rng["seed"] == 289513473
    assert rng["session_key"]["values"] == [0, 289513473]
    assert rng["session_key"]["digest"] == EXPECTED_INITIAL
    assert rng["draw_key"]["values"] == [2308264947, 2363689365]
    assert rng["draw_key"]["digest"] == EXPECTED_DRAW
    assert rng["next_session_key"]["values"] == [613528892, 572711044]
    assert rng["next_session_key"]["digest"] == EXPECTED_NEXT
    assert rng["next_session_key_is_part_of_result_state_transition"] is True

    noise = rec["boundary11b"]["actual_mlx_noise"]
    assert noise["digest"] == EXPECTED_NOISE
    assert noise["shape"] == [1, 129280]
    assert noise["dtype"] == "FP32"
    assert noise["all_positive"] is True and noise["all_finite"] is True
    assert "endpoint policy" in noise["classification"]

    sm = rec["boundary11b"]["mlx_sampling_arithmetic"]["softmax"]
    assert sm["api"] == "mx.softmax(scaled_logits, axis=-1, stream=mx.gpu)"
    assert sm["digest"] == EXPECTED_PROBS
    assert sm["max_abs_vs_independent"] <= sm["predeclared_contract"]["max_abs_lte"]
    assert sm["sum_abs_error_vs_independent"] <= sm["predeclared_contract"]["sum_abs_error_lte"]
    assert sm["bit_exact_claim"] is False

    tok = rec["boundary11b"]["runtime_sampled_token_authority"]
    assert tok["target_mlx_sampled_token_id"] == 9468
    assert tok["independent_log_domain_token"] == 9468
    assert tok["winner_agreement"] is True
    assert tok["tie_count"] == 1
    assert tok["classification"] == "pinned target-MLX-runtime sampled token"
    assert tok["forbidden_classifications_absent"] is True

    sep = rec["boundary10b_separation"]
    assert sep["boundary10b_token"] == 795
    assert sep["not_expected_value_for_boundary11b"] is True

    rep = rec["reproducibility_authority"]
    assert rep["primary_same_key_rerun"]["same_logits"] is True
    assert rep["primary_same_key_rerun"]["same_generated_noise"] is True
    assert rep["primary_same_key_rerun"]["same_next_session_key"] is True
    assert rep["primary_same_key_rerun"]["same_sampled_token"] is True
    assert rep["secondary_seed"]["noise_differs"] is True
    assert rep["secondary_seed"]["different_token_is_diagnostic_only"] is True

    pyt = rec["pytorch_compatibility_classification"]
    assert pyt["pytorch_noise_bitstream_used"] is False
    assert pyt["pytorch_seed_parity_required"] is False
    assert pyt["pytorch_sampled_token_parity_required"] is False

    stop = rec["stop_boundary"]
    assert stop["stopped_after"]["target_mlx_sampled_token_id"] == 9468
    assert stop["stopped_after"]["next_session_key"] == [613528892, 572711044]
    assert stop["next_source_operation"] == "main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None"
    assert stop["main_hidden_assembly"] == "not executed"
    assert stop["transformer_forward_return"] == "not executed"

    guard = rec["authority_contamination_guard"]
    assert all(guard.values())
    assert all(rec["gates"].values())
    assert "no PyTorch/MLX RNG bitwise parity" in rec["non_claims"]
    assert "no main_hidden assembly" in rec["non_claims"]

    blob = json.dumps(rec)
    for phrase in FORBIDDEN_BROAD:
        # Forbidden phrases are allowed only as explicit forbidden examples, not claims.
        if phrase in blob:
            assert phrase in rec.get("forbidden_wording", [])

    print(f"Boundary11 closeout check PASS: {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
