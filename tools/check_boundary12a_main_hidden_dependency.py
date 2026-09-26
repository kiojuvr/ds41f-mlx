#!/usr/bin/env python3
"""Check Boundary12a main_hidden / Engram dependency audit artifact."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "main-hidden-engram-dependency-audit.json"
CKPT = Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    rec = json.loads(ART.read_text())

    assert subprocess.run([sys.executable, str(ROOT / "tools" / "check_boundary11_closeout.py")], cwd=ROOT).returncode == 0
    assert rec["schema"] == "ds41f.boundary12a-main-hidden-engram-dependency-audit.v1"
    assert rec["ok"] is True
    assert rec["base_head"] == "4701f5e1c73e7638744f169439f227b9dfa098fa"
    assert rec["not_omlx_derived"] is True
    assert rec["semantic_authority"] == "local pinned official source/config only"
    assert rec["source_hash_method"] == "raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1"

    ids = rec["source_identities"]
    assert ids["model_py"]["sha256"] == sha(CKPT / "inference/model.py")
    assert ids["engram_py"]["sha256"] == sha(CKPT / "inference/engram.py")
    assert ids["config_json"]["sha256"] == sha(CKPT / "inference/config.json")
    assert ids["tokenizer_json"]["sha256"] == sha(CKPT / "tokenizer.json")
    assert ids["tokenizer_config_json"]["sha256"] == sha(CKPT / "tokenizer_config.json")
    assert ids["model_safetensors_index_json"]["sha256"] == sha(CKPT / "model.safetensors.index.json")

    cfg = rec["local_config"]
    assert cfg["dspark_target_layer_ids"] == [37, 38, 39]
    assert cfg["engram_layer_ids"] == [1, 14]
    assert cfg["hc_mult"] == 4
    assert cfg["dim"] == 5120
    assert cfg["n_layers"] == 40
    assert cfg["n_mtp_layers"] == 3
    assert cfg["matches_expected_public_checkpoint_values"] is True

    ordering = rec["ordering_conclusions"]
    assert ordering["target_layer_capture_occurs_after_optional_engram_update"] is True
    assert ordering["target_layer_capture_occurs_before_Block_forward"] is True
    assert "torch.cat(main_hiddens, dim=-1) if main_hiddens else None" in ordering["main_hidden_concat_order_recorded"]
    assert ordering["return_tuple_order_recorded"] == "return output_ids, logits, main_hidden at line 1272"

    ops = [x["operation"] for x in rec["producer_ordering"]]
    assert any("self.engram_hash" in x for x in ops)
    assert any("layer.engram" in x and "h =" in x for x in ops)
    assert any("main_hiddens.append(h.mean(dim=2))" == x for x in ops)
    assert ops.index("if layer.engram is not None:") < ops.index("if i in self.target_layer_ids:")
    assert ops.index("main_hiddens.append(h.mean(dim=2))") < ops.index("h, pre_mix = layer(h, start_pos, pre_mix, image_mask)")

    assert rec["upstream_engram_dependency_recognized"] is True
    dep = "\n".join(rec["dependency_graph"])
    assert "Engram@1" in dep and "Engram@14" in dep and "pre-Block37 h -> mean(HC)" in dep

    shape = rec["structural_main_hidden_contract"]
    assert shape["released_config_branch"] == "non-empty main_hiddens branch"
    assert shape["empty_branch_contract"] == "if target_layer_ids is empty then main_hidden = None"
    assert shape["expected_fixture_shape"] == [1, 2, 15360]

    mean = rec["mean_dtype_arithmetic_contract_review"]
    assert mean["source_operation"] == "h.mean(dim=2)"
    assert "not promoted to numeric authority" in mean["accumulation_semantics"]

    inv = rec["current_repo_engram_authority_inventory"]
    assert inv["engram_numeric_authority_present"] is False
    assert inv["official_numerical_validation_authority_present"] is False

    gate = rec["promotion_gate"]
    assert gate["official_config_has_non_empty_engram_layer_ids"] is True
    assert gate["Transformer_forward_applies_Engram_upstream_of_target_hidden_captures"] is True
    assert gate["current_runtime_lacks_validated_Engram_numerical_authority"] is True
    assert gate["main_hidden_numeric_promotion_blocked_by_engram"] is True

    claims = rec["claims"]
    assert claims["numeric_main_hidden_authority"] is False
    assert claims["full_Transformer_forward_authority"] is False
    assert claims["return_tuple_correctness_executed_or_validated"] is False
    assert "official numeric main_hidden values" in rec["current_no_engram_connected_path_cannot_validate"]
    assert "official full Transformer.forward return" in rec["current_no_engram_connected_path_cannot_validate"]
    assert "structural placement of main_hidden capture" in rec["current_no_engram_connected_path_can_validate"]

    assert all(rec["authority_source_guards"].values())
    assert all(rec["gates"].values())
    assert rec["recommended_next_boundary"]["if_audit_matches_public_and_no_engram_numeric_authority"] == "Boundary 12b: Engram semantic foundation"

    forbidden = ["numeric main_hidden authority", "Transformer.forward return correctness", "full-model correctness"]
    for phrase in forbidden:
        assert phrase in "\n".join(rec["non_claims"])

    print(f"Boundary12a main_hidden dependency audit check PASS: {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
