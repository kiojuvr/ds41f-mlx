"""Checks source-extension evidence; never certifies a live/native protocol gate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/m31"


def load(name):
    return json.loads((OUT / name).read_text())


def test_m31_stops_at_real_native_build_gate():
    record = load("qualification.json")
    assert record["decision"] == "BUILD_OR_BINDING_BLOCKED"
    assert record["production_mtp"] == "OFF"
    assert record["public_mtp_option"] == "disabled"
    assert record["mtp_persistence"] == "fail closed"
    assert not record["protocol_gate_passed"]
    assert not record["mtp_semantic_hook_installed"]
    assert not record["terminal_metadata_to_emission_installed"]
    assert record["model_execution_count"] == 0
    assert record["live_mtp_throughput_tok_s"] is None
    assert not record["extension"]["python_native_runtime_qualified"]
    for case in record["live_qualification"].values():
        assert case["status"] == "BLOCKED_NATIVE_BUILD"
        assert case["target_frontiers"] is None
        assert case["dspark_frontiers"] is None
        assert case["acceptance_topology"] is None


def test_patch_and_recipe_identities_are_explicit():
    identity = load("runtime-identities.json")
    assert identity["recipe_base_revision"] == "8cadfede7063c896b944e7bae05daa3549ae97ea"
    assert identity["recipe_patched_revision"] == "066d2ef2ed0deb574a0e6b5e6136316d6f296d55"
    assert identity["omlx_revision"] == "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"
    assert identity["patched_native_binary_sha256"] is None
    assert identity["recipe_patch_sha256"] == hashlib.sha256((OUT / "recipe-semantic-preview.patch").read_bytes()).hexdigest()
    assert identity["tokenizer_sha256"] == identity["base_tokenizer_sha256"]
    assert identity["ds41f_dependency_changes"] == []
    assert identity["omlx_tool_calling_import_ok"]
    assert identity["packages"]["jsonschema"] == "4.26.0"


def test_canonical_source_parity_is_exact_except_clock():
    result = load("canonical-parity.json")
    assert result["equal"]
    assert result["case_count"] == 22
    assert result["comparison_count"] == 64
    assert all(case["equal"] for case in result["comparisons"])
    assert load("canonical-base.json") == load("canonical-patched.json")


def test_preview_maps_terminal_to_completing_id_not_retokenized_span():
    result = load("source-preview-mapping.json")
    rows = [row for case in result["cases"] for row in case["rows"]]
    assert rows
    assert any(row["pending_canonical_ids_before"] for row in rows)
    assert any(row["terminal"] and row["terminal"]["source_start"] < row["source_bytes_before"] for row in rows)
    for row in rows:
        assert row["mapping_exact"]
        assert row["canonical_terminal"] == row["terminal"]
        assert row["target_frontier"] is None
        assert row["dspark_frontier"] is None
        if row["terminal"]:
            assert row["safe_token_count"] == row["completing_token_index"]
            assert row["completing_token_index"] < len(row["candidate_ids"])
        else:
            assert row["safe_token_count"] == len(row["candidate_ids"])
            assert row["completing_token_index"] is None


def test_cost_is_incremental_source_only_not_claimed_mtp_throughput():
    result = load("source-preview-mapping.json")
    assert result["candidate_window_ids"] == 6
    assert result["preview_latency"]["samples"] == 15400
    assert result["clone_latency"]["samples"] == 15400
    assert result["parser_history_replay"] == 0
    assert "NOT native Python or live MTP" in result["scope"]


def test_real_native_build_failure_is_not_hidden_by_compile_only_check():
    build = (OUT / "native-build.log").read_text()
    bindings = (OUT / "binding-tests.log").read_text()
    assert "opencv" in build
    assert "Failed to find installed OpenCV" in build
    assert "ModuleNotFoundError" in bindings
    assert "test_semantic_preview" in bindings
    assert "test_bindings" in bindings
