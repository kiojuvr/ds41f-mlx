#!/usr/bin/env python3
"""Record the source-extension build gate, without implying a live MTP run."""
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PIN = "8cadfede7063c896b944e7bae05daa3549ae97ea"
OMLX_PIN = "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"
PATCH_REV = "066d2ef2ed0deb574a0e6b5e6136316d6f296d55"
BASE = Path("/tmp/ds41f-m31-base")
PATCHED = Path("/tmp/ds41f-m31-recipe")
OUT = ROOT / "artifacts/m31"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def git(path, *args):
    return command("git", "-C", str(path), *args)


def main():
    if git(BASE, "rev-parse", "HEAD") != RECIPE_PIN or git(BASE, "status", "--porcelain"):
        raise RuntimeError("base recipe authority is not clean/exact")
    if git(PATCHED, "rev-parse", "HEAD") != PATCH_REV or git(PATCHED, "status", "--porcelain"):
        raise RuntimeError("patched recipe is not exact committed candidate")
    omlx = Path.home() / "omlx-0.7.0.release"
    if git(omlx, "rev-parse", "HEAD") != OMLX_PIN or git(omlx, "status", "--porcelain"):
        raise RuntimeError("oMLX authority is not clean/exact")
    files = git(PATCHED, "diff", "--name-only", RECIPE_PIN, PATCH_REV).splitlines()
    hashes = {file: {"base_sha256": sha(BASE / file) if (BASE / file).exists() else None,
                     "patched_sha256": sha(PATCHED / file)} for file in files}
    sys.path.insert(0, str(omlx))
    import omlx.api.tool_calling as tool_calling
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    native_base = Path.home() / ".venvs/omlx-0.7.0.release/lib/python3.13/site-packages/deepseek_recipe/_native.abi3.so"
    m30_identity = json.loads((ROOT / "artifacts/m30/runtime-identities.json").read_text())
    if sha(native_base) != m30_identity["native_sha256"]:
        raise RuntimeError("preserved release binary differs from M30 identity")
    identity = {"ds41f_base_commit": "bd28b8186667e0942b333c22c574aa1a107b2328",
                "ds41f_result_commit_lookup": "git log -1 --format=%H -- artifacts/m31/qualification.json",
                "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                "packages": packages, "rustc": command("rustc", "-Vv"), "cargo": command("cargo", "-V"),
                "recipe_base_revision": RECIPE_PIN, "recipe_patched_revision": PATCH_REV,
                "recipe_base_worktree_clean": True, "recipe_patched_worktree_clean": True,
                "recipe_source_hashes": hashes,
                "recipe_cargo_lock_sha256": sha(PATCHED / "Cargo.lock"),
                "base_recipe_cargo_lock_sha256": sha(BASE / "Cargo.lock"),
                "recipe_patch_sha256": sha(OUT / "recipe-semantic-preview.patch"),
                "recipe_distribution_source_version": tomllib.loads((PATCHED / "deepseek-recipe-python/pyproject.toml").read_text())["project"]["version"],
                "native_source_version": "0.1.0", "patched_native_binary_sha256": None,
                "patched_native_status": "not built or installed: OpenCV development metadata missing",
                "preserved_release_native_sha256": sha(native_base),
                "preserved_release_native_usage": "identity only; never imported for M31 extension tests",
                "tokenizer_sha256": sha(PATCHED / "static/tokenizers/v41/tokenizer.json"),
                "base_tokenizer_sha256": sha(BASE / "static/tokenizers/v41/tokenizer.json"),
                "omlx_revision": OMLX_PIN, "omlx_worktree_clean": True,
                "omlx_tool_calling_import": tool_calling.__file__, "omlx_tool_calling_import_ok": True,
                "omlx_mtp_verify_source_sha256": sha(omlx / "omlx/patches/mlx_lm_mtp/batch_generator.py"),
                "source_preview_driver_binary_sha256": sha(Path("/tmp/ds41f-m31-cargo/release/m31-parity-driver")),
                "ds41f_dependency_changes": [], "preserved_release_installation_modified": False}
    (OUT / "runtime-identities.json").write_text(json.dumps(identity, indent=2) + "\n")
    parity = json.loads((OUT / "canonical-parity.json").read_text())
    mapping = json.loads((OUT / "source-preview-mapping.json").read_text())
    cases = ["ordinary_text", "EOS", "max_tokens", "one_token_stop", "multi_token_stop",
             "shared_prefix_stop", "unicode_adjacent_stop", "stop_spanning_verify_blocks",
             "one_valid_DSML_call", "multiple_calls_one_DSML_block", "thinking_to_DSML",
             "DSML_close_spanning_candidate_groups", "full_acceptance", "partial_acceptance",
             "immediate_rejection", "terminal_inside_accepted_prefix", "terminal_as_bonus_correction",
             "rejection_before_possible_terminal", "interruption_before_tool_completion",
             "interruption_committed_undelivered_safe_prefix", "quiescence_after_tool_completion",
             "P6_P5_tool_result_MTP_reentry"]
    record = {"schema": "ds41f.m31.qualification.v1", "ds41f_base_commit": identity["ds41f_base_commit"],
              "decision": "BUILD_OR_BINDING_BLOCKED", "protocol_gate_passed": False,
              "production_mtp": "OFF", "public_mtp_option": "disabled", "mtp_persistence": "fail closed",
              "recipe_base_revision": RECIPE_PIN, "recipe_patch_revision": PATCH_REV,
              "recipe_patch_sha256": identity["recipe_patch_sha256"], "omlx_revision": OMLX_PIN,
              "extension": {"synchronous_session": True, "same_decoder_and_state_machine": True,
                            "explicit_terminal_observation": True, "python_api_source_implemented": True,
                            "python_native_runtime_qualified": False,
                            "rust_canonical_parity": parity["equal"], "rust_parity_comparisons": parity["comparison_count"],
                            "rust_unit_tests": "10 passed; includes state fingerprints and error nonmutation",
                            "binding_typecheck": "PASS with DOCS_RS=1, compile-only; not a linked/runnable native build",
                            "binding_tests": "collection blocked: no newly built deepseek_recipe module"},
              "mtp_semantic_hook_installed": False, "terminal_metadata_to_emission_installed": False,
              "live_qualification": {case: {"status": "BLOCKED_NATIVE_BUILD", "target_frontiers": None,
                                      "dspark_frontiers": None, "canonical_parser_frontier": None,
                                      "canonical_history_frontier": None, "acceptance_topology": None}
                                     for case in cases},
              "source_only_preview_cost": {"preview": mapping["preview_latency"], "clone": mapping["clone_latency"],
                                           "scope": mapping["scope"], "window_ids": 6},
              "live_preview_cost": None, "live_mtp_throughput_tok_s": None, "live_mtp_acceptance": None,
              "off_on_live_parity": "not run", "tool_result_reentry": "not run: protocol gate blocked",
              "operational_soak": "not run", "model_execution_count": 0,
              "model_history_replay": 0, "full_target_cache_repack": 0, "new_verify_during_quiesce": 0,
              "new_proposal_during_quiesce": 0,
              "zero_counter_scope": "no model or quiesce executed; NOT live MTP zero-replay proof",
              "parser_history_replay_for_preview": 0,
              "historical_fixture_diagnostic_history_feeds": 5,
              "parser_replays_outside_preview": "historical fixture fed in raw/chat parity runs for each source and once incrementally in mapping driver; diagnostic only, never per preview/cycle",
              "blocker": "Real PyO3 build fails in opencv 0.93.7: opencv4.pc/OpenCVConfig.cmake/headers unavailable. DOCS_RS cannot qualify native runtime.",
              "next_scope": "build exact native dependency, run binding parity, then implement pre-commit clamp + emission metadata and live tool/stop/frontier/reentry gate; no operational soak"}
    (OUT / "qualification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"decision": record["decision"], "recipe_patch_revision": PATCH_REV,
                      "canonical_parity": parity["equal"], "patched_native_binary": None}, indent=2))


if __name__ == "__main__":
    main()
