"""Unified operational qualification runner for the ds41f scoped text release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.provenance import inspect_runtime

ROOT = Path(__file__).resolve().parents[1]

CHEAP_GATES = [
    [sys.executable, "-m", "unittest", "tests/test_stateful_request_policy.py", "tests/test_runtime_config.py", "tests/test_m22_release_metadata.py"],
    ["cargo", "test"],
    [sys.executable, "tools/check_legacy_import_integrity.py"],
    [sys.executable, "tools/check_native_import_dependencies.py"],
    [sys.executable, "tools/check_repository_self_containment.py"],
    ["cmake", "-S", "native", "-B", "native/build"],
    ["cmake", "--build", "native/build"],
    ["ctest", "--test-dir", "native/build", "--output-on-failure"],
]

FULL_REAL_MODEL_GATES = [
    [sys.executable, "tools/run_m12_sessionized_http_qualification.py"],
    [sys.executable, "tools/run_m13_repeated_tool_agent_qualification.py"],
    [sys.executable, "tools/run_m14_termination_qualification.py"],
]

REAL_MODEL_ARTIFACTS = [
    ROOT / "artifacts/m10/restored-long-session-qualification.json",
    ROOT / "artifacts/m12/sessionized-http-qualification.json",
    ROOT / "artifacts/m13/repeated-tool-agent-qualification.json",
    ROOT / "artifacts/m14/termination-qualification.json",
]


def qualification_env() -> dict[str, str]:
    import os
    cfg = load_runtime_config()
    paths = [str(ROOT), str(cfg.omlx_path)]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ":".join(paths + ([existing] if existing else []))
    env.setdefault("DS41F_CHECKPOINT", str(cfg.checkpoint_path))
    env.setdefault("DS41F_OMLX_PATH", str(cfg.omlx_path))
    env.setdefault("DS41F_RECIPE_PATH", str(cfg.recipe_path))
    env.setdefault("DS41F_KV_ROOT", str(cfg.kv_root))
    return env


def run_command(cmd: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=qualification_env())
        status = "PASS" if proc.returncode == 0 else "FAIL"
        return {"command": cmd, "status": status, "returncode": proc.returncode, "seconds": time.time() - started, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        return {"command": cmd, "status": "FAIL", "returncode": None, "seconds": time.time() - started, "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "", "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "", "error": "timeout"}
    except Exception as exc:
        return {"command": cmd, "status": "FAIL", "returncode": None, "seconds": time.time() - started, "stdout_tail": "", "stderr_tail": "", "error": repr(exc)}


def artifact_path(mode: str, output: Path | None) -> Path:
    if output is not None:
        return output
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return ROOT / "artifacts" / "release" / f"qualification-{mode}-{stamp}.json"


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def summarize_existing_real_model_evidence() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in REAL_MODEL_ARTIFACTS:
        data = load_json_if_present(path)
        if data is None:
            out[str(path.relative_to(ROOT))] = {"present": False}
            continue
        summary: dict[str, Any] = {"present": True, "schema": data.get("schema"), "ok": data.get("ok", data.get("passed"))}
        text = json.dumps(data)[:200000]
        summary["mentions_zero_replay"] = "prompt_replay_count\": 0" in text or "\"replay\": 0" in text
        summary["mentions_zero_repack"] = "full_cache_repack_count\": 0" in text or "\"repack\": 0" in text
        summary["mentions_eos_stop"] = "last_token_is_eos\": true" in text or "finish_reason\": \"stop\"" in text
        out[str(path.relative_to(ROOT))] = summary
    return out


def identity_projection(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "ds41f_release_version": provenance.get("release_manifest", {}).get("release", {}).get("version"),
        "ds41f_runtime_source_sha256": provenance.get("ds41f", {}).get("runtime_source_identity", {}).get("sha256"),
        "ds41f_rust_boundary_sha256": provenance.get("ds41f", {}).get("rust_boundary_identity", {}).get("sha256"),
        "ds41f_release_packaging_sha256": provenance.get("ds41f", {}).get("release_packaging_identity", {}).get("sha256"),
        "omlx_revision": provenance.get("omlx", {}).get("revision"),
        "omlx_local_identity_sha256": provenance.get("omlx", {}).get("local_identity_sha256"),
        "omlx_decode_native_sha256": provenance.get("omlx", {}).get("decode_native_identity", {}).get("sha256"),
        "runtime_package_versions": provenance.get("packages"),
        "deepseek_recipe_revision": provenance.get("deepseek_recipe", {}).get("revision"),
        "deepseek_recipe_local_identity_sha256": provenance.get("deepseek_recipe", {}).get("local_identity_sha256"),
        "checkpoint_fingerprint": provenance.get("checkpoint_fingerprint"),
        "production": provenance.get("production"),
    }


def compare_identity_projection(expected: dict[str, Any], current: dict[str, Any], *, artifact: Path) -> dict[str, Any]:
    prev_proj = expected
    curr_proj = identity_projection(current)
    diffs = []
    for key in sorted(set(prev_proj) | set(curr_proj)):
        if prev_proj.get(key) != curr_proj.get(key):
            diffs.append({"key": key, "artifact": prev_proj.get(key), "current": curr_proj.get(key)})
    return {"status": "CURRENT_RUNTIME_MATCHES_ARTIFACT" if not diffs else "EXPENSIVE_QUALIFICATION_STALE", "differences": diffs, "artifact": str(artifact)}


def compare_artifact_identity(artifact: Path, current: dict[str, Any]) -> dict[str, Any]:
    previous = json.loads(artifact.read_text())
    prev_proj = previous.get("tested_runtime_identity") or identity_projection(previous.get("provenance", {}))
    return compare_identity_projection(prev_proj, current, artifact=artifact)


def compare_evidence_attestation(attestation: Path, current: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(attestation.read_text())
    expected = data.get("inherited_tested_runtime_identity") or data.get("tested_runtime_identity")
    if not expected:
        return {"status": "INVALID_ATTESTATION", "artifact": str(attestation), "differences": [{"key": "inherited_tested_runtime_identity", "artifact": None, "current": "required"}]}
    result = compare_identity_projection(expected, current, artifact=attestation)
    if result["status"] == "CURRENT_RUNTIME_MATCHES_ARTIFACT":
        result["status"] = "INHERITED_EXPENSIVE_EVIDENCE_VALID"
    return result


def derive_status(mode: str, provenance: dict[str, Any], gates: list[dict[str, Any]], real_model: list[dict[str, Any]]) -> str:
    if provenance.get("status") == "FAIL":
        return "FAILED"
    if any(g.get("status") == "FAIL" for g in gates + real_model):
        return "FAILED"
    if mode == "inspect":
        return "ENVIRONMENT_VALID" if provenance.get("status") == "PASS" else "ENVIRONMENT_CHANGED"
    if mode == "quick":
        return "QUICK_RUNTIME_QUALIFIED" if provenance.get("status") == "PASS" else "QUICK_RUNTIME_QUALIFIED_WITH_PROVENANCE_WARNING"
    if mode == "full":
        return "FULL_RELEASE_REQUALIFIED" if provenance.get("status") == "PASS" else "FULL_RELEASE_REQUALIFIED_WITH_PROVENANCE_WARNING"
    return "FAILED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run unified ds41f operational qualification")
    parser.add_argument("--mode", choices=("inspect", "quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, help="artifact output path")
    parser.add_argument("--skip-cheap-gates", action="store_true", help="only inspect provenance/config")
    parser.add_argument("--real-model", action="store_true", help="run representative real-model qualification in quick mode")
    parser.add_argument("--check-artifact", type=Path, help="compare current runtime identity with an existing qualification artifact and exit")
    parser.add_argument("--check-evidence", type=Path, help="compare current runtime identity with an expensive-evidence migration/attestation artifact and exit")
    args = parser.parse_args(argv)

    cfg = load_runtime_config()
    cfg.apply_environment()
    cfg.apply_import_paths()
    started = time.time()
    provenance = inspect_runtime(cfg)
    if args.check_artifact is not None:
        result = compare_artifact_identity(args.check_artifact, provenance)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "CURRENT_RUNTIME_MATCHES_ARTIFACT" else 2
    if args.check_evidence is not None:
        result = compare_evidence_attestation(args.check_evidence, provenance)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "INHERITED_EXPENSIVE_EVIDENCE_VALID" else 2

    cheap_results: list[dict[str, Any]] = []
    if args.mode in {"quick", "full"} and not args.skip_cheap_gates:
        for cmd in CHEAP_GATES:
            cheap_results.append(run_command(cmd, timeout=300))

    real_results: list[dict[str, Any]] = []
    run_real = args.mode == "full" or args.real_model
    if run_real:
        for cmd in FULL_REAL_MODEL_GATES:
            real_results.append(run_command(cmd, timeout=None))

    status = derive_status(args.mode, provenance, cheap_results, real_results)
    artifact = {
        "schema": "ds41f.release-qualification.v1",
        "created_at": time.time(),
        "mode": args.mode,
        "status": status,
        "duration_s": time.time() - started,
        "provenance": provenance,
        "tested_runtime_identity": identity_projection(provenance),
        "resolved_runtime_config": cfg.to_json(),
        "production_selectors": provenance.get("production"),
        "termination_config": {"deepseek_v41_eos_token_id": 1, "stateful_stop_strings": "rejected_before_mutation"},
        "gates": {"cheap_static": cheap_results, "real_model": real_results},
        "real_model_evidence_summary": summarize_existing_real_model_evidence(),
        "historical_long_context_evidence": "artifacts/m6/performance-qualification/result.json",
        "resource_policy": {"qualification_temp_artifacts": "retained under artifacts/release unless caller deletes them", "user_kv_artifacts": "retained when explicitly requested"},
        "invalidation_rules": {
            "expensive_real_model_qualification_stale_when": [
                "ds41f runtime_source_identity sha256 changes",
                "oMLX base revision or approved local identity sha256 changes",
                "deepseek-recipe base revision or approved local identity sha256 changes",
                "checkpoint fingerprint changes",
                "production selector or MTP/DSpark/speculation state changes"
            ],
            "rust_or_packaging_boundary_requalification_when": ["ds41f rust_boundary_identity sha256 changes", "release_packaging_identity sha256 changes"],
            "not_stale_when_only": ["generated artifacts change", "documentation changes", "unrelated non-runtime repository state changes"],
            "check_command": "python -m ds41f_mlx.qualify --check-artifact <artifact>",
            "inherited_evidence_check_command": "python -m ds41f_mlx.qualify --check-evidence <attestation>"
        },
        "outcome_semantics": {
            "ENVIRONMENT_VALID": "inspect-only config/provenance passed pinned checks",
            "QUICK_RUNTIME_QUALIFIED": "preflight and cheap/runtime gates passed; no fresh 200K claim",
            "FULL_RELEASE_REQUALIFIED": "quick gates plus configured real-model gates passed on this machine",
            "INHERITED_FULL_RELEASE_QUALIFIED": "current runtime identity matches migrated historical full-model evidence and current cheap/operator gates passed",
            "FAILED": "one or more required gates failed",
        },
    }
    out = artifact_path(args.mode, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "artifact": str(out), "provenance_status": provenance.get("status")}, indent=2))
    return 0 if status != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
