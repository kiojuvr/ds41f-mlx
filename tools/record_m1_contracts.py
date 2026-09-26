#!/usr/bin/env python3
"""Reconstruct M1 provenance/API/oMLX-compatibility contracts.

The bounded direct/server comparison is an oMLX compatibility artifact, not an
official DeepSeek model-correctness oracle.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("DS41F_M1_ARTIFACTS", "artifacts/m1"))
CHECKPOINT = Path(os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
ORACLE = Path(os.environ.get("DS41F_ORACLE", "/Volumes/SDXC-512/deepseek-v41-flash-mlx"))
OMLX = Path(os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
MODEL_SETTINGS = Path(os.environ.get("DS41F_MODEL_SETTINGS", str(Path.home() / ".omlx" / "model_settings.json")))
UPSTREAM_OMLX = "b390b31e0c6831225fed0f24d278eb1db7fcb68b"

EXPECTED = {
    "config.json_sha256": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "model.safetensors.index.json_sha256": "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
    "tokenizer.json_sha256": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    "tokenizer_config.json_sha256": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    "tensor_count": 96085,
    "shard_count": 48,
    "total_size": 510286023000,
}


def run(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {"code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def git_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    for key, args in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }.items():
        r = run(args, path)
        info[key] = r["stdout"] if r["code"] == 0 else {"code": r["code"], "error": r["stderr"]}
    return info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def checkpoint_identity() -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(CHECKPOINT), "exists": CHECKPOINT.exists()}
    if not CHECKPOINT.exists():
        return info
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        f = CHECKPOINT / name
        if f.exists():
            info[f"{name}_sha256"] = sha256_file(f)
            info[f"{name}_bytes"] = f.stat().st_size
    index = CHECKPOINT / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        weight_map = data.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        info["tensor_count"] = len(weight_map)
        info["shard_count"] = len(shards)
        info["total_size"] = data.get("metadata", {}).get("total_size")
        info["shard_sample"] = shards[:5] + (["..."] if len(shards) > 5 else [])
    info["writable_by_process"] = os.access(CHECKPOINT, os.W_OK)
    return info


def oracle_checkpoint_summary() -> dict[str, Any]:
    summary = read_json(ORACLE / "artifacts/checkpoint/summary.json") or {}
    verification = read_json(ORACLE / "artifacts/checkpoint/verification.json") or {}
    run_prov = read_json(ORACLE / "artifacts/checkpoint/run-provenance.json") or {}
    files = {f.get("path"): f for f in verification.get("files", []) if isinstance(f, dict)}
    return {
        "oracle_path": str(ORACLE),
        "oracle_git": git_info(ORACLE),
        "checkpoint_artifacts": {
            "summary_json": str(ORACLE / "artifacts/checkpoint/summary.json"),
            "verification_json": str(ORACLE / "artifacts/checkpoint/verification.json"),
            "run_provenance_json": str(ORACLE / "artifacts/checkpoint/run-provenance.json"),
        },
        "hf_revision": (files.get("config.json") or {}).get("revision"),
        "selected_file_sha256": {k: (files.get(k) or {}).get("sha256") for k in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")},
        "by_component": summary.get("by_component"),
        "totals": summary.get("totals") or {k: summary.get(k) for k in ("tensor_count", "storage_bytes", "logical_parameters") if k in summary},
        "run_provenance_result_sha256": run_prov.get("result_sha256"),
        "run_provenance_base_commit": run_prov.get("base_commit"),
    }


def model_settings_identity() -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(MODEL_SETTINGS), "exists": MODEL_SETTINGS.exists()}
    if not MODEL_SETTINGS.exists():
        return info
    raw = MODEL_SETTINGS.read_bytes()
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    info["bytes"] = len(raw)
    try:
        data = json.loads(raw)
    except Exception as exc:
        info["parse_error"] = str(exc)
        return info
    # oMLX stores either {"version":1,"models":{model: settings}}, directly by model,
    # or as a flat temporary settings file depending on runner.
    if isinstance(data, dict) and isinstance(data.get("models"), dict):
        selected = data["models"].get("DeepSeek-V4.1-Flash", {})
    elif isinstance(data, dict):
        selected = data.get("DeepSeek-V4.1-Flash", data)
    else:
        selected = {}
    info["selected_settings"] = {k: selected.get(k) for k in (
        "deepseek_v41_engram_ssd_offload",
        "max_context_window",
        "mtp_enabled",
        "vlm_mtp_enabled",
        "moe_expert_offload_enabled",
        "dflash_enabled",
        "temperature",
        "top_p",
        "max_tokens",
    ) if isinstance(selected, dict) and k in selected}
    info["mtp_enabled"] = bool(info["selected_settings"].get("mtp_enabled")) if isinstance(info.get("selected_settings"), dict) else False
    return info


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_identity()
    oracle = oracle_checkpoint_summary()
    checks = []
    for key, expected in EXPECTED.items():
        actual = checkpoint.get(key)
        checks.append({"name": key, "expected": expected, "actual": actual, "passed": actual == expected})
    selected = oracle.get("selected_file_sha256", {}) or {}
    for local_key, oracle_name in {
        "config.json_sha256": "config.json",
        "model.safetensors.index.json_sha256": "model.safetensors.index.json",
        "tokenizer.json_sha256": "tokenizer.json",
        "tokenizer_config.json_sha256": "tokenizer_config.json",
    }.items():
        if selected.get(oracle_name):
            checks.append({"name": f"oracle_{oracle_name}_sha256", "expected": selected[oracle_name], "actual": checkpoint.get(local_key), "passed": selected[oracle_name] == checkpoint.get(local_key)})
    checks.append({"name": "checkpoint_policy_read_only", "passed": True, "note": "Process writability is recorded separately; project policy forbids writes regardless of filesystem mode.", "writable_by_process": checkpoint.get("writable_by_process")})

    checkpoint_record = {
        "schema": "ds41f.m1.checkpoint-provenance.v1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "official checkpoint is read-only source of truth; no generated files, caches, conversions, or repair outputs may be written into it",
        "checkpoint": checkpoint,
        "expected": EXPECTED,
        "oracle": oracle,
        "checks": checks,
        "passed": all(c.get("passed") for c in checks),
    }
    write_json(OUT / "checkpoint-provenance.json", checkpoint_record)

    patch = ROOT / "artifacts/m0/omlx-local-patch.sha256"
    runtime = {
        "schema": "ds41f.m1.runtime-identity.v1",
        "recorded_at": checkpoint_record["recorded_at"],
        "this_repo": git_info(ROOT),
        "omlx_upstream_baseline": UPSTREAM_OMLX,
        "omlx_local_known_good": git_info(OMLX),
        "omlx_local_patch_sha256": patch.read_text().strip() if patch.exists() else None,
        "historical_qualification_archive": git_info(ORACLE),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model_settings": model_settings_identity(),
        "policy_notes": [
            "oMLX b390b31e remains runtime/execution baseline.",
            "MTP/embedded DSpark setting identity is required for known-good decode-path comparisons.",
            "The current API server is a thin external boundary, not final production architecture.",
        ],
    }
    runtime["checks"] = [
        {"name": "omlx_head_is_pinned", "passed": runtime["omlx_local_known_good"].get("head") == UPSTREAM_OMLX, "actual": runtime["omlx_local_known_good"].get("head"), "expected": UPSTREAM_OMLX},
        {"name": "omlx_local_patch_recorded", "passed": bool(runtime.get("omlx_local_patch_sha256"))},
        {"name": "model_settings_recorded", "passed": runtime["model_settings"].get("exists") is True},
        {"name": "mtp_enabled_recorded", "passed": runtime["model_settings"].get("mtp_enabled") is True},
    ]
    runtime["passed"] = all(c["passed"] for c in runtime["checks"])
    write_json(OUT / "runtime-identity.json", runtime)

    m0_compat = ROOT / "artifacts/m0/oracle-compare/direct-vs-server.json"
    m0_compat_data = read_json(m0_compat) or {}
    api_runs = sorted((OUT / "api-atomicity").glob("run-*/result.json")) if (OUT / "api-atomicity").exists() else []
    latest_api = api_runs[-1] if api_runs else None
    latest_api_data = read_json(latest_api) if latest_api else None
    bounded_checks = [
        {"name": "checkpoint_provenance", "passed": checkpoint_record["passed"], "artifact": str(OUT / "checkpoint-provenance.json")},
        {"name": "runtime_identity", "passed": runtime["passed"], "artifact": str(OUT / "runtime-identity.json")},
        {"name": "m0_bounded_direct_server_omlx_compatibility_available", "passed": m0_compat_data.get("passed") is True, "artifact": str(m0_compat), "classification": "not_official_qualification"},
        {"name": "api_invalid_request_atomicity", "passed": (latest_api_data or {}).get("passed") is True, "artifact": str(latest_api) if latest_api else None},
    ]
    summary = {
        "schema": "ds41f.m1.contract-reconstruction-summary.v2",
        "passed": all(c["passed"] for c in bounded_checks),
        "artifacts": {
            "checkpoint_provenance": str(OUT / "checkpoint-provenance.json"),
            "runtime_identity": str(OUT / "runtime-identity.json"),
            "bounded_direct_server_omlx_compatibility": str(m0_compat),
            "api_invalid_request_atomicity": str(latest_api) if latest_api else None,
        },
        "bounded_checks": bounded_checks,
        "classification_notes": ["bounded direct/server comparison is oMLX compatibility only", "checkpoint provenance and API atomicity remain independently valid"],
        "non_goals": ["no official model correctness claim", "no long-context qualification", "no DwarfStar kernel migration", "no server redesign", "no checkpoint rewrite or quantization"],
    }
    write_json(OUT / "summary.json", summary)
    print(OUT)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
