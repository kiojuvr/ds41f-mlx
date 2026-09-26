#!/usr/bin/env python3
"""Record M0 source and checkpoint identity without modifying the checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

OMLX = Path(os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
DS4 = Path(os.environ.get("DS41F_DS4", str(Path.home() / "ds4")))
ORACLE = Path(os.environ.get("DS41F_ORACLE", "/Volumes/SDXC-512/deepseek-v41-flash-mlx"))
CHECKPOINT = Path(os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
OUT = Path(os.environ.get("DS41F_M0_ARTIFACTS", "artifacts/m0"))
UPSTREAM_OMLX = "b390b31e0c6831225fed0f24d278eb1db7fcb68b"


def run(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr


def git_info(path: Path) -> dict:
    info: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    for key, args in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }.items():
        code, out, err = run(args, path)
        info[key] = out.strip() if code == 0 else {"error": err.strip(), "code": code}
    return info


def checkpoint_identity(path: Path) -> dict:
    info: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    for f in (config, index, path / "tokenizer.json", path / "tokenizer_config.json"):
        if f.exists():
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            info[f.name + "_sha256"] = h
            info[f.name + "_bytes"] = f.stat().st_size
    if index.exists():
        data = json.loads(index.read_text())
        weight_map = data.get("weight_map", {})
        info["metadata"] = data.get("metadata", {})
        info["tensor_count"] = len(weight_map)
        info["shard_count"] = len(set(weight_map.values()))
        info["shards"] = sorted(set(weight_map.values()))[:5] + (["..."] if len(set(weight_map.values())) > 5 else [])
    info["writable_by_process"] = os.access(path, os.W_OK)
    return info


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    diff_code, diff, diff_err = run(["git", "diff", "--binary"], OMLX)
    if diff_code != 0:
        diff = ""
    diff_path = OUT / "omlx-local-patch.diff"
    diff_path.write_text(diff)
    diff_sha = hashlib.sha256(diff.encode()).hexdigest()
    (OUT / "omlx-local-patch.sha256").write_text(diff_sha + "\n")

    source = {
        "schema": "ds41f.m0.source-identity.v1",
        "omlx_upstream_baseline": UPSTREAM_OMLX,
        "omlx_local_known_good": git_info(OMLX),
        "omlx_local_patch_diff": str(diff_path),
        "omlx_local_patch_sha256": diff_sha,
        "omlx_local_patch_note": "DeepSeek-V4.1 image-token parser fix is expected and recorded as local known-good patch.",
        "dwarfstar": git_info(DS4),
        "qualification_archive": git_info(ORACLE),
        "this_repo": git_info(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    (OUT / "source-identity.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")

    checkpoint = {
        "schema": "ds41f.m0.checkpoint-identity.v1",
        "checkpoint": checkpoint_identity(CHECKPOINT),
        "policy": "read-only source of truth; do not write generated files into checkpoint directory",
    }
    (OUT / "checkpoint-identity.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")

    print(f"wrote {OUT / 'source-identity.json'}")
    print(f"wrote {OUT / 'checkpoint-identity.json'}")
    if checkpoint["checkpoint"].get("writable_by_process"):
        print("warning: checkpoint directory appears writable by this process; runtime must still treat it as read-only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
