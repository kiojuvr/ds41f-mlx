"""One-command release acceptance for the current ds41f local deployment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.provenance import inspect_runtime
from ds41f_mlx.qualify import CHEAP_GATES, run_command, identity_projection
from ds41f_mlx.release import load_release_manifest

ROOT = Path(__file__).resolve().parents[1]


def artifact_path(output: Path | None) -> Path:
    if output is not None:
        return output
    return ROOT / "artifacts" / "release" / f"acceptance-{time.strftime('%Y%m%d-%H%M%S')}.json"


def run_cargo_acceptance(timeout: int | None = 1800) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("DS41F_PYTHON", sys.executable)
    started = time.time()
    proc = subprocess.run(["cargo", "run", "--bin", "m21_real_acceptance"], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {"command":["cargo","run","--bin","m21_real_acceptance"], "status":"PASS" if proc.returncode == 0 else "FAIL", "returncode":proc.returncode, "seconds":time.time()-started, "stdout_tail":proc.stdout[-8000:], "stderr_tail":proc.stderr[-8000:]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run canonical ds41f release acceptance")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--skip-cheap-gates", action="store_true", help="skip repository/native cheap gates; still run provenance and real Rust/server acceptance")
    args = ap.parse_args(argv)
    started = time.time()
    cfg = load_runtime_config(); cfg.apply_environment(); cfg.apply_import_paths()
    provenance = inspect_runtime(cfg)
    gates: list[dict[str, Any]] = []
    if not args.skip_cheap_gates:
        gates.extend(run_command(cmd, timeout=300) for cmd in CHEAP_GATES)
    real = run_cargo_acceptance()
    status = "PASS" if provenance.get("status") == "PASS" and all(g.get("status") == "PASS" for g in gates) and real.get("status") == "PASS" else "FAILED"
    artifact = {
        "schema":"ds41f.release-acceptance.v1",
        "created_at":time.time(),
        "duration_s":time.time()-started,
        "status":status,
        "release_manifest":load_release_manifest(),
        "provenance":provenance,
        "tested_runtime_identity":identity_projection(provenance),
        "gates":{"cheap":gates, "real_rust_http_server":real},
        "scope":"Current release acceptance: package/release identity, configured dependency provenance, Rust boundary build/test through cheap gates, real server startup/readiness/stateless/SSE-cancel/stateful/shutdown via m21_real_acceptance. Does not run long-context or historical exhaustive campaigns.",
    }
    out=artifact_path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(artifact, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status":status,"artifact":str(out),"provenance_status":provenance.get("status")}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
