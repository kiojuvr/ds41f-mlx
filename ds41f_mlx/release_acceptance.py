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


def run_rust_acceptance(timeout: int | None = 1800, *, installed: bool = False) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("DS41F_PYTHON", sys.executable)
    started = time.time()
    bundled = Path(env.get("DS41F_ACCEPTANCE_BIN", str(ROOT / "bin" / "m21_real_acceptance")))
    if installed or bundled.exists():
        cmd = [str(bundled)]
        cwd = ROOT
    else:
        cmd = ["cargo", "run", "--bin", "m21_real_acceptance"]
        cwd = ROOT
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {"command":cmd, "status":"PASS" if proc.returncode == 0 else "FAIL", "returncode":proc.returncode, "seconds":time.time()-started, "stdout_tail":proc.stdout[-8000:], "stderr_tail":proc.stderr[-8000:]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run canonical ds41f release acceptance")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--skip-cheap-gates", action="store_true", help="skip repository/native cheap gates; still run provenance and real Rust/server acceptance")
    ap.add_argument("--installed", action="store_true", help="installed/bundled release mode: skip source-tree cheap gates and use bundled Rust acceptance binary")
    args = ap.parse_args(argv)
    started = time.time()
    cfg = load_runtime_config(); cfg.apply_environment(); cfg.apply_import_paths()
    provenance = inspect_runtime(cfg)
    gates: list[dict[str, Any]] = []
    if not args.skip_cheap_gates and not args.installed:
        commands = CHEAP_GATES
        if (ROOT/'release/promotion.json').exists():
            commands = [[sys.executable, '-m', 'ds41f_mlx.projection'], ['cargo', 'test', '--locked']]
        gates.extend(run_command(cmd, timeout=300) for cmd in commands)
    real = run_rust_acceptance(installed=args.installed)
    status = "PASS" if provenance.get("status") == "PASS" and all(g.get("status") == "PASS" for g in gates) and real.get("status") == "PASS" else "FAILED"
    artifact = {
        "schema":"ds41f.release-acceptance.v1",
        "created_at":time.time(),
        "duration_s":time.time()-started,
        "status":status,
        "release_manifest":load_release_manifest(),
        "provenance":provenance,
        "tested_runtime_identity":identity_projection(provenance),
        "gates":{"cheap":gates, "real_rust_http_server":real, "installed_mode": args.installed},
        "scope":"Current release acceptance: package/release identity, configured dependency provenance, Rust boundary build/test through cheap gates, real server startup/readiness/stateless/SSE-cancel/stateful/shutdown via m21_real_acceptance. Does not run long-context or historical exhaustive campaigns.",
    }
    out=artifact_path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(artifact, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status":status,"artifact":str(out),"provenance_status":provenance.get("status")}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
