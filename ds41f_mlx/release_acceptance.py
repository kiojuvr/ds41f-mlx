"""Historical release helpers; OFF production acceptance entry point is retired."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

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
    raise SystemExit('Historical OFF release acceptance is retired; use ds41f-accept --runtime-url URL --output PATH')


if __name__ == "__main__":
    raise SystemExit(main())
