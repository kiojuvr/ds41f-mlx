#!/usr/bin/env python3
"""Probe the Python environment needed by the M0 oMLX bridge.

This does not load the model. It verifies imports and records function
signatures so the heavy smoke can be run only after the execution environment is
known.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("DS41F_OMLX_ENV_OUT", "artifacts/m0/omlx-env")) / time.strftime("probe-%Y%m%d-%H%M%S.json")
PYTHON = os.environ.get("DS41F_PYTHON", sys.executable)
OMLX = os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2"))

PROBE = r'''
import importlib, inspect, json, os, platform, sys
sys.path.insert(0, os.environ["DS41F_OMLX"])
result = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "omlx_path": os.environ["DS41F_OMLX"],
    "imports": {},
}
for name in ["mlx", "mlx.core", "mlx_lm", "mlx_lm.generate", "transformers", "omlx.patches.deepseek_v41.loading", "omlx.patches.deepseek_v41.processing"]:
    try:
        mod = importlib.import_module(name)
        entry = {"ok": True, "file": getattr(mod, "__file__", None), "version": getattr(mod, "__version__", None)}
        if name == "mlx_lm.generate":
            for symbol in ["generate_step", "stream_generate", "generate", "BatchGenerator"]:
                obj = getattr(mod, symbol, None)
                if obj is not None:
                    try:
                        entry[symbol + "_signature"] = str(inspect.signature(obj))
                    except Exception as exc:
                        entry[symbol + "_signature_error"] = repr(exc)
        result["imports"][name] = entry
    except BaseException as exc:
        result["imports"][name] = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
print(json.dumps(result, sort_keys=True))
'''


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DS41F_OMLX"] = OMLX
    proc = subprocess.run([PYTHON, "-c", PROBE], env=env, text=True, capture_output=True)
    record = {
        "schema": "ds41f.m0.omlx-env-probe.v1",
        "probe_python": PYTHON,
        "probe_omlx": OMLX,
        "host_python": sys.version,
        "host_platform": platform.platform(),
        "returncode": proc.returncode,
        "stderr": proc.stderr,
    }
    if proc.stdout.strip():
        try:
            record["result"] = json.loads(proc.stdout)
        except Exception:
            record["stdout_raw"] = proc.stdout
    else:
        record["stdout_raw"] = proc.stdout
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(OUT)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
