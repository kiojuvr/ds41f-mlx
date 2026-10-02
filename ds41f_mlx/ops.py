"""Canonical operator CLI for the ds41f release."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.provenance import inspect_runtime
from ds41f_mlx.release import load_release_manifest

ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str, args: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", module, *args], cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ds41f release operations")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect", help="inspect release manifest, configuration and provenance")
    sub.add_parser("start", help="start the local runtime server (delegates to ds41f_mlx.serve)")
    sub.add_parser("quick", help="run quick operational qualification")
    sub.add_parser("accept", help="run one-command release acceptance")
    sub.add_parser("full", help="run explicit full qualification")
    args, rest = parser.parse_known_args(argv)

    if args.cmd == "inspect":
        cfg = load_runtime_config(); cfg.apply_environment(); cfg.apply_import_paths()
        provenance = inspect_runtime(cfg)
        print(json.dumps({"schema":"ds41f.ops.inspect.v1", "status": provenance.get("status"), "release_manifest": load_release_manifest(), "provenance": provenance}, indent=2, sort_keys=True))
        return 0 if provenance.get("status") in {"PASS", "WARNING"} else 2
    if args.cmd == "start":
        return _run_module("ds41f_mlx.serve", rest)
    if args.cmd == "quick":
        return _run_module("ds41f_mlx.qualify", ["--mode", "quick", *rest])
    if args.cmd == "accept":
        return _run_module("ds41f_mlx.release_acceptance", rest)
    if args.cmd == "full":
        return _run_module("ds41f_mlx.qualify", ["--mode", "full", *rest])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
