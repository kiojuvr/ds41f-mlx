"""Canonical operator CLI for the ds41f release."""
from __future__ import annotations

import argparse
import json
import os
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
    for command in sub.choices.values():
        command.add_argument('--profile', choices=['standard-off','mtp-singleton-v1'], default='standard-off')
    args, rest = parser.parse_known_args(argv)

    if args.profile == 'mtp-singleton-v1':
        if args.cmd == 'start':
            os.execv(sys.executable, [sys.executable, '-m', 'ds41f_mlx.serve', '--profile', args.profile, *rest])
        if args.cmd == 'accept':
            return _run_module('ds41f_mlx.mtp_acceptance', rest)
        if args.cmd != 'inspect':
            parser.error('MTP supports inspect/start/accept only')
        try:
            from .mtp_identity import config, inspect
            print(json.dumps(inspect(config()), indent=2, sort_keys=True))
            return 0
        except (ValueError, OSError, ImportError) as exc:
            print(json.dumps({'status':'FAIL','profile':args.profile,'error':str(exc)}))
            return 2

    if args.cmd == "inspect":
        cfg = load_runtime_config(); cfg.apply_environment(); cfg.apply_import_paths()
        provenance = inspect_runtime(cfg)
        print(json.dumps({"schema":"ds41f.ops.inspect.v1", "status": provenance.get("status"), "release_manifest": load_release_manifest(), "provenance": provenance}, indent=2, sort_keys=True))
        return 0 if provenance.get("status") in {"PASS", "WARNING"} else 2
    if args.cmd == "start":
        os.execv(sys.executable, [sys.executable, '-m', 'ds41f_mlx.serve', *rest])
    if args.cmd == "quick":
        return _run_module("ds41f_mlx.qualify", ["--mode", "quick", *rest])
    if args.cmd == "accept":
        return _run_module("ds41f_mlx.release_acceptance", rest)
    if args.cmd == "full":
        return _run_module("ds41f_mlx.qualify", ["--mode", "full", *rest])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
