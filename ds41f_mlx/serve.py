"""Canonical launcher for the ds41f scoped text runtime."""
from __future__ import annotations

import argparse
import json
import sys

from ds41f_mlx.config import load_runtime_config, validate_runtime_config
from ds41f_mlx.provenance import inspect_runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch ds41f local text runtime")
    parser.add_argument("--host", help="override DS41F_HOST")
    parser.add_argument("--port", type=int, help="override DS41F_PORT")
    parser.add_argument("--print-config", action="store_true", help="print resolved config/provenance and exit")
    parser.add_argument("--no-validate", action="store_true", help="skip path validation before launch")
    args = parser.parse_args(argv)

    cfg = load_runtime_config()
    if args.host is not None or args.port is not None:
        from dataclasses import replace
        cfg = replace(cfg, host=args.host or cfg.host, port=args.port or cfg.port)
    cfg.apply_environment()
    cfg.apply_import_paths()

    report = inspect_runtime(cfg)
    if args.print_config:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] in {"PASS", "WARNING"} else 2
    if not args.no_validate:
        failures = [f for f in validate_runtime_config(cfg) if f["status"] == "FAIL"]
        if failures:
            print(json.dumps({"status": "FAIL", "failures": failures, "config": cfg.to_json()}, indent=2), file=sys.stderr)
            return 2

    print(json.dumps({"status": "STARTING", "config": cfg.to_json(), "provenance_status": report["status"]}, indent=2), flush=True)
    try:
        import uvicorn
    except Exception as exc:
        print(f"uvicorn is required to launch the server: {exc}", file=sys.stderr)
        return 2

    # Import after cfg.apply_import_paths() so deepseek-recipe/oMLX checkouts do not
    # require undocumented PYTHONPATH state.
    from ds41f_mlx.serving.server import create_app

    app = create_app(model_id=cfg.model_id, recipe_path=cfg.recipe_path)
    uvicorn.run(app, host=cfg.host, port=cfg.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
