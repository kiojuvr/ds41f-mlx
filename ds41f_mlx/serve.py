"""Canonical launcher for the ds41f scoped text runtime."""
from __future__ import annotations

import argparse
import json
import sys

from ds41f_mlx.config import load_runtime_config, validate_runtime_config
from ds41f_mlx.provenance import inspect_runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch ds41f local text runtime")
    parser.add_argument('--profile', choices=['standard-off','mtp-singleton-v1'], default='standard-off')
    parser.add_argument("--host", help="override DS41F_HOST")
    parser.add_argument("--port", type=int, help="override DS41F_PORT")
    parser.add_argument("--print-config", action="store_true", help="print resolved config/provenance and exit")
    parser.add_argument("--no-validate", action="store_true", help="skip path validation before launch")
    args = parser.parse_args(argv)

    if args.profile == 'mtp-singleton-v1':
        if args.no_validate:
            parser.error('MTP identity validation cannot be disabled')
        try:
            from .mtp_identity import config, inspect
            cfg = config(args.host, args.port)
            report = inspect(cfg)
        except (ValueError, OSError, ImportError) as exc:
            print(json.dumps({'status':'FAIL','error':str(exc)}), file=sys.stderr)
            return 2
        if args.print_config:
            print(json.dumps(report,indent=2))
            return 0
        cfg.apply_environment()
        from .serving.mtp_public import LocalMTPBackend
        from .serving.server import create_app
        backend = LocalMTPBackend(runtime_config=cfg)
        backend.dependency_identity = report['identity_sha256']
        app = create_app(backend=backend, runtime_config=cfg, profile=args.profile)
        import uvicorn
        from .serving.local_h11 import LocalH11Protocol
        uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1, proxy_headers=False, ws='none', loop='asyncio',
                    limit_concurrency=8, backlog=8, timeout_keep_alive=5,
                    h11_max_incomplete_event_size=16384, http=LocalH11Protocol)
        # Shutdown does not claim persistence or recovery. Await shielded retirement.
        import asyncio
        async def retire():
            for sid in list(backend.sessions):
                await backend.close_stateful_session(sid)
        asyncio.run(retire())
        backend.close()
        return 0

    cfg = load_runtime_config()
    if args.host is not None or args.port is not None:
        from dataclasses import replace
        cfg = replace(cfg, host=args.host or cfg.host, port=args.port or cfg.port)
    cfg.apply_environment()
    cfg.apply_import_paths()

    report = inspect_runtime(cfg)
    from .delivery import RECORD
    if RECORD.exists() and (args.no_validate or report['status'] != 'PASS'):
        print(json.dumps({'status': 'FAIL', 'error': 'source-delivered OFF provenance cannot be bypassed', 'provenance': report}), file=sys.stderr)
        return 2
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
