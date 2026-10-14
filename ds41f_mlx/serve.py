"""Canonical launcher for the ds41f scoped text runtime."""
from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger('ds41f.operator.launcher')

from ds41f_mlx.config import load_runtime_config, validate_runtime_config


def main(argv: list[str] | None = None) -> int:
    # Acquire before dependency imports/model loading; lifetime is kernel-owned.
    argv = sys.argv[1:] if argv is None else argv
    if '--print-config' in argv or '--help' in argv or '-h' in argv:
        return _main(argv)
    from .observability import Projection
    from .operator_control import Control
    from .runtime_logging import History, install
    history = History()
    log_output = install(history, sys.stderr)
    projection = Projection()
    try:
        control = Control(projection, logs=history)
    except OSError:
        logger.error('CONFLICT: ds41f runtime/control socket already in use')
        log_output.drain()
        return 2
    try:
        logger.info('STARTING; runtime admission begins')
        result = _main(argv, control=control, projection=projection)
        if result:
            projection.lifecycle('FAILED')
        else:
            logger.info('shutdown complete; runtime retired')
        return result
    except Exception:
        projection.lifecycle('FAILED')
        # Arbitrary exception details/traceback remain headless only, once.
        logging.getLogger('ds41f.diagnostic').exception('runtime launcher failed')
        return 2
    finally:
        control.close()
        log_output.drain()


def _main(argv, *, control=None, projection=None) -> int:
    parser = argparse.ArgumentParser(description="Launch ds41f local text runtime")
    parser.add_argument('--profile', choices=['standard-off','mtp-singleton-v1','mtp-serving-v1'], default='mtp-serving-v1')
    from .config import REASONING_EFFORTS
    from dataclasses import replace
    parser.add_argument('--default-reasoning-effort', choices=REASONING_EFFORTS,
                        help='reasoning effort for requests without reasoning controls')
    parser.add_argument("--host", help="override DS41F_HOST")
    parser.add_argument("--port", type=int, help="override DS41F_PORT")
    parser.add_argument("--print-config", action="store_true", help="print resolved config/provenance and exit")
    parser.add_argument("--no-validate", action="store_true", help="skip path validation before launch")
    args = parser.parse_args(argv)

    if args.profile in ('mtp-singleton-v1', 'mtp-serving-v1'):
        if args.no_validate:
            parser.error('MTP identity validation cannot be disabled')
        try:
            from .mtp_identity import config, inspect
            cfg = config(args.host, args.port, profile=args.profile)
            if args.default_reasoning_effort is not None:
                cfg = replace(cfg, default_reasoning_effort=args.default_reasoning_effort)
            if args.profile == 'mtp-singleton-v1' and cfg.default_reasoning_effort != 'none':
                raise ValueError('singleton profile requires reasoning none')
            if control is not None:
                control.reserve_service(cfg.host, cfg.port)
            report = inspect(cfg)
            report['profile'] = report['serving_profile'] = args.profile
        except (ValueError, OSError, ImportError) as exc:
            logger.error('admission failed: %s', type(exc).__name__)
            logging.getLogger('ds41f.diagnostic').error('admission detail: %s', exc)
            return 2
        if args.print_config:
            print(json.dumps(report,indent=2))
            return 0
        logger.info('runtime admission passed profile=%s glm_profile=%s default_reasoning_effort=%s',
                    args.profile, report['glm_profile'], cfg.default_reasoning_effort)
        cfg.apply_environment()
        from .serving.mtp_public import LocalMTPBackend
        from .serving.server import create_app
        if args.profile == 'mtp-serving-v1':
            from .serving.production_mtp import ProductionMTPBackend
            backend = ProductionMTPBackend(runtime_config=cfg)
        else:
            backend = LocalMTPBackend(runtime_config=cfg)
        backend.dependency_identity = report['identity_sha256']
        backend.glm_profile = report['glm_profile']
        backend.telemetry = projection
        if projection is not None:
            projection.lifecycle('STARTING', dict(host=cfg.host, port=cfg.port, profile=args.profile,
                glm_profile=report['glm_profile'],
                context_tokens=backend.context_tokens if args.profile == 'mtp-serving-v1' else None))
        app = create_app(backend=backend, runtime_config=cfg, profile=args.profile)
        if args.profile == 'mtp-serving-v1':
            # Canonical startup owns loading; telemetry only projects it.
            async def load_model():
                projection.lifecycle('LOADING')
                try:
                    await backend._call(backend.load)
                except BaseException:
                    projection.lifecycle('FAILED')
                    raise
                if not control.shutdown_requested:
                    projection.lifecycle('READY')
            app.router.add_event_handler('startup', load_model)
        import uvicorn
        from .serving.local_h11 import LocalH11Protocol
        control.run(app, host=cfg.host, port=cfg.port, workers=1, proxy_headers=False, ws='none', loop='asyncio',
                    limit_concurrency=8, backlog=8, timeout_keep_alive=5,
                    h11_max_incomplete_event_size=16384, http=LocalH11Protocol)
        # Shutdown does not claim persistence or recovery. Await shielded retirement.
        import asyncio
        async def retire():
            if args.profile == 'mtp-serving-v1':
                await backend.shutdown_serving()
                return
            for sid in list(backend.sessions):
                await backend.close_stateful_session(sid)
        asyncio.run(retire())
        backend.close()
        return 0

    cfg = load_runtime_config()
    if args.default_reasoning_effort is not None:
        cfg = replace(cfg, default_reasoning_effort=args.default_reasoning_effort)
    if args.host is not None or args.port is not None:
        from dataclasses import replace
        cfg = replace(cfg, host=args.host or cfg.host, port=args.port or cfg.port)
    cfg.apply_environment()
    cfg.apply_import_paths()
    if control is not None:
        try:
            control.reserve_service(cfg.host, cfg.port)
        except OSError:
            logger.error('CONFLICT: service socket already in use')
            return 2

    from .delivery import inspect as inspect_installed_off
    if args.no_validate:
        parser.error('installed execution admission cannot be disabled')
    try:
        report = inspect_installed_off(cfg)
    except (ValueError, OSError, ImportError) as exc:
        logger.error('admission failed: %s', type(exc).__name__)
        logging.getLogger('ds41f.diagnostic').error('admission detail: %s', exc)
        return 2
    if args.print_config:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] in {"PASS", "WARNING"} else 2
    if not args.no_validate:
        failures = [f for f in validate_runtime_config(cfg) if f["status"] == "FAIL"]
        if failures:
            logger.error('runtime configuration validation failed')
            logging.getLogger('ds41f.diagnostic').error('%s', json.dumps({'failures': failures, 'config': cfg.to_json()}))
            return 2

    logger.info('STARTING; runtime admission passed profile=%s default_reasoning_effort=%s',
                args.profile, cfg.default_reasoning_effort)
    try:
        import uvicorn
    except Exception as exc:
        logger.error('uvicorn is required to launch the server')
        logging.getLogger('ds41f.diagnostic').error('%s', exc)
        return 2

    # Import after cfg.apply_import_paths() so deepseek-recipe/oMLX checkouts do not
    # require undocumented PYTHONPATH state.
    from ds41f_mlx.serving.server import create_app

    app = create_app(model_id=cfg.model_id, recipe_path=cfg.recipe_path, runtime_config=cfg)
    if projection is not None:
        projection.lifecycle('READY', dict(host=cfg.host, port=cfg.port, profile=args.profile))
    control.run(app, host=cfg.host, port=cfg.port, proxy_headers=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
