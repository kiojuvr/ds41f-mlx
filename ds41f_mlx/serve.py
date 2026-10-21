"""Canonical launcher for the ds41f scoped text runtime."""
from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger('ds41f.operator.launcher')

# Production execution is MTP-only; historical OFF internals are not dispatched.


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
    parser.add_argument('--profile', choices=['mtp-singleton-v1','mtp-serving-v1'], default='mtp-serving-v1')
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

    parser.error('Standard-OFF production execution is retired')


if __name__ == "__main__":
    raise SystemExit(main())
