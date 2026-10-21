"""Canonical operator CLI for the MTP-only ds41f Runtime."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        from .operator_tui import main as tui_main
        return tui_main()
    parser = argparse.ArgumentParser(description='ds41f MTP operations; Standard-OFF is retired')
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('inspect', help='inspect installed execution and exact checkpoint identity')
    sub.add_parser('start', help='start the installed MTP runtime')
    sub.add_parser('accept', help='run bounded installed ordinary MTP acceptance')
    for command in sub.choices.values():
        command.add_argument('--profile', choices=['mtp-singleton-v1', 'mtp-serving-v1'],
                             default='mtp-serving-v1')
    args, rest = parser.parse_known_args(argv)
    if args.cmd == 'start':
        os.execv(sys.executable, [sys.executable, '-I', '-m', 'ds41f_mlx.serve', '--profile', args.profile, *rest])
    if args.cmd == 'accept':
        if args.profile != 'mtp-serving-v1':
            parser.error('use ordinary installed acceptance; singleton evidence is historical')
        return subprocess.call([sys.executable, '-I', '-m', 'ds41f_mlx.production_acceptance', *rest])
    try:
        from .mtp_identity import config, inspect
        cfg = config(profile=args.profile)
        report = inspect(cfg)
        report['profile'] = report['serving_profile'] = args.profile
        if args.profile == 'mtp-serving-v1':
            from .serving.capacity import context_envelope, ORDINARY_BODY_BYTES, ORDINARY_OUTPUT_CEILING
            report['serving_limits'] = dict(context_tokens=context_envelope(cfg.checkpoint_path),
                                            output_tokens=ORDINARY_OUTPUT_CEILING, body_bytes=ORDINARY_BODY_BYTES)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(json.dumps({'status': 'FAIL', 'profile': args.profile, 'error': str(exc)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
