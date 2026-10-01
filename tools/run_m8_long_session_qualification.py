#!/usr/bin/env python3
"""Run M8 real-checkpoint repeated append/decode qualification.

This is intentionally an explicit target-hardware runner. It exercises the
production live-cache lifecycle; it does not persist KV and does not use a replay
fallback on failure.
"""
from __future__ import annotations

import argparse, json, subprocess, time
from pathlib import Path

from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession, PRODUCTION_PREFILL_SELECTOR
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession, M8ContinuationError


def parse_suffixes(text: str) -> list[list[int]]:
    turns = []
    for part in text.split(';'):
        part = part.strip()
        if not part:
            continue
        turns.append([int(x) for x in part.replace(',', ' ').split()])
    return turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument('--omlx-path', type=Path, default=DEFAULT_OMLX)
    ap.add_argument('--out', type=Path, default=Path('artifacts/m8/long-session-qualification.json'))
    ap.add_argument('--kv-dir', type=Path, default=Path('/Volumes/USB-SSD-RAID-0/ds41f-mlx/m8-kv'))
    ap.add_argument('--initial-prefix', default='0 3')
    ap.add_argument('--initial-terminal', type=int, default=15)
    ap.add_argument('--suffixes', default='16 17;18 19 20;21;22 23 24 25')
    ap.add_argument('--decode-tokens-per-turn', type=int, default=2)
    ap.add_argument('--cancel-after-first-token', action='store_true')
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.kv_dir.mkdir(parents=True, exist_ok=True)
    initial_prefix = [int(x) for x in args.initial_prefix.replace(',', ' ').split()]
    suffixes = parse_suffixes(args.suffixes)
    result = {
        'schema': 'ds41f.m8.long-session-qualification.v1',
        'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint': str(args.checkpoint),
        'omlx_path': str(args.omlx_path),
        'kv_dir_policy': str(args.kv_dir),
        'production_prefill_selector': PRODUCTION_PREFILL_SELECTOR,
        'initial_prefix_len': len(initial_prefix),
        'turn_suffix_lengths': [len(s) for s in suffixes],
        'started_at': time.time(),
        'turns': [],
        'passed': False,
    }
    try:
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
        model, _ = rt.load_model()
        prefill = DwarfStarMLXPrefillSession(model, omlx_path=args.omlx_path)
        pre = prefill.prefill(initial_prefix)
        cfg = OMLXDecodeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
        gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=args.initial_terminal, config=cfg, max_tokens=args.decode_tokens_per_turn)
        sess = M8LiveContinuationSession(model=model, live_cache=[], token_history=gen.current_token_history(), config=cfg)
        sess.generation = gen
        for i in range(args.decode_tokens_per_turn):
            rep = sess.next_token()
            if rep is None:
                break
            if args.cancel_after_first_token:
                break
        sess.ensure_idle('initial_turn_boundary')
        for turn_index, suffix in enumerate(suffixes):
            t0 = time.perf_counter()
            sess.begin_turn_from_suffix(suffix, max_tokens=args.decode_tokens_per_turn)
            generated = []
            for i in range(args.decode_tokens_per_turn):
                rep = sess.next_token()
                if rep is None:
                    break
                generated.append(int(rep.token))
                if args.cancel_after_first_token and i == 0:
                    break
            if args.cancel_after_first_token:
                sess.cancel_turn('qualification_cancel')
            else:
                sess.ensure_idle('qualification_turn_boundary')
            diag = sess.diagnostics()
            result['turns'].append({'turn_index': turn_index, 'suffix': suffix, 'generated': generated, 'seconds': time.perf_counter() - t0, 'diagnostics': diag})
            if diag['total_prompt_replay_count'] != 0 or diag['total_full_cache_repack_count'] != 0 or not diag['all_cache_offsets_equal_frontier']:
                raise M8ContinuationError('M8 diagnostic invariant failed')
        result['final_diagnostics'] = sess.diagnostics()
        result['passed'] = True
        sess.close(); rt.close()
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        result['finished_at'] = time.time()
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
