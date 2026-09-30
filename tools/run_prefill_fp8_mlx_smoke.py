#!/usr/bin/env python3
"""Bounded structural smoke of the command executor (no serving or timing)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ds41f_mlx.prefill_fp8_mlx import SweepPlanner, DwarfStarFP8MLXPrefillExecutorSetup
from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=int, choices=(2048, 8192), required=True)
    p.add_argument('--mtp-off', action='store_true', required=True)
    p.add_argument('--no-benchmark', action='store_true', required=True)
    p.add_argument('--exercise-suffix', action='store_true')
    p.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
    return p


def geometry(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype)} if value is not None else None


def main(argv=None):
    args = parser().parse_args(argv)
    if args.exercise_suffix != (args.tokens == 8192):
        parser().error('8192 requires --exercise-suffix; 2048 is the short smoke')
    try:
        import mlx.core as mx
        from omlx.patches.deepseek_v41.loading import load
    except ImportError as exc:
        print(json.dumps({'status': 'NOT RUN', 'reason': str(exc)}))
        return 2
    if not (args.checkpoint / 'config.json').is_file():
        print(json.dumps({'status': 'NOT RUN', 'reason': 'checkpoint unavailable', 'checkpoint': str(args.checkpoint)}))
        return 2
    model = None
    try:
        # Match the reviewed target runtime's SSD-backed Engram configuration.
        # Resident tables exhaust Metal memory before this smoke can execute.
        model, _processor = load(args.checkpoint, preserve_mtp=False, engram_ssd_offload=True)
        lm = model.language_model
        if len(lm.layers) != 40 or lm._config.compress_ratios[20] != 1:
            raise RuntimeError('expected DeepSeek-V4.1 Flash, 40 layers, layer20 ratio=1')
        with tempfile.TemporaryDirectory(prefix='ds41f-smoke-') as tmp:
            planner = SweepPlanner(load_native_prefill_library(compile_native_prefill_library(Path(tmp))))
            plan = planner.build_from_model(lm, ctx=32768, remaining=args.tokens)
        if plan.count != args.tokens or plan.decoder_suffix != args.exercise_suffix:
            raise RuntimeError('unexpected planner geometry')
        # Deterministic valid text token IDs; no generation or output-head benchmark.
        token_ids = [1] * args.tokens
        setup = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=mx).prepare(plan, token_ids)
        runner, arena = setup.block_runner, setup.arena
        for command in plan.commands:
            runner.execute_command(command, arena)
            # Explicit bounded materialization, no independent transformer loop.
            values = [arena.carry.current.value, arena.carry.next.value, arena.carry.pre.value]
            values.extend(value for cache in runner.working_cache for value in cache.cache if value is not None)
            mx.eval(*[v for v in values if v is not None])
        cache = runner.working_cache
        frontiers = [int(c[0].item()) for c in cache]
        assert frontiers == [args.tokens] * 40
        assert int(cache[20][2].shape[1]) == args.tokens
        assert int(cache[20][3].shape[1]) == args.tokens
        assert all(int(c[1].shape[1]) == lm._config.window_size for c in cache)
        assert runner.suffix_math.full_source_generation_count == int(args.exercise_suffix)
        assert not runner.prefill_continuation_exported and runner.full_cache_repack_count == 0
        print(json.dumps({
            'status': 'PASS', 'structural_only': True,
            'cache_frontiers': frontiers,
            'layer20_cache2': geometry(cache[20][2]),
            'layer20_cache3': geometry(cache[20][3]),
            'layer20_compressor_pending': [geometry(cache[20][i]) for i in (4, 5)],
            'window_cache_geometry': [geometry(c[1]) for c in cache],
            'full_source_generation_count': runner.suffix_math.full_source_generation_count,
            'PrefillContinuationState_export_count': int(runner.prefill_continuation_exported),
            'full_cache_repack_count': runner.full_cache_repack_count,
        }, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'FAIL', 'reason': f'{type(exc).__name__}: {exc}'}))
        return 1
    finally:
        if model is not None:
            model.close()


if __name__ == '__main__':
    raise SystemExit(main())
