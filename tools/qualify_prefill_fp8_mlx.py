#!/usr/bin/env python3
"""Bounded P3/P4 evidence only: 8192 same-input source and window checks.

Requires both real structural smokes to have passed before invocation. No
serving selection, timing, decoder handoff, or independent transformer loop.
Exactness here applies only to packed state produced by identical loaded
operators on identical input, not to whole-model backend trajectories.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import DwarfStarFP8MLXPrefillExecutorSetup, SweepPlanner, SweepCommandKind, SweepPhase
from tools.run_prefill_fp8_mlx_smoke import geometry


def runtime_authority(lang):
    path = Path(lang.__file__).resolve()
    def git(*args):
        result = subprocess.run(['git', '-C', str(path.parent), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    # Bind the adapter's keyword contract before loading hundreds of GiB.
    contracts = {
        'hc_mixes': ((None,) * 5, {}), 'hc_pre_norm': ((None,) * 4, {}),
        'hc_post': ((None,) * 4, {}), 'rope': ((None,) * 4, {'inverse': True}),
        'pack_activation': ((None,), {'bits': 4, 'group_size': 16, 'e4m3_scale': True}),
        'quantize_activation': ((None,), {'bits': 4}),
        'packed_index_topk': ((None,) * 6, {'block_count': 2, 'block_size': 16}),
        'packed_index_scores': ((None,) * 6, {}),
        'packed_sparse_attention': ((None,) * 7, {}),
    }
    checks = {}
    for name, (args, kwargs) in contracts.items():
        try:
            operation = getattr(lang, name)
            if not callable(operation):
                raise TypeError('not callable')
            inspect.signature(operation).bind(*args, **kwargs)
            checks[name] = 'compatible'
        except (AttributeError, TypeError, ValueError) as exc:
            checks[name] = str(exc)
    import omlx
    return {
        'omlx_version': getattr(omlx, '__version__', None),
        'distribution_version': importlib.metadata.version('omlx'),
        'mlx_version': importlib.metadata.version('mlx'),
        'python_executable': sys.executable,
        'language_path': str(path), 'language_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'git_revision': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
        'operation_contracts': checks,
    }


def qualify(args, report):
    import mlx.core as mx
    import omlx.patches.deepseek_v41.language as lang
    from omlx.patches.deepseek_v41.loading import load
    report['runtime_authority'] = runtime_authority(lang)
    incompatible = {k: v for k, v in report['runtime_authority']['operation_contracts'].items() if v != 'compatible'}
    if incompatible:
        raise RuntimeError(f'oMLX API drift: {incompatible}')
    print(json.dumps({'runtime_authority': report['runtime_authority']}), flush=True)
    report['stage'] = 'checkpoint_load'
    model = load(args.checkpoint, preserve_mtp=False, engram_ssd_offload=True)[0]
    try:
        lm = model.language_model
        report['stage'] = 'planner_and_setup'
        with tempfile.TemporaryDirectory(prefix='ds41f-qualification-') as tmp:
            planner = SweepPlanner(load_native_prefill_library(compile_native_prefill_library(Path(tmp))))
            plan = planner.build_from_model(lm, ctx=32768, remaining=8192)
        setup = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=mx).prepare(plan, [1] * 8192)
        arena, runner = setup.arena, setup.block_runner
        cache20 = runner.working_cache[20]
        attn20 = lm.layers[20].attn
        role = 'command'
        calls = {'command_compressor': [], 'reference_compressor': [], 'command_index_wk': [], 'reference_index_wk': [], 'ordinary_decoder_blocks': []}
        report['operation_calls'] = calls
        original_compressor = type(attn20.compressor).__call__
        original_wk = type(attn20.indexer.wk).__call__
        original_block = lang.Block.__call__
        def compressor(obj, x, cache, start):
            if obj is attn20.compressor:
                calls[role + '_compressor'].append({'start': start, 'rows': int(x.shape[1])})
            return original_compressor(obj, x, cache, start)
        def wk(obj, x, *a, **kw):
            if obj is attn20.indexer.wk:
                calls[role + '_index_wk'].append(int(x.shape[1]))
            return original_wk(obj, x, *a, **kw)
        def block(obj, *a, **kw):
            if obj.attn._layer >= 20:
                calls['ordinary_decoder_blocks'].append(obj.attn._layer)
                raise RuntimeError('ordinary Block fallback during suffix qualification')
            return original_block(obj, *a, **kw)
        def exact(actual, expected, boundary):
            report['stage'] = boundary
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(f'{boundary}: geometry/dtype mismatch')
            mismatches = int(mx.sum(actual != expected).item())
            if mismatches:
                report['first_divergence'] = {'boundary': boundary, 'mismatched_bytes': mismatches}
                raise RuntimeError(f'{boundary}: {mismatches} mismatched packed bytes')
            return {'shape': list(actual.shape), 'dtype': str(actual.dtype), 'mismatched_bytes': 0}
        window_checks = []
        report['window_checks'] = window_checks
        source_reference = None
        seen_layers = set()
        with ExitStack() as stack:
            stack.enter_context(patch.object(type(attn20.compressor), '__call__', compressor))
            stack.enter_context(patch.object(type(attn20.indexer.wk), '__call__', wk))
            stack.enter_context(patch.object(lang.Block, '__call__', block))
            for command in plan.commands:
                report['stage'] = f'command:{command.index}:{command.kind.value}:layer{command.layer}'
                full_prepare = command.kind is SweepCommandKind.DECODER_PREPARE_SUFFIX and arena.decoder_prepare_role(command) == 'decoder_full_source_publish'
                local_prepare = command.kind is SweepCommandKind.DECODER_PREPARE_SUFFIX and not full_prepare
                before_frontier = None
                if local_prepare:
                    layer = lm.layers[command.layer]
                    cache = runner.working_cache[command.layer]
                    before_frontier = int(cache[0].item()) if cache[0] is not None else 0
                    offset, rows = command.offset, command.rows
                    h = arena.carry.current.value[:, offset:offset+rows]
                    pre = arena.carry.pre.value[:, offset:offset+rows]
                    x = lang.hc_pre_norm(h, pre, layer.attn_norm.weight, layer.attn_norm.eps)
                    _, kv_input = layer.attn._input_projections(x)
                    expected_window = lang.pack_activation(lang.rope(layer.attn.kv_norm(kv_input), mx.arange(offset, offset+rows), lm._config, bool(lm._config.compress_ratios[command.layer])))
                    mx.eval(expected_window)
                    # Deliberately non-contiguous distinctive prior state. The full
                    # replacement must equal the independent current-row projection.
                    cache[1] = mx.full((1, 128, expected_window.shape[-1]), 255, dtype=mx.uint8)
                    mx.eval(cache[1])
                runner.execute_command(command, arena)
                mx.eval(arena.carry.current.value, arena.carry.next.value, arena.carry.pre.value,
                        *[v for cache in runner.working_cache for v in cache.cache if v is not None])
                if command.kind is SweepCommandKind.ENCODE_ROWS:
                    seen_layers.add(command.layer)
                if full_prepare:
                    if cache20[0] is not None and int(cache20[0].item()) != 0:
                        raise RuntimeError('full-source prepare advanced logical frontier')
                    pending = setup.publication_manager.pending_cumulative_by_layer[20]
                    report['full_prepare_has_row_spans'] = bool(setup.publication_manager.pending_spans_by_layer.get(20))
                    assert not report['full_prepare_has_row_spans']
                    assert setup.publication_manager.visible_cumulative['kv'].layer != 20
                    assert pending['kv'].value is cache20[2]
                    report['producer_private_before_frontier'] = True
                    # Existing ordinary Attention is the reference source path. Its
                    # full query output is not evaluated or used as a decoder oracle.
                    role = 'reference'
                    reference_cache = lm.make_cache()[20]
                    x = lang.hc_pre_norm(arena.encoder_final_h, arena.encoder_final_pre, lm.layers[20].attn_norm.weight, lm.layers[20].attn_norm.eps)
                    reference_shared = {}
                    unused_output = attn20(x, reference_cache, reference_shared, 0)
                    mx.eval(reference_cache[2], reference_cache[3])
                    source_reference = reference_cache[2], reference_cache[3]
                    del unused_output, reference_shared, reference_cache, x
                    role = 'command'
                    report['source_comparison'] = {
                        'reference': 'loaded ordinary Attention.__call__, same full encoder-final pre-norm input, start=0',
                        'compressed_kv': exact(cache20[2], source_reference[0], 'source_compressor'),
                        'index_k': exact(cache20[3], source_reference[1], 'index_k'),
                    }
                if local_prepare:
                    after_frontier = int(cache[0].item()) if cache[0] is not None else 0
                    assert after_frontier == before_frontier
                    result = exact(cache[1], expected_window, f'local_window_preparation:layer{command.layer}')
                    window_checks.append({'layer': command.layer, 'offset': command.offset, 'rows': command.rows, 'stale_seed': 255, 'frontier_unchanged': True, **result})
                    del expected_window, h, pre, x, kv_input
                if command.kind is SweepCommandKind.ENCODE_ROWS and command.layer == 20:
                    exact(cache20[2], source_reference[0], 'layer20_query_preserves_compressed_kv')
                    exact(cache20[3], source_reference[1], 'layer20_query_preserves_index_k')
        report['stage'] = 'final_invariants'
        frontiers = [int(c[0].item()) for c in runner.working_cache]
        assert frontiers == [8192] * 40 and seen_layers == set(range(40))
        assert calls['command_compressor'] == [{'start': 0, 'rows': 8192}]
        assert calls['command_index_wk'] == [8192]
        assert calls['ordinary_decoder_blocks'] == []
        assert runner.suffix_math.full_source_generation_count == 1
        assert not runner.prefill_continuation_exported and runner.full_cache_repack_count == 0
        assert runner.final_logits_suppressed and runner.final_logits is None
        assert all(int(c[1].shape[1]) == 128 for c in runner.working_cache)
        assert all(c.shape == (1, 0, 512) and c.dtype == mx.bfloat16 for c in cache20.cache[4:6])
        first_queries = {}
        for layer, start, old_length in runner.suffix_math.query_old_lengths:
            first_queries.setdefault(layer, {'absolute_start': start, 'old_length': old_length})
        assert len(first_queries) == 20 and all(v['old_length'] == 127 for v in first_queries.values())
        report.update(status='PASS', cache_frontiers=frontiers,
                      layer20_slots=[geometry(v) for v in cache20.cache],
                      all_cache_slots=[[geometry(v) for v in c.cache] for c in runner.working_cache],
                      first_suffix_queries=first_queries,
                      full_source_generation_count=runner.suffix_math.full_source_generation_count,
                      PrefillContinuationState_export_count=0, full_cache_repack_count=0,
                      final_prefix_logits_suppressed=True,
                      executed_commands=len(runner.records), executed_layers=sorted(seen_layers))
    finally:
        model.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = {'status': 'FAIL', 'structural_and_bounded_same_input_only': True,
              'checkpoint': str(args.checkpoint), 'tokens': 8192,
              'load_kwargs': {'preserve_mtp': False, 'engram_ssd_offload': True},
              'comparison_policy': 'Exact packed bytes only for same-input identical loaded source/window operators; no cross-topology hidden/logit equality gate.'}
    try:
        qualify(args, report)
    except Exception as exc:
        report['reason'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'stage': report.get('stage'), 'evidence': str(args.out)}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
