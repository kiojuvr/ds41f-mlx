#!/usr/bin/env python3
"""Bounded real P6 deferred-append precursor smoke; structural evidence, no timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult, handoff_to_generation
from tools.qualify_prefill_fp8_mlx import runtime_authority


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('complete-16384', 'pending-16384'), required=True)
    p.add_argument('--no-benchmark', action='store_true', required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
    p.add_argument('--generated-tokens', type=int, default=2, choices=(1, 2, 3))
    p.add_argument('--out', type=Path)
    return p


def token(i: int) -> int:
    return 16 + ((37 * i) % 4096)


def identity(cache):
    return {'list': id(cache), 'layers': {str(i): {'object': id(cache[i]), 'slots': [id(cache[i][s]) for s in range(7)]} for i in (0, 20, 39)}}


def main(argv=None):
    args = parser().parse_args(argv)
    report = {
        'status': 'FAIL',
        'case': args.case,
        'load_kwargs': {'preserve_mtp': False, 'engram_ssd_offload': True},
        'benchmark': False,
    }
    model = session = None
    try:
        import mlx.core as mx
        import omlx.patches.deepseek_v41.language as lang
        from omlx.patches.deepseek_v41.loading import load
        from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
        from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
        report['runtime_authority'] = runtime_authority(lang)
        if any(v != 'compatible' for v in report['runtime_authority']['operation_contracts'].values()):
            raise RuntimeError('imported runtime API incompatible with reviewed adapter')
        report['stage'] = 'checkpoint_load'
        model, _ = load(args.checkpoint, preserve_mtp=False, engram_ssd_offload=True)
        lm = model.language_model
        if args.case == 'complete-16384':
            T = 16384
            prefix_ids = [token(i) for i in range(T)]
            terminal = token(T)
            app = DeferredPrefillAppend.create(lm, lm.make_cache(), prefix_ids, committed_frontier=0, mx=mx)
            report['plan'] = [(s.start, s.count, s.mode.value) for s in app.plan.segments]
            report['stage'] = 'p6_execute_all'
            app.execute_all()
            cache = app.live_cache
            report['frontiers_after_seal'] = [int(c.size()) for c in cache]
            assert report['frontiers_after_seal'] == [T] * 40
            assert app.commit_certificate is not None
            assert app.segment_records and any(r.cone_rows == 2541 for r in app.segment_records)
            assert all(r.materialized for r in app.segment_records if r.mode.value != 'ORDINARY_COMPLETE_RANGE')
            final_arena = app.final_execution.arena
            assert final_arena.p6_final_cone_detached and final_arena.p6_final_cone_rows == 2541
            assert final_arena.encoder_final_h is None and final_arena.encoder_final_pre is None
            assert final_arena.carry.current.rows <= 2541 and final_arena.carry.next.rows <= 2541 and final_arena.carry.pre.rows <= 2541
            report['identity_after_prefill'] = identity(cache)
            result = LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=prefix_ids)
            report['identity_before_transfer'] = identity(result.live_cache)
            assert report['identity_before_transfer'] == report['identity_after_prefill']
            forwarded = []
            original_forward = type(lm)._forward
            def forward(obj, input_ids, cache=None, *a, **kw):
                if obj is lm:
                    try:
                        ids = input_ids.tolist()
                    except Exception:
                        ids = str(input_ids)
                    forwarded.append({'input_ids': ids, 'frontiers_before': [int(c.size()) for c in cache]})
                return original_forward(obj, input_ids, cache, *a, **kw)
            original_admission = OMLXGenerationSession.from_prefilled_cache
            def admission(cls, model_arg, cache_arg, ids, *a, **kw):
                assert cache_arg is cache
                admitted = original_admission(model_arg, cache_arg, ids, *a, **kw)
                assert admitted.initial_cache is cache
                assert admitted.prefix_tokens == prefix_ids
                return admitted
            cfg = OMLXDecodeConfig(omlx_path=Path(lang.__file__).resolve().parents[3], checkpoint_path=args.checkpoint,
                                   preserve_mtp=False, engram_ssd_offload=True)
            with patch.object(type(lm), '_forward', forward), patch.object(OMLXGenerationSession, 'from_prefilled_cache', classmethod(admission)):
                report['stage'] = 'p5_handoff_terminal'
                session = handoff_to_generation(result, model, terminal_prompt_token=terminal, config=cfg, max_tokens=4)
                assert list(session.active_cache_offsets()) == [T + 1] * 40
                assert session.prompt_replay_count == 0 and len(forwarded) == 1
                report['generated_tokens'] = []
                report['generated_frontiers'] = []
                for step in range(args.generated_tokens):
                    response = session.next_token()
                    assert response is not None
                    report['generated_tokens'].append(response.token)
                    fronts = list(session.active_cache_offsets())
                    report['generated_frontiers'].append(fronts)
                    assert fronts == [T + 2 + step] * 40
                    if response.finish_reason is not None:
                        break
            report.update(status='PASS', stage='complete', terminal_bootstrap_forward_count=len(forwarded),
                          prompt_replay_count=session.prompt_replay_count, full_cache_repack_count=app.final_execution.runner.full_cache_repack_count,
                          PrefillContinuationState_export_count=int(app.final_execution.runner.prefill_continuation_exported))
        else:
            T = 24577
            prefix_ids = [token(i) for i in range(T)]
            app = DeferredPrefillAppend.create(lm, lm.make_cache(), prefix_ids, committed_frontier=0, mx=mx)
            report['plan'] = [(s.start, s.count, s.mode.value) for s in app.plan.segments]
            app.begin()
            pre_runner_cache = app.live_cache
            report['stage'] = 'p6_first_source_only'
            first = app.execute_segment(app.plan.segments[0])
            cache = app.live_cache
            report['public_frontiers'] = [int(c.size()) for c in cache]
            assert report['public_frontiers'] == [0] * 40
            assert app.E == 16384 and app.D == 0 and app.engram_history_position == 16384
            assert app.coverage.source_by_layer[20] == 16384
            assert first.runner.closed
            assert first.arena.carry.current.value is None and first.arena.carry.pre.value is None and first.arena.input_ids.value is None
            assert first.arena.p6_source_materialized and first.arena.p6_source_materialization_events
            assert not lm.layers[20].calls and all(not lm.layers[i].calls for i in range(21, 40))
            raw_rejected = False
            try:
                OMLXGenerationSession.from_prefilled_cache(model, pre_runner_cache, prefix_ids)
            except Exception:
                raw_rejected = True
            assert raw_rejected, 'raw generation admission unexpectedly accepted pending P6 cache'
            report.update(status='PASS', stage='pending-boundary-complete', private_E=app.E, private_D=app.D,
                          layer20_source_coverage=app.coverage.source_by_layer[20], source_runner_closed=first.runner.closed,
                          historical_records=len(app.segment_records))
    except ImportError as exc:
        report.update(status='NOT RUN', reason=str(exc))
    except Exception as exc:
        report['reason'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        if session is not None:
            session.close()
        if model is not None:
            model.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else (2 if report['status'] == 'NOT RUN' else 1)


if __name__ == '__main__':
    raise SystemExit(main())
