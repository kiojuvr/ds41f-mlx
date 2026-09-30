#!/usr/bin/env python3
"""Bounded real P5 same-cache terminal bootstrap; structural evidence, no timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import (
    DwarfStarFP8MLXPrefillExecutorSetup, LivePrefillResult, SweepPlanner, handoff_to_generation,
)
from tools.qualify_prefill_fp8_mlx import runtime_authority


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-prompt-tokens', type=int, choices=(2049, 8193), required=True)
    p.add_argument('--mtp-off', action='store_true', required=True)
    p.add_argument('--no-benchmark', action='store_true', required=True)
    p.add_argument('--generated-tokens', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
    p.add_argument('--out', type=Path)
    return p


def identity(cache):
    return {'list': id(cache), 'layers': {str(i): {'object': id(cache[i]), 'slots': [id(cache[i][s]) for s in range(7)]} for i in (0, 20, 39)}}


def main(argv=None):
    args = parser().parse_args(argv)
    prefix_count = args.full_prompt_tokens - 1
    report = {'status': 'FAIL', 'prefix_tokens': prefix_count, 'full_prompt_tokens': args.full_prompt_tokens,
              'terminal_prompt_token': 3, 'prefix_fixture': '[1] repeated; terminal 3 held out',
              'load_kwargs': {'preserve_mtp': False, 'engram_ssd_offload': True}}
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
        print(json.dumps({'runtime_authority': report['runtime_authority']}), flush=True)
        report['stage'] = 'checkpoint_load'
        model, _ = load(args.checkpoint, preserve_mtp=False, engram_ssd_offload=True)
        lm = model.language_model
        with tempfile.TemporaryDirectory(prefix='ds41f-p5-smoke-') as tmp:
            planner = SweepPlanner(load_native_prefill_library(compile_native_prefill_library(Path(tmp))))
            plan = planner.build_from_model(lm, ctx=32768, remaining=prefix_count)
        assert plan.count == prefix_count and plan.decoder_suffix == (prefix_count == 8192)
        # Request-owned complete prefix; never include the held-out terminal.
        prefix_ids = [1] * prefix_count
        setup = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=mx).prepare(plan, prefix_ids)
        runner, arena = setup.block_runner, setup.arena
        for command in plan.commands:
            report['stage'] = f'prefill:{command.index}:{command.kind.value}:layer{command.layer}'
            runner.execute_command(command, arena)
            mx.eval(arena.carry.current.value, arena.carry.next.value, arena.carry.pre.value,
                    *[v for cache in runner.working_cache for v in cache.cache if v is not None])
        cache = runner.working_cache
        report['pre_handoff_frontiers'] = [int(c.size()) for c in cache]
        assert report['pre_handoff_frontiers'] == [prefix_count] * 40
        report['identity_after_prefill'] = identity(cache)
        result = LivePrefillResult.from_committed(setup, prefix_token_ids=prefix_ids)
        assert result.live_cache is cache
        report['identity_before_transfer'] = identity(result.live_cache)
        assert report['identity_before_transfer'] == report['identity_after_prefill']
        forwarded = []
        report['decode_forward_calls'] = forwarded
        original_forward = type(lm)._forward
        def forward(obj, input_ids, cache=None, *a, **kw):
            if obj is lm:
                forwarded.append({'input_ids': input_ids.tolist(), 'frontiers_before': [int(c.size()) for c in cache]})
            return original_forward(obj, input_ids, cache, *a, **kw)
        original_admission = OMLXGenerationSession.from_prefilled_cache
        def admission(cls, model_arg, cache_arg, ids, *a, **kw):
            assert cache_arg is cache
            report['identity_at_generation_admission'] = identity(cache_arg)
            assert report['identity_at_generation_admission'] == report['identity_after_prefill']
            admitted = original_admission(model_arg, cache_arg, ids, *a, **kw)
            assert admitted.initial_cache is cache
            report['prefix_all_tokens_seed_count'] = len(admitted.prefix_tokens)
            report['prefix_all_tokens_seed_matches_request'] = admitted.prefix_tokens == prefix_ids
            return admitted
        cfg = OMLXDecodeConfig(omlx_path=Path(lang.__file__).resolve().parents[3], checkpoint_path=args.checkpoint,
                               preserve_mtp=False, engram_ssd_offload=True)
        with patch.object(type(lm), '_forward', forward), patch.object(OMLXGenerationSession, 'from_prefilled_cache', classmethod(admission)):
            report['stage'] = 'handoff_and_terminal_bootstrap'
            session = handoff_to_generation(result, model, terminal_prompt_token=3, config=cfg, max_tokens=4)
            report['post_bootstrap_frontiers'] = list(session.active_cache_offsets())
            assert report['post_bootstrap_frontiers'] == [prefix_count + 1] * 40
            assert len(forwarded) == 1 and forwarded[0]['input_ids'] == [[3]]
            assert forwarded[0]['frontiers_before'] == [prefix_count] * 40
            assert session.bootstrap_inserted_prompt == (3,) and session.prompt_replay_count == 0
            assert runner.working_cache is None and session.initial_cache == []
            report['BatchGenerator_inserted_prompt'] = list(session.bootstrap_inserted_prompt)
            report['terminal_bootstrap_forward_count'] = len(forwarded)
            report['cache_authority_owner'] = result.cache_authority_owner
            report['generated_tokens'] = []
            report['generated_frontiers'] = []
            for step in range(args.generated_tokens):
                report['stage'] = f'bounded_generation:{step}'
                response = session.next_token()
                assert response is not None
                report['generated_tokens'].append(response.token)
                frontiers = list(session.active_cache_offsets())
                report['generated_frontiers'].append(frontiers)
                assert frontiers == [prefix_count + 2 + step] * 40
                assert session.prompt_replay_count == 0
                if response.finish_reason is not None:
                    break
        assert len(report['generated_tokens']) >= 1
        assert result.handoff_count == 1
        assert runner.full_cache_repack_count == 0 and not runner.prefill_continuation_exported
        assert runner.final_logits_suppressed and runner.final_logits is None
        report.update(status='PASS', stage='complete', handoff_count=result.handoff_count,
                      prompt_replay_count=session.prompt_replay_count,
                      full_cache_repack_count=runner.full_cache_repack_count,
                      PrefillContinuationState_export_count=int(runner.prefill_continuation_exported),
                      final_prefix_logits_suppressed=runner.final_logits_suppressed,
                      full_source_generation_count=runner.suffix_math.full_source_generation_count,
                      one_live_cache_handoff_no_replay_qualified_within_new_path=True,
                      production_selection_changed=False)
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
