#!/usr/bin/env python3
"""M9 save/teardown/restore/resume qualification for official checkpoint."""
from __future__ import annotations

import argparse, json, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession, PRODUCTION_PREFILL_SELECTOR
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession, M8TurnRecord
from ds41f_mlx.runtime.kv_persistence import DEFAULT_KV_ROOT, save_m8_idle_state, restore_m8_idle_state
from tools.run_m8_long_session_qualification import DEFAULT_RECIPE, load_recipe_codec, assistant_message_from_generated, generate_until_boundary, git_rev, git_dirty, rss_mb


def _base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        'schema': 'ds41f.m9.kv-persistence-qualification.v1',
        'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_dirty': git_dirty(_REPO_ROOT),
        'checkpoint': str(args.checkpoint),
        'omlx_path': str(args.omlx_path),
        'omlx_revision': git_rev(args.omlx_path),
        'recipe_path': str(args.recipe_path),
        'artifact_root': str(args.artifact_root),
        'production_prefill_selector': PRODUCTION_PREFILL_SELECTOR,
        'started_at': time.time(),
        'passed': False,
    }


def phase_save(args: argparse.Namespace) -> int:
    result = _base_result(args); result['phase'] = 'save'; rt = None; sess = None
    try:
        codec = load_recipe_codec(args.recipe_path); result['recipe_codec'] = codec.__dict__ | {'tokenizer': type(codec.tokenizer).__name__}
        messages = [{'role': 'user', 'content': args.initial_user}]
        tokens = codec.encode_chat(messages)
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
        model, _ = rt.load_model()
        cfg = OMLXDecodeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
        pre = DwarfStarMLXPrefillSession(model, omlx_path=args.omlx_path).prefill(tokens[:-1])
        gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=tokens[-1], config=cfg, max_tokens=args.decode_tokens + 16)
        sess = M8LiveContinuationSession(model=model, live_cache=[], token_history=gen.current_token_history(), config=cfg)
        sess.generation = gen
        sess.turn_records.append(M8TurnRecord(0, len(tokens)-1, 1, 0, tokens[-1], (), sess.frontier, int(gen.prompt_replay_count), 0, 0.0, 0.0))
        generated, decode_s, _ = generate_until_boundary(sess, args.decode_tokens)
        assistant = codec.decode(generated); messages.append(assistant_message_from_generated(assistant))
        info = save_m8_idle_state(artifact_root=args.artifact_root, model=model, live_cache=sess.live_cache, all_tokens=sess.token_history, checkpoint=args.checkpoint, omlx_path=args.omlx_path, diagnostics=sess.diagnostics())
        state = {'artifact': str(info.path), 'messages': messages, 'assistant_text': assistant, 'frontier': sess.frontier}
        args.state.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        result.update({'artifact': info.to_json(), 'frontier': sess.frontier, 'generated': generated, 'assistant_text_sample': assistant[:200], 'decode_seconds': decode_s, 'diagnostics': sess.diagnostics(), 'rss_mb': rss_mb(), 'passed': True})
    except Exception as exc:
        result['error'] = repr(exc); result['traceback'] = traceback.format_exc()
    finally:
        try:
            if sess is not None: sess.close()
        except Exception as exc: result['close_error'] = repr(exc)
        if rt is not None: rt.close()
        result['finished_at'] = time.time(); args.save_out.write_text(json.dumps(result, indent=2, ensure_ascii=False)); print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['passed'] else 1


def phase_resume(args: argparse.Namespace) -> int:
    result = _base_result(args); result['phase'] = 'resume'; rt = None; sess = None
    try:
        state = json.loads(args.state.read_text())
        codec = load_recipe_codec(args.recipe_path); result['recipe_codec'] = codec.__dict__ | {'tokenizer': type(codec.tokenizer).__name__}
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
        model, _ = rt.load_model()
        cache, history, manifest = restore_m8_idle_state(artifact_path=Path(state['artifact']), model=model, checkpoint=args.checkpoint, omlx_path=args.omlx_path)
        cfg = OMLXDecodeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
        sess = M8LiveContinuationSession.from_live_cache(model=model, live_cache=cache, token_history=history, config=cfg)
        messages = state['messages'] + [{'role': 'user', 'content': args.followup}]
        next_tokens = codec.encode_chat(messages)
        prefix_ok = next_tokens[:sess.frontier] == sess.token_history
        sess.begin_turn_from_recipe_tokens(next_tokens, max_tokens=args.decode_tokens + 16)
        generated, decode_s, _ = generate_until_boundary(sess, args.decode_tokens)
        diag = sess.diagnostics()
        result.update({'artifact': state['artifact'], 'restored_frontier': len(history), 'manifest_frontier': manifest['frontier'], 'exact_prefix_extension': prefix_ok, 'new_suffix_len': len(next_tokens)-len(history), 'generated': generated, 'assistant_text_sample': codec.decode(generated)[:200], 'decode_seconds': decode_s, 'decode_tok_s': len(generated)/decode_s if decode_s else None, 'diagnostics': diag, 'rss_mb': rss_mb()})
        result['passed'] = bool(prefix_ok and diag['total_prompt_replay_count'] == 0 and diag['total_full_cache_repack_count'] == 0 and diag['all_cache_offsets_equal_frontier'])
    except Exception as exc:
        result['error'] = repr(exc); result['traceback'] = traceback.format_exc()
    finally:
        try:
            if sess is not None: sess.close()
        except Exception as exc: result['close_error'] = repr(exc)
        if rt is not None: rt.close()
        result['finished_at'] = time.time(); args.resume_out.write_text(json.dumps(result, indent=2, ensure_ascii=False)); print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['passed'] else 1


def orchestrate(args: argparse.Namespace) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True); args.artifact_root.mkdir(parents=True, exist_ok=True)
    state = args.out.with_suffix('.state.json'); save_out = args.out.with_suffix('.save.json'); resume_out = args.out.with_suffix('.resume.json')
    base = [sys.executable, __file__, '--phase']
    common = ['--checkpoint', str(args.checkpoint), '--omlx-path', str(args.omlx_path), '--recipe-path', str(args.recipe_path), '--artifact-root', str(args.artifact_root), '--state', str(state), '--decode-tokens', str(args.decode_tokens), '--initial-user', args.initial_user, '--followup', args.followup]
    s = subprocess.run(base + ['save', '--save-out', str(save_out)] + common)
    r = subprocess.run(base + ['resume', '--resume-out', str(resume_out)] + common) if s.returncode == 0 else None
    combined = {'schema': 'ds41f.m9.kv-persistence-orchestrated.v1', 'save': json.loads(save_out.read_text()) if save_out.exists() else None, 'resume': json.loads(resume_out.read_text()) if resume_out.exists() else None, 'passed': s.returncode == 0 and r is not None and r.returncode == 0}
    args.out.write_text(json.dumps(combined, indent=2, ensure_ascii=False)); print(json.dumps(combined, indent=2, ensure_ascii=False))
    return 0 if combined['passed'] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['orchestrate','save','resume'], default='orchestrate')
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument('--omlx-path', type=Path, default=DEFAULT_OMLX)
    ap.add_argument('--recipe-path', type=Path, default=DEFAULT_RECIPE)
    ap.add_argument('--artifact-root', type=Path, default=DEFAULT_KV_ROOT)
    ap.add_argument('--out', type=Path, default=Path('artifacts/m9/kv-persistence-qualification.json'))
    ap.add_argument('--save-out', type=Path, default=Path('artifacts/m9/save.json'))
    ap.add_argument('--resume-out', type=Path, default=Path('artifacts/m9/resume.json'))
    ap.add_argument('--state', type=Path, default=Path('artifacts/m9/state.json'))
    ap.add_argument('--decode-tokens', type=int, default=4)
    ap.add_argument('--initial-user', default='Answer tersely. Start with ready and one noun.')
    ap.add_argument('--followup', default='Continue with one color word.')
    args = ap.parse_args()
    for p in (args.out, args.save_out, args.resume_out, args.state): p.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == 'save': return phase_save(args)
    if args.phase == 'resume': return phase_resume(args)
    return orchestrate(args)

if __name__ == '__main__':
    raise SystemExit(main())
