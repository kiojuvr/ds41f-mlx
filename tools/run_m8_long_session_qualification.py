#!/usr/bin/env python3
"""Run M8 real-checkpoint repeated DeepSeek-recipe continuation qualification.

The primary path encodes a complete multi-turn text conversation each turn and
then requires the completed GenerationBatch ``all_tokens`` to be an exact prefix
of the next recipe encoding.  Synthetic token suffixes are supported only by the
``--synthetic-suffixes`` diagnostic option and are not sufficient for M8 closeout.
"""
from __future__ import annotations

import argparse, json, os, resource, subprocess, sys, time, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession, PRODUCTION_PREFILL_SELECTOR
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession, M8ContinuationError, M8TurnRecord

DEFAULT_RECIPE = Path('/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe')


@dataclass
class RecipeCodec:
    tokenizer: Any
    source: str
    revision: str | None
    version: str | None

    def render_chat(self, messages: list[dict[str, str]]) -> str:
        # Official DeepSeek-recipe V4.1 text-only rendering, transcribed from
        # deepseek-recipe-encoding/src/v4 for the simple chat subset used here.
        # Chat Completions default to thinking mode in the pinned M7 recipe path.
        bos = '<｜begin▁of▁sentence｜>'
        user = '<｜User｜>'
        assistant = '<｜Assistant｜>'
        eos = '<｜end▁of▁sentence｜>'
        think = '<think>'
        endthink = '</think>'
        out = [bos]
        if messages and messages[0]['role'] != 'system':
            out.append('<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n')
        prev_role = None
        for msg in messages:
            role, content = msg['role'], msg['content']
            if role == 'system':
                out.append('<｜System｜>' + content)
            elif role == 'user':
                out.append(('\n\n' if prev_role == 'user' else user) + content)
            elif role == 'assistant':
                reasoning = msg.get('reasoning_content', '')
                out.append(assistant + think + reasoning + endthink + content + eos)
            else:
                raise ValueError(f'unsupported text-only role {role!r}')
            prev_role = role
        out.append(assistant + think)
        return ''.join(out)

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        prompt = self.render_chat(messages)
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        ids = encoded.ids if hasattr(encoded, 'ids') else encoded
        return [int(x) for x in ids]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode([int(t) for t in token_ids], skip_special_tokens=False)


def git_dirty(path: Path) -> bool | None:
    try:
        status = subprocess.check_output(['git', '-C', str(path), 'status', '--short', '--', '.', ':(exclude)artifacts/m8/long-session-qualification.json'], text=True).strip()
        return bool(status)
    except Exception:
        return None


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def load_recipe_codec(recipe_path: Path) -> RecipeCodec:
    # Prefer the pinned Python binding when available.  The local environment may
    # lack the native extension; then use tokenizers plus the official renderer
    # source transcribed above and record that provenance explicitly.
    try:
        if str(recipe_path / 'deepseek-recipe-python' / 'python') not in sys.path:
            sys.path.insert(0, str(recipe_path / 'deepseek-recipe-python' / 'python'))
        from deepseek_recipe import Tokenizer  # type: ignore
        tok = Tokenizer.from_file(str(recipe_path / 'static' / 'tokenizers' / 'v41' / 'tokenizer.json'))
        # Adapter to tokenizers-like API.
        class _Tok:
            def encode(self, text: str, add_special_tokens: bool = False):
                class _Enc:
                    def __init__(self, ids): self.ids = ids
                return _Enc([int(x) for x in tok.encode(text)])
            def decode(self, ids, skip_special_tokens: bool = False):
                return tok.decode([int(x) for x in ids])
        return RecipeCodec(_Tok(), 'deepseek_recipe.python_binding', git_rev(recipe_path), pkg_version('deepseek-recipe'))
    except Exception as exc:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(recipe_path / 'static' / 'tokenizers' / 'v41' / 'tokenizer.json'))
        return RecipeCodec(tok, f'official_renderer_transcription+tokenizers_fallback: {type(exc).__name__}: {exc}', git_rev(recipe_path), pkg_version('tokenizers'))


def parse_suffixes(text: str) -> list[list[int]]:
    turns = []
    for part in text.split(';'):
        part = part.strip()
        if part:
            turns.append([int(x) for x in part.replace(',', ' ').split()])
    return turns


def rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.  Keep this as coarse process
    # qualification telemetry rather than a hot-path dependency.
    return raw / (1024.0 * 1024.0) if raw > 10_000_000 else raw / 1024.0


def assistant_message_from_generated(text: str) -> dict[str, str]:
    marker = '</think>'
    if marker in text:
        reasoning, content = text.split(marker, 1)
        return {'role': 'assistant', 'reasoning_content': reasoning, 'content': content}
    return {'role': 'assistant', 'reasoning_content': text, 'content': ''}


def generate_until_boundary(sess: M8LiveContinuationSession, count: int, *, cancel_after_first: bool = False) -> tuple[list[int], float, bool]:
    generated: list[int] = []
    t0 = time.perf_counter()
    cancelled = False
    for i in range(count):
        rep = sess.next_token()
        if rep is None:
            break
        generated.append(int(rep.token))
        if cancel_after_first and i == 0:
            sess.cancel_turn('qualification_cancel')
            cancelled = True
            break
    if not cancelled:
        sess.ensure_idle('qualification_turn_boundary')
    return generated, time.perf_counter() - t0, cancelled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument('--omlx-path', type=Path, default=DEFAULT_OMLX)
    ap.add_argument('--recipe-path', type=Path, default=DEFAULT_RECIPE)
    ap.add_argument('--out', type=Path, default=Path('artifacts/m8/long-session-qualification.json'))
    ap.add_argument('--kv-dir', type=Path, default=Path('/Volumes/USB-SSD-RAID-0/ds41f-mlx/m8-kv'))
    ap.add_argument('--initial-user', default='Answer tersely. Start with the word ready, then give one short noun.')
    ap.add_argument('--followup', action='append', default=[], help='Follow-up user turn; may be repeated.')
    ap.add_argument('--turns', type=int, default=8)
    ap.add_argument('--decode-tokens-per-turn', type=int, default=6)
    ap.add_argument('--cancel-turn', type=int, default=3, help='0-based follow-up turn to cancel after one token; negative disables')
    ap.add_argument('--synthetic-suffixes', default=None, help='diagnostic-only token suffix matrix; bypasses recipe boundary')
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.kv_dir.mkdir(parents=True, exist_ok=True)
    followups = list(args.followup) or [
        'Now add a color in two words.',
        'Continue with a small animal, no sentence.',
        'Give a number and a punctuation mark.',
        'Add one short verb.',
        'Give one weather word.',
        'Add one location noun.',
        'Give one adjective.',
        'Finish with one object noun.',
    ]
    while len(followups) < args.turns:
        followups.append(f'Respond with marker {len(followups)} and one word.')
    followups = followups[:args.turns]

    result: dict[str, Any] = {
        'schema': 'ds41f.m8.long-session-qualification.v2',
        'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_dirty': git_dirty(_REPO_ROOT),
        'checkpoint': str(args.checkpoint),
        'omlx_path': str(args.omlx_path),
        'omlx_revision': git_rev(args.omlx_path),
        'recipe_path': str(args.recipe_path),
        'kv_dir_policy_for_future_persistence': str(args.kv_dir),
        'production_prefill_selector': PRODUCTION_PREFILL_SELECTOR,
        'decode_tokens_per_turn': args.decode_tokens_per_turn,
        'turns_requested': args.turns,
        'started_at': time.time(),
        'rss_mb_start': rss_mb(),
        'turns': [],
        'invalid_input_gate': None,
        'synthetic_mode': args.synthetic_suffixes is not None,
        'passed': False,
    }
    rt = None
    sess = None
    try:
        codec = load_recipe_codec(args.recipe_path)
        result['recipe_codec'] = {'source': codec.source, 'revision': codec.revision, 'version': codec.version}
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
        model, _ = rt.load_model()
        cfg = OMLXDecodeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)

        if args.synthetic_suffixes is not None:
            initial = [0, 3]
            suffixes = parse_suffixes(args.synthetic_suffixes)
            pre = DwarfStarMLXPrefillSession(model, omlx_path=args.omlx_path).prefill(initial)
            gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=15, config=cfg, max_tokens=args.decode_tokens_per_turn + 16)
            sess = M8LiveContinuationSession(model=model, live_cache=[], token_history=gen.current_token_history(), config=cfg)
            sess.generation = gen
            sess.turn_records.append(M8TurnRecord(0, len(initial), 1, 0, 15, (), sess.frontier, int(gen.prompt_replay_count), 0, 0.0, 0.0))
            initial_generated, initial_decode_s, _ = generate_until_boundary(sess, args.decode_tokens_per_turn)
            result['initial'] = {'mode': 'synthetic', 'prefix_len': len(initial), 'generated': initial_generated, 'decode_seconds': initial_decode_s}
            for i, suffix in enumerate(suffixes):
                t0 = time.perf_counter(); before = sess.frontier
                sess.begin_turn_from_suffix(suffix, max_tokens=args.decode_tokens_per_turn + 16)
                generated, decode_s, cancelled = generate_until_boundary(sess, args.decode_tokens_per_turn, cancel_after_first=(i == args.cancel_turn))
                diag = sess.diagnostics()
                result['turns'].append({'turn_index': i, 'mode': 'synthetic', 'frontier_before': before, 'suffix_len': len(suffix), 'generated': generated, 'seconds': time.perf_counter() - t0, 'decode_seconds': decode_s, 'cancelled': cancelled, 'diagnostics': diag, 'rss_mb': rss_mb()})
        else:
            messages = [{'role': 'user', 'content': args.initial_user}]
            initial_tokens = codec.encode_chat(messages)
            if len(initial_tokens) < 2:
                raise M8ContinuationError('initial recipe encoding too short')
            pre = DwarfStarMLXPrefillSession(model, omlx_path=args.omlx_path).prefill(initial_tokens[:-1])
            gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=initial_tokens[-1], config=cfg, max_tokens=args.decode_tokens_per_turn + 16)
            sess = M8LiveContinuationSession(model=model, live_cache=[], token_history=gen.current_token_history(), config=cfg)
            sess.generation = gen
            sess.turn_records.append(M8TurnRecord(0, len(initial_tokens) - 1, 1, 0, initial_tokens[-1], (), sess.frontier, int(gen.prompt_replay_count), 0, 0.0, 0.0))
            initial_generated, initial_decode_s, _ = generate_until_boundary(sess, args.decode_tokens_per_turn)
            assistant_text = codec.decode(initial_generated)
            messages.append(assistant_message_from_generated(assistant_text))
            result['initial'] = {'mode': 'recipe', 'prompt_tokens': len(initial_tokens), 'terminal': initial_tokens[-1], 'generated': initial_generated, 'assistant_text_sample': assistant_text[:200], 'frontier_after': sess.frontier, 'decode_seconds': initial_decode_s, 'rss_mb': rss_mb()}

            # Invalid/non-extension gate before any valid next append.
            before_frontier = sess.frontier
            before_diag = sess.diagnostics()
            try:
                bad = list(sess.token_history); bad[-1] = (bad[-1] + 1) % 1000; bad.append(42)
                sess.begin_turn_from_recipe_tokens(bad, max_tokens=args.decode_tokens_per_turn)
                result['invalid_input_gate'] = {'passed': False, 'error': 'accepted non-extension'}
            except M8ContinuationError as exc:
                result['invalid_input_gate'] = {'passed': sess.frontier == before_frontier and sess.diagnostics()['frontier'] == before_diag['frontier'], 'error': repr(exc), 'frontier': sess.frontier}

            for i, user_text in enumerate(followups):
                messages.append({'role': 'user', 'content': user_text})
                next_tokens = codec.encode_chat(messages)
                prefix_ok = next_tokens[:sess.frontier] == sess.token_history
                mismatch = None
                if not prefix_ok:
                    j = 0
                    for a, b in zip(next_tokens, sess.token_history):
                        if a != b: break
                        j += 1
                    mismatch = {'common_prefix': j, 'session_frontier': sess.frontier, 'next_len': len(next_tokens), 'session_at_mismatch': sess.token_history[j:j+8], 'next_at_mismatch': next_tokens[j:j+8]}
                t0 = time.perf_counter(); before = sess.frontier
                sess.begin_turn_from_recipe_tokens(next_tokens, max_tokens=args.decode_tokens_per_turn + 16)
                generated, decode_s, cancelled = generate_until_boundary(sess, args.decode_tokens_per_turn, cancel_after_first=(i == args.cancel_turn))
                assistant_text = codec.decode(generated)
                messages.append(assistant_message_from_generated(assistant_text))
                diag = sess.diagnostics()
                result['turns'].append({'turn_index': i, 'mode': 'recipe', 'frontier_before': before, 'next_recipe_tokens': len(next_tokens), 'exact_prefix_extension': prefix_ok, 'mismatch': mismatch, 'new_suffix_len': len(next_tokens) - before, 'generated': generated, 'assistant_text_sample': assistant_text[:200], 'seconds': time.perf_counter() - t0, 'decode_seconds': decode_s, 'decode_tok_s': (len(generated) / decode_s) if decode_s > 0 else None, 'cancelled': cancelled, 'diagnostics': diag, 'rss_mb': rss_mb()})
                if not prefix_ok:
                    raise M8ContinuationError('recipe exact-prefix-extension contract failed')

        final = sess.diagnostics() if sess is not None else None
        result['final_diagnostics'] = final
        result['rss_mb_finish'] = rss_mb()
        gates = [
            final is not None,
            result.get('invalid_input_gate', {'passed': True})['passed'],
            final is None or final['total_prompt_replay_count'] == 0,
            final is None or final['total_full_cache_repack_count'] == 0,
            final is None or final['all_cache_offsets_equal_frontier'],
            all(t['diagnostics']['all_cache_offsets_equal_frontier'] for t in result['turns']),
            all(t.get('exact_prefix_extension', True) for t in result['turns']),
        ]
        result['passed'] = bool(all(gates))
        if not result['passed']:
            raise M8ContinuationError('one or more M8 gates failed')
    except Exception as exc:
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()
    finally:
        try:
            if sess is not None:
                sess.close()
        except Exception as exc:
            result['close_error'] = repr(exc)
        try:
            if rt is not None:
                rt.close()
        except Exception:
            pass
        result['finished_at'] = time.time()
        result['duration_s'] = result['finished_at'] - result['started_at']
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
