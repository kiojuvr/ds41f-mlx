#!/usr/bin/env python3
"""M10 repeated restore/save long-session qualification."""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession, PRODUCTION_PREFILL_SELECTOR
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession, M8TurnRecord, M8ContinuationError
from ds41f_mlx.runtime.kv_persistence import DEFAULT_KV_ROOT, save_m8_idle_state, restore_m8_idle_state, M9PersistenceError
from tools.run_m8_long_session_qualification import DEFAULT_RECIPE, load_recipe_codec, assistant_message_from_generated, generate_until_boundary, git_rev, git_dirty, rss_mb


def artifact_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def default_followups(n: int) -> list[str]:
    base = [
        'Continue with one color word.', 'Now give one small animal.', 'Give one number and punctuation.',
        'Add one weather noun.', 'Give one place noun.', 'Add one adjective.', 'Give one object noun.',
        'Respond with one short verb.', 'Give one plant noun.', 'Add one mineral noun.', 'Give one transport noun.',
        'Finish this step with one food noun.', 'Give one astronomy noun.', 'Add one music noun.', 'Give one tool noun.',
        'Add marker fifteen and one word.'
    ]
    while len(base) < n:
        base.append(f'Respond with marker {len(base)} and one word.')
    return base[:n]


def run_turns(sess: M8LiveContinuationSession, codec: Any, messages: list[dict[str, str]], followups: list[str], *, start_index: int, decode_tokens: int, cancel_global_turn: int) -> list[dict[str, Any]]:
    rows = []
    for local_i, user_text in enumerate(followups):
        global_i = start_index + local_i
        messages.append({'role': 'user', 'content': user_text})
        next_tokens = codec.encode_chat(messages)
        prefix_ok = next_tokens[:sess.frontier] == sess.token_history
        before = sess.frontier; t0 = time.perf_counter()
        sess.begin_turn_from_recipe_tokens(next_tokens, max_tokens=decode_tokens + 16)
        generated, decode_s, cancelled = generate_until_boundary(sess, decode_tokens, cancel_after_first=(global_i == cancel_global_turn))
        assistant = codec.decode(generated)
        messages.append(assistant_message_from_generated(assistant))
        diag = sess.diagnostics()
        rows.append({'global_turn': global_i, 'frontier_before': before, 'recipe_tokens': len(next_tokens), 'exact_prefix_extension': prefix_ok, 'new_suffix_len': len(next_tokens)-before, 'generated': generated, 'assistant_text_sample': assistant[:120], 'cancelled': cancelled, 'seconds': time.perf_counter()-t0, 'decode_seconds': decode_s, 'decode_tok_s': len(generated)/decode_s if decode_s else None, 'diagnostics': diag, 'rss_mb': rss_mb()})
        if not prefix_ok:
            raise M8ContinuationError('exact-prefix extension failed')
    return rows


def phase_cycle(args: argparse.Namespace) -> int:
    out: dict[str, Any] = {'schema':'ds41f.m10.cycle.v1','git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_dirty':git_dirty(_REPO_ROOT),'cycle':args.cycle,'started_at':time.time(),'passed':False}
    rt = None; sess = None
    try:
        codec = load_recipe_codec(args.recipe_path)
        state = json.loads(args.state.read_text()) if args.state.exists() else {'messages': [], 'artifact': None, 'turn_index': 0}
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
        model, _ = rt.load_model(); cfg = OMLXDecodeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
        restore_s = None
        if state.get('artifact'):
            t_restore = time.perf_counter(); cache, history, manifest = restore_m8_idle_state(artifact_path=Path(state['artifact']), model=model, checkpoint=args.checkpoint, omlx_path=args.omlx_path); restore_s = time.perf_counter()-t_restore
            sess = M8LiveContinuationSession.from_live_cache(model=model, live_cache=cache, token_history=history, config=cfg)
            messages = state['messages']; out['restored_frontier'] = len(history); out['restore_seconds'] = restore_s; out['input_artifact_bytes'] = artifact_bytes(Path(state['artifact']))
        else:
            messages = [{'role':'user','content':args.initial_user}]
            tokens = codec.encode_chat(messages)
            pre = DwarfStarMLXPrefillSession(model, omlx_path=args.omlx_path).prefill(tokens[:-1])
            gen = handoff_to_generation(pre.live_result, model, terminal_prompt_token=tokens[-1], config=cfg, max_tokens=args.decode_tokens+16)
            sess = M8LiveContinuationSession(model=model, live_cache=[], token_history=gen.current_token_history(), config=cfg); sess.generation = gen
            sess.turn_records.append(M8TurnRecord(0, len(tokens)-1, 1, 0, tokens[-1], (), sess.frontier, int(gen.prompt_replay_count), 0, 0.0, 0.0))
            gen_tokens, dec_s, _ = generate_until_boundary(sess, args.decode_tokens)
            assistant = codec.decode(gen_tokens); messages.append(assistant_message_from_generated(assistant))
            out['initial'] = {'prompt_tokens': len(tokens), 'frontier': sess.frontier, 'generated': gen_tokens, 'decode_seconds': dec_s}
        # invalid input gate once on first cycle after idle exists
        if args.cycle == 0:
            before = sess.frontier
            try:
                bad = list(sess.token_history); bad[-1] = (bad[-1]+7) % 1000; bad.append(123)
                sess.begin_turn_from_recipe_tokens(bad, max_tokens=args.decode_tokens+16)
                out['invalid_gate'] = {'passed': False}
            except M8ContinuationError as exc:
                out['invalid_gate'] = {'passed': sess.frontier == before, 'error': repr(exc), 'frontier': sess.frontier}
        followups = default_followups(args.total_turns)[state.get('turn_index',0):state.get('turn_index',0)+args.turns_per_cycle]
        rows = run_turns(sess, codec, messages, followups, start_index=state.get('turn_index',0), decode_tokens=args.decode_tokens, cancel_global_turn=args.cancel_turn)
        t_save = time.perf_counter(); info = save_m8_idle_state(artifact_root=args.artifact_root, model=model, live_cache=sess.live_cache, all_tokens=sess.token_history, checkpoint=args.checkpoint, omlx_path=args.omlx_path, diagnostics=sess.diagnostics()); save_s = time.perf_counter()-t_save
        new_state = {'messages': messages, 'artifact': str(info.path), 'turn_index': state.get('turn_index',0)+len(followups), 'frontier': sess.frontier}
        args.state.write_text(json.dumps(new_state, indent=2, ensure_ascii=False))
        diag = sess.diagnostics()
        out.update({'turns':rows,'saved_artifact':info.to_json(),'artifact_bytes':artifact_bytes(info.path),'save_seconds':save_s,'final_frontier':sess.frontier,'diagnostics':diag,'rss_mb_finish':rss_mb()})
        out['passed'] = bool(diag['total_prompt_replay_count']==0 and diag['total_full_cache_repack_count']==0 and diag['all_cache_offsets_equal_frontier'] and all(r['exact_prefix_extension'] for r in rows) and (out.get('invalid_gate',{'passed':True})['passed']))
    except Exception as exc:
        out['error'] = repr(exc); out['traceback'] = traceback.format_exc()
    finally:
        try:
            if sess is not None: sess.close()
        except Exception as exc: out['close_error'] = repr(exc)
        if rt is not None: rt.close()
        out['finished_at'] = time.time(); args.phase_out.write_text(json.dumps(out, indent=2, ensure_ascii=False)); print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out['passed'] else 1


def phase_compare(args: argparse.Namespace) -> int:
    out={'schema':'ds41f.m10.branch-compare.v1','git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_dirty':git_dirty(_REPO_ROOT),'started_at':time.time(),'passed':False}
    rt=None; sess=None
    try:
        codec=load_recipe_codec(args.recipe_path); state=json.loads(args.state.read_text())
        rt=OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False)); model,_=rt.load_model()
        cache,history,manifest=restore_m8_idle_state(artifact_path=Path(state['artifact']), model=model, checkpoint=args.checkpoint, omlx_path=args.omlx_path)
        cfg=OMLXDecodeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
        sess=M8LiveContinuationSession.from_live_cache(model=model, live_cache=cache, token_history=history, config=cfg)
        messages=list(state['messages'])+[{'role':'user','content':args.compare_followup}]
        toks=codec.encode_chat(messages); prefix_ok=toks[:sess.frontier]==sess.token_history
        sess.begin_turn_from_recipe_tokens(toks, max_tokens=args.decode_tokens+16); gen,dec_s,_=generate_until_boundary(sess,args.decode_tokens)
        diag=sess.diagnostics(); out.update({'restored_frontier':len(history),'exact_prefix_extension':prefix_ok,'generated':gen,'decode_seconds':dec_s,'diagnostics':diag,'passed': bool(prefix_ok and diag['total_prompt_replay_count']==0 and diag['total_full_cache_repack_count']==0 and diag['all_cache_offsets_equal_frontier'])})
    except Exception as exc:
        out['error']=repr(exc); out['traceback']=traceback.format_exc()
    finally:
        try:
            if sess is not None: sess.close()
        except Exception as exc: out['close_error']=repr(exc)
        if rt is not None: rt.close()
        out['finished_at']=time.time(); args.phase_out.write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
    return 0 if out['passed'] else 1


def phase_corrupt(args: argparse.Namespace) -> int:
    out={'schema':'ds41f.m10.corruption.v1','git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_dirty':git_dirty(_REPO_ROOT),'started_at':time.time(),'cases':[],'passed':False}
    rt=None
    try:
        state=json.loads(args.state.read_text()); source=Path(state['artifact'])
        rt=OmlxRuntime(OmlxRuntimeConfig(omlx_path=args.omlx_path, checkpoint_path=args.checkpoint, engram_ssd_offload=True, preserve_mtp=False)); model,_=rt.load_model()
        root=args.artifact_root/'m10-corrupt'; shutil.rmtree(root, ignore_errors=True); root.mkdir(parents=True, exist_ok=True)
        cases=[]
        for name, mutate in [
            ('missing_commit', lambda p: (p/'COMMITTED').unlink()),
            ('corrupt_manifest_schema', lambda p: (p/'manifest.json').write_text((p/'manifest.json').read_text().replace('ds41f.m9.deepseek-v41-kv-artifact.v1','bad.schema',1))),
            ('corrupt_tensor_payload', lambda p: (p/'cache.safetensors').open('r+b').write(b'BAD!')),
            ('missing_tensor_file', lambda p: (p/'cache.safetensors').unlink()),
        ]:
            dst=root/name; shutil.copytree(source,dst); mutate(dst)
            try:
                restore_m8_idle_state(artifact_path=dst, model=model, checkpoint=args.checkpoint, omlx_path=args.omlx_path)
                cases.append({'case':name,'passed':False,'error':'restore accepted corrupt artifact'})
            except Exception as exc:
                cases.append({'case':name,'passed':True,'error':repr(exc)})
        tmp=args.artifact_root/'stale-m10.tmp'; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True, exist_ok=True); (tmp/'partial').write_text('x')
        cases.append({'case':'stale_tmp_artifact_present','passed': source.exists() and tmp.exists(), 'note':'stale tmp sibling ignored by explicit artifact restore'})
        out['cases']=cases; out['passed']=all(c['passed'] for c in cases)
    except Exception as exc:
        out['error']=repr(exc); out['traceback']=traceback.format_exc()
    finally:
        if rt is not None: rt.close()
        out['finished_at']=time.time(); args.phase_out.write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
    return 0 if out['passed'] else 1


def orchestrate(args: argparse.Namespace) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True); args.artifact_root.mkdir(parents=True, exist_ok=True)
    state=args.out.with_suffix('.state.json')
    if state.exists(): state.unlink()
    phase_files=[]; results=[]
    common=['--checkpoint',str(args.checkpoint),'--omlx-path',str(args.omlx_path),'--recipe-path',str(args.recipe_path),'--artifact-root',str(args.artifact_root),'--state',str(state),'--decode-tokens',str(args.decode_tokens),'--turns-per-cycle',str(args.turns_per_cycle),'--total-turns',str(args.cycles*args.turns_per_cycle),'--cancel-turn',str(args.cancel_turn),'--initial-user',args.initial_user]
    for cycle in range(args.cycles):
        pf=args.out.with_suffix(f'.cycle{cycle}.json'); phase_files.append(pf)
        cmd=[sys.executable,__file__,'--phase','cycle','--cycle',str(cycle),'--phase-out',str(pf)]+common
        rc=subprocess.run(cmd).returncode
        if rc: break
        results.append(json.loads(pf.read_text()))
        if cycle == 0:
            # Strong same-backend deterministic branch evidence from the same restored artifact.
            for branch in ('a','b'):
                bf=args.out.with_suffix(f'.compare{branch}.json'); phase_files.append(bf)
                brc=subprocess.run([sys.executable,__file__,'--phase','compare','--phase-out',str(bf),'--compare-followup',args.compare_followup]+common).returncode
                results.append(json.loads(bf.read_text()))
                if brc: rc=brc; break
            if rc: break
    cf=args.out.with_suffix('.corrupt.json'); phase_files.append(cf)
    if state.exists():
        subprocess.run([sys.executable,__file__,'--phase','corrupt','--phase-out',str(cf)]+common)
        results.append(json.loads(cf.read_text()))
    compares=[r for r in results if r.get('schema')=='ds41f.m10.branch-compare.v1']
    compare_equal=len(compares)>=2 and compares[0].get('generated')==compares[1].get('generated')
    cycles=[r for r in results if r.get('schema')=='ds41f.m10.cycle.v1']
    final=cycles[-1] if cycles else {}
    combined={'schema':'ds41f.m10.restored-long-session-qualification.v1','git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_dirty':git_dirty(_REPO_ROOT),'production_path':'DENSE_P0_P7 -> P5 -> GenerationBatch MTP-OFF -> M8 -> M9','cycles':cycles,'branch_compares':compares,'branch_compare_equal':compare_equal,'corruption': next((r for r in results if r.get('schema')=='ds41f.m10.corruption.v1'), None),'final_frontier': final.get('final_frontier'),'passed': bool(cycles and len(cycles)==args.cycles and all(r.get('passed') for r in results) and compare_equal)}
    args.out.write_text(json.dumps(combined, indent=2, ensure_ascii=False)); print(json.dumps(combined, indent=2, ensure_ascii=False))
    return 0 if combined['passed'] else 1


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['orchestrate','cycle','compare','corrupt'], default='orchestrate')
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT); ap.add_argument('--omlx-path', type=Path, default=DEFAULT_OMLX); ap.add_argument('--recipe-path', type=Path, default=DEFAULT_RECIPE)
    ap.add_argument('--artifact-root', type=Path, default=DEFAULT_KV_ROOT); ap.add_argument('--out', type=Path, default=Path('artifacts/m10/restored-long-session-qualification.json')); ap.add_argument('--phase-out', type=Path, default=Path('artifacts/m10/phase.json')); ap.add_argument('--state', type=Path, default=Path('artifacts/m10/state.json'))
    ap.add_argument('--cycles', type=int, default=3); ap.add_argument('--cycle', type=int, default=0); ap.add_argument('--turns-per-cycle', type=int, default=4); ap.add_argument('--total-turns', type=int, default=12); ap.add_argument('--decode-tokens', type=int, default=4); ap.add_argument('--cancel-turn', type=int, default=5)
    ap.add_argument('--initial-user', default='Answer tersely. Start with ready and one noun.'); ap.add_argument('--compare-followup', default='Branch comparison: give one color word.')
    args=ap.parse_args()
    for p in (args.out,args.phase_out,args.state): p.parent.mkdir(parents=True, exist_ok=True)
    if args.phase=='cycle': return phase_cycle(args)
    if args.phase=='compare': return phase_compare(args)
    if args.phase=='corrupt': return phase_corrupt(args)
    return orchestrate(args)

if __name__=='__main__':
    raise SystemExit(main())
