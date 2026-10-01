#!/usr/bin/env python3
"""Qualification-only real P6 long matrix runner; no benchmark promotion.

This runner records post-segment progress/timing and enforces only post-segment
pathological low-throughput checks. It is not an in-segment no-progress watchdog;
use tools/qualification_supervisor.py for future P7/P8 stuck-worker protection.
"""
from __future__ import annotations

import argparse, hashlib, json, struct, sys, time, traceback
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult, P6AppendPlanner, SchedulingCoordinator, handoff_to_generation, validate_committed_cache
from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommandKind
from tools.run_prefill_fp8_mlx_p6_smoke import qualification_preflight, runtime_authority


def token(i:int)->int: return 16 + ((37*i) % 4096)
def tokens(n:int): return [token(i) for i in range(n)]
def digest(ids):
    h=hashlib.sha256()
    for x in ids: h.update(struct.pack('<i', int(x)))
    return h.hexdigest()

def fronts(cache): return [int(c.size()) for c in cache]
def geom(x): return None if x is None else {'shape': list(getattr(x,'shape',())), 'dtype': str(getattr(x,'dtype',None))}
def cache_geom(cache):
    return {str(i): [geom(cache[i][s]) for s in range(7)] for i in (0,2,8,14,20,39)}

def scan_type_name(root, name):
    stack=[root]; seen=set()
    while stack:
        obj=stack.pop(); oid=id(obj)
        if oid in seen: continue
        seen.add(oid)
        if type(obj).__name__ == name: return True
        if isinstance(obj,(str,bytes,int,float,bool,type(None))): continue
        if isinstance(obj,dict): stack.extend(obj.values())
        elif isinstance(obj,(list,tuple,set)): stack.extend(obj)
        else: stack.extend(getattr(obj,'__dict__',{}).values())
    return False

def real_p7_preflight(model, lm, storage):
    layers={}
    for layer_id in getattr(lm._config,'engram_layer_ids',(1,14)):
        layer=lm.layers[int(layer_id)]
        wrapper=getattr(layer,'engram',None); embed=getattr(wrapper,'embed',None)
        layers[str(layer_id)]={'wrapper_class':type(wrapper).__module__+'.'+type(wrapper).__name__, 'embed_class':type(embed).__module__+'.'+type(embed).__name__, 'embed_is_disk':isinstance(embed, storage.DiskEngramEmbedding), 'embed_resident_is_none':getattr(embed,'_resident',None) is None}
        if not layers[str(layer_id)]['embed_is_disk'] or not layers[str(layer_id)]['embed_resident_is_none']:
            raise RuntimeError('real P7 preflight failed Engram SSD topology')
    if getattr(lm,'_engram_prefetch',None) is None: raise RuntimeError('missing language_model._engram_prefetch')
    if getattr(model,'_moe_offload_plan',None) is not None: raise RuntimeError('outer model has _moe_offload_plan')
    if scan_type_name(lm,'OffloadedExpert'): raise RuntimeError('OffloadedExpert installed')
    coord=SchedulingCoordinator(lm); coord.revoke()
    return {'model_donor_class':type(lm._engram_prefetch).__module__+'.'+type(lm._engram_prefetch).__name__, 'layers':layers, 'outer_moe_offload_plan':None, 'offloaded_expert_present':False, 'coordinator_admission':'PASS'}

@contextmanager
def instrument_engram_reads(lm, storage, report):
    embed_to_layer={id(lm.layers[int(i)].engram.embed): int(i) for i in getattr(lm._config,'engram_layer_ids',(1,14))}
    original=storage.DiskEngramEmbedding._read_rows
    reads=[]
    def wrapped(self, host):
        import threading
        kind='background_prefetch' if threading.current_thread().name.startswith('v41-engram') else 'foreground_fallback'
        reads.append({'kind':kind,'layer':embed_to_layer.get(id(self)),'rows':int(getattr(host,'size',0))})
        return original(self, host)
    with patch.object(storage.DiskEngramEmbedding, '_read_rows', wrapped):
        try:
            yield reads
        finally:
            report['engram_read_instrumentation']={'reads':reads,'background_prefetch_reads':sum(1 for r in reads if r['kind']=='background_prefetch'),'foreground_fallback_reads':sum(1 for r in reads if r['kind']=='foreground_fallback')}


def segment_diag(app, seg, execn, elapsed):
    recs = execn.runner.records if execn is not None else []
    last = recs[-1] if recs else None
    p7_events = []
    if execn is not None and getattr(execn.runner, 'scheduling_coordinator', None) is not None:
        p7_events = list(execn.runner.scheduling_coordinator.telemetry.events)
    return {
        'seq': seg.seq, 'mode': seg.mode.value, 'start': seg.start, 'count': seg.count,
        'elapsed_wall_s': elapsed, 'effective_tokens_per_s': (seg.count / elapsed if elapsed > 0 else None),
        'C': app.C, 'E': app.E, 'D': app.D, 'T': app.T,
        'last_completed_command': None if last is None else getattr(last.kind, 'value', str(last.kind)),
        'last_completed_layer': None if last is None else last.layer,
        'records': len(recs),
        'p7': {
            'enabled': bool(p7_events),
            'prefetch_submissions': sum(1 for e in p7_events if e.get('event') == 'engram_prefetch_submit'),
            'logical_match_consumptions': sum(1 for e in p7_events if e.get('event') == 'engram_consume' and e.get('logical_match')),
            'donor_observed_consumptions': sum(1 for e in p7_events if e.get('event') == 'engram_consume' and e.get('donor_issue_observed')),
            'fallback_mismatches': sum(1 for e in p7_events if e.get('event') == 'engram_prefetch_mismatch'),
            'drains': sum(1 for e in p7_events if e.get('event') == 'engram_prefetch_drain'),
            'read_ahead_already_ready': sum(1 for e in p7_events if e.get('event') == 'ssd_read_ahead' and e.get('result') == 'ALREADY_READY'),
        },
        'source_generations': dict(app.source_generation_counts),
        'public_frontiers_head': fronts(app.live_cache)[:4],
    }

def execute_with_watchdog(app, report, *, baselines=None):
    if app.state.value == 'valid': app.begin()
    report['segments'] = []
    from ds41f_mlx.prefill_fp8_mlx.block_runner import OfficialFP8MLXBlockRunner
    original_execute_command = OfficialFP8MLXBlockRunner.execute_command
    progress_kinds = {SweepCommandKind.ENCODE_ROWS, SweepCommandKind.END_LAYER, SweepCommandKind.P6_SOURCE_COMPLETE_AND_DETACH_CONE, SweepCommandKind.CHECKPOINT_MAY_COMMIT}
    for seg in app.plan.segments:
        def heartbeat_execute_command(runner, command, arena):
            result = original_execute_command(runner, command, arena)
            if command.kind in progress_kinds:
                rec = runner.records[-1] if runner.records else None
                msg={'event':'progress','case':report.get('case'),'segment':seg.seq,'command_kind':command.kind.value,'layer':command.layer,'offset':command.offset,'rows':command.rows,'absolute_start':getattr(rec,'absolute_start',None),'C':app.C,'E':app.E,'D':app.D,'T':app.T,'monotonic':time.monotonic()}
                print(json.dumps(msg), flush=True)
            return result
        t0=time.monotonic()
        with patch.object(OfficialFP8MLXBlockRunner, 'execute_command', heartbeat_execute_command):
            execn=app.execute_segment(seg)
        elapsed=time.monotonic()-t0
        d=segment_diag(app, seg, execn, elapsed); report['segments'].append(d); print(json.dumps({'segment': d}), flush=True)
        if seg.count >= 4096 and d['effective_tokens_per_s'] is not None and d['effective_tokens_per_s'] < 10:
            raise RuntimeError(f"watchdog: pathological throughput at segment {seg.seq}: {d['effective_tokens_per_s']:.3f} tok/s")
        if baselines:
            base = baselines.get(seg.mode.value)
            if base and elapsed > 4.0 * base * (seg.count / 16384.0):
                raise RuntimeError(f"watchdog: segment {seg.seq} exceeded 4x normalized precursor")
    app.final_seal()


def p5_check(lm, model, lang, checkpoint, app, ids, report, *, generated=2):
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
    T=len(ids); terminal=token(T)
    result=LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=ids)
    forwarded=[]; orig=type(lm)._forward
    def forward(obj, input_ids, cache=None, *a, **kw):
        if obj is lm:
            try: v=input_ids.tolist()
            except Exception: v=str(input_ids)
            forwarded.append({'input_ids': v, 'frontiers_before': fronts(cache) if cache is not None else None})
        return orig(obj, input_ids, cache, *a, **kw)
    cfg=OMLXDecodeConfig(omlx_path=Path(lang.__file__).resolve().parents[3], checkpoint_path=checkpoint, preserve_mtp=False, engram_ssd_offload=True)
    session=None
    try:
        with patch.object(type(lm), '_forward', forward):
            session=handoff_to_generation(result, model, terminal_prompt_token=terminal, config=cfg, max_tokens=4)
            assert list(session.active_cache_offsets()) == [T+1]*40
            out=[]; f=[]
            for step in range(generated):
                r=session.next_token(); assert r is not None
                out.append(r.token); f.append(list(session.active_cache_offsets()))
                assert f[-1] == [T+2+step]*40
                if r.finish_reason is not None: break
        report['p5']={'status':'PASS','prompt_replay_count':session.prompt_replay_count,'forward_count':len(forwarded),'generated_tokens':out,'generated_frontiers':f}
    finally:
        if session is not None: session.close()

def make_app(lm, mx, ids, *, C=0, deferral=True, cache=None):
    cache = lm.make_cache() if cache is None else cache
    if deferral:
        return DeferredPrefillAppend.create(lm, cache, ids, committed_frontier=C, mx=mx)
    plan=P6AppendPlanner().matched_control_plan(C=C, T=len(ids))
    return DeferredPrefillAppend(language_model=lm, live_cache=cache, request_token_history=tuple(ids), plan=plan, mx=mx)

def compare_p7_control(candidate, control, report):
    import mlx.core as mx
    cc, ck = candidate.live_cache, control.live_cache
    cmp={'frontiers_equal':fronts(cc)==fronts(ck),'cache_geom_equal':cache_geom(cc)==cache_geom(ck),'engram_history_geom_equal':geom(cc[0][6])==geom(ck[0][6]),'source_coverage_equal':dict(candidate.coverage.source_by_layer)==dict(control.coverage.source_by_layer),'layer20_slot2_exact':None,'layer20_slot3_exact':None}
    for slot,name in ((2,'layer20_slot2_exact'),(3,'layer20_slot3_exact')):
        a,b=cc[20][slot], ck[20][slot]
        cmp[name]= bool(a.shape==b.shape and a.dtype==b.dtype and mx.all(a==b).item())
    report['p7_scheduling_control_comparison']=cmp
    if not all(cmp.values()):
        raise RuntimeError('P7 scheduling-disabled control comparison failed')


def compare_A(candidate, control, report):
    import mlx.core as mx
    cc, ck = candidate.live_cache, control.live_cache
    cmp={'frontiers_equal': fronts(cc)==fronts(ck), 'layer20_slot2_exact': None, 'layer20_slot3_exact': None,
         'candidate_geom': cache_geom(cc), 'control_geom': cache_geom(ck),
         'source_generation_counts': [dict(candidate.source_generation_counts), dict(control.source_generation_counts)]}
    for slot,name in ((2,'layer20_slot2_exact'),(3,'layer20_slot3_exact')):
        a,b=cc[20][slot], ck[20][slot]
        cmp[name]= bool(a.shape==b.shape and a.dtype==b.dtype and mx.all(a==b).item())
    report['matched_control_comparison']=cmp
    if not (cmp['frontiers_equal'] and cmp['layer20_slot2_exact'] and cmp['layer20_slot3_exact']):
        raise RuntimeError('matched control comparison failed at layer20 source/index state')

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--case', required=True, choices=['tiny-16385','A-24577','A-control','B-fresh-49155','B-continued','failure-rebuild','p7-complete-2048','p7-complete-8192','p7-complete-16384','p7-pending-16384','p7-A-24577','p7-A-scheduling-control','p7-failure-drain'])
    ap.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
    ap.add_argument('--out', type=Path)
    ap.add_argument('--p7-overlap', action='store_true', help='qualification-only: enable P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM scheduling')
    args=ap.parse_args(argv)
    report={'case':args.case,'status':'FAIL','checkpoint':str(args.checkpoint),'p7_overlap_enabled':bool(args.p7_overlap)}
    model=None
    try:
        import mlx.core as mx
        import inspect
        import omlx.patches.deepseek_v41.language as lang
        import omlx.patches.deepseek_v41.storage as storage
        from omlx.patches.deepseek_v41.loading import load
        report['environment_preflight']=qualification_preflight(lang,args.checkpoint)
        if report['environment_preflight']['mlx_version']!='0.32.2': raise RuntimeError('MLX qualification version mismatch')
        report['runtime_authority']=runtime_authority(lang)
        report['stage']='checkpoint_load'
        load_kwargs={'preserve_mtp':False,'engram_ssd_offload':True}
        if 'moe_expert_offload_resident_fraction' in inspect.signature(load).parameters:
            load_kwargs['moe_expert_offload_resident_fraction']=None
        report['loader_args']=dict(load_kwargs)
        model,_=load(args.checkpoint,**load_kwargs); lm=model.language_model
        report['real_p7_preflight']=real_p7_preflight(model,lm,storage)
        if args.p7_overlap or args.case.startswith('p7-') and args.case not in {'p7-A-scheduling-control'}:
            lm._p7_enable_overlap = True
        def run_app(name, app, ids, *, p5=True):
            report['plan']=[(s.start,s.count,s.mode.value) for s in app.plan.segments]
            report['token_digest_sha256_int32le']=digest(ids)
            with instrument_engram_reads(lm, storage, report):
                execute_with_watchdog(app, report)
            report.setdefault('engram_read_runs', []).append({'name':name, **report.get('engram_read_instrumentation', {})})
            report['frontiers_after_seal']=fronts(app.live_cache); report['cache_geom']=cache_geom(app.live_cache)
            validate_committed_cache(app.commit_certificate, ids)
            report['commit']={'C':app.C,'E':app.E,'D':app.D,'T':app.T,'history_position':app.engram_history_position,'source_coverage':dict(app.coverage.source_by_layer),'full_cache_repack_count':app.final_execution.runner.full_cache_repack_count,'exported':bool(app.final_execution.runner.prefill_continuation_exported)}
            if p5: p5_check(lm, model, lang, args.checkpoint, app, ids, report)
            return app
        if args.case=='p7-complete-2048':
            lm._p7_enable_overlap=True; ids=tokens(2048); app=make_app(lm,mx,ids); run_app(args.case,app,ids)
            assert report['frontiers_after_seal']==[2048]*40
            assert report['p5']['prompt_replay_count']==0 and report['commit']['full_cache_repack_count']==0 and not report['commit']['exported']
            if report['engram_read_instrumentation']['background_prefetch_reads'] <= 0: raise RuntimeError('P7 donor background reads not observed')
        elif args.case=='p7-complete-8192':
            lm._p7_enable_overlap=True; ids=tokens(8192); app=make_app(lm,mx,ids); run_app(args.case,app,ids)
            assert report['frontiers_after_seal']==[8192]*40
            if report['engram_read_instrumentation']['background_prefetch_reads'] < 2: raise RuntimeError('expected both Engram tables to prefetch')
            if report['engram_read_instrumentation']['foreground_fallback_reads'] != 0: raise RuntimeError('unexpected foreground Engram fallback')
        elif args.case=='p7-complete-16384':
            lm._p7_enable_overlap=True; ids=tokens(16384); app=make_app(lm,mx,ids); run_app(args.case,app,ids)
            assert report['frontiers_after_seal']==[16384]*40
            if report['engram_read_instrumentation']['background_prefetch_reads'] < 4: raise RuntimeError('expected two chunks for both Engram tables')
            if report['engram_read_instrumentation']['foreground_fallback_reads'] != 0: raise RuntimeError('unexpected foreground Engram fallback')
        elif args.case=='p7-pending-16384':
            lm._p7_enable_overlap=True; ids=tokens(24577); app=make_app(lm,mx,ids); app.begin(); seg=app.plan.segments[0]
            with instrument_engram_reads(lm, storage, report):
                from ds41f_mlx.prefill_fp8_mlx.block_runner import OfficialFP8MLXBlockRunner
                t0=time.monotonic(); execn=app.execute_segment(seg); elapsed=time.monotonic()-t0
            d=segment_diag(app,seg,execn,elapsed); report['segments']=[d]; print(json.dumps({'segment':d}), flush=True)
            assert (app.C,app.E,app.D)==(0,16384,0)
            assert fronts(app.live_cache)==[0]*40
            rejected={}
            try: LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=ids)
            except Exception as e: rejected['p5']=str(e)
            try:
                from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
                OMLXGenerationSession.from_prefilled_cache(model, app.live_cache, [])
            except Exception as e: rejected['raw_generation']=str(e)
            if 'p5' not in rejected or 'raw_generation' not in rejected: raise RuntimeError('pending checkpoint unexpectedly admissible')
            report['pending_16384']={'C':app.C,'E':app.E,'D':app.D,'rejections':rejected,'donor_pending_after_close':getattr(lm._engram_prefetch,'_pending',None) is not None}
            if report['pending_16384']['donor_pending_after_close']: raise RuntimeError('donor pending survived source-only close')
            nxt=app._make_segment_execution(app.plan.segments[1]); nxt.runner.close(); report['pending_16384']['next_runner_reused_donor']=True
        elif args.case=='p7-A-24577':
            lm._p7_enable_overlap=True; ids=tokens(24577); app=make_app(lm,mx,ids); run_app(args.case,app,ids)
            assert report['plan']==[(0,16384,'ENCODER_SOURCE_ONLY'),(16384,8192,'FINAL_ENCODER_DECODER'),(24576,1,'ORDINARY_COMPLETE_RANGE')]
            assert report['segments'][0]['E']==16384 and report['segments'][0]['D']==0 and report['segments'][0]['public_frontiers_head']==[0]*4
            assert report['segments'][1]['E']==24576 and report['segments'][1]['D']==24576
            if report['engram_read_instrumentation']['background_prefetch_reads'] < 6: raise RuntimeError('expected source and completing segment Engram prefetch')
        elif args.case=='p7-A-scheduling-control':
            ids=tokens(24577)
            lm._p7_enable_overlap=True; cand=run_app('p7-on', make_app(lm,mx,ids), ids, p5=False)
            lm._p7_enable_overlap=False; control=run_app('p7-off', make_app(lm,mx,ids), ids, p5=False)
            compare_p7_control(cand, control, report); p5_check(lm, model, lang, args.checkpoint, cand, ids, report)
        elif args.case=='p7-failure-drain':
            lm._p7_enable_overlap=True; ids=tokens(24577); app=make_app(lm,mx,ids); app.begin(); first=app._make_segment_execution(app.plan.segments[0]); first.runner.scheduling_coordinator.set_command_stream(app.plan.segments[0].commands); first.runner.execute_command(app.plan.segments[0].commands[0], first.arena); first.runner.execute_command(app.plan.segments[0].commands[1], first.arena); app.active_execution=first; app.fail('qualification injected P7 in-flight failure')
            rejected={}
            try: first.runner.execute_command(app.plan.segments[0].commands[2], first.arena)
            except Exception as e: rejected['old_runner']=str(e)
            if getattr(lm._engram_prefetch,'_pending',None) is not None: raise RuntimeError('donor pending survived failure')
            fresh=make_app(lm,mx,tokens(2048)); run_app('fresh-after-failure', fresh, tokens(2048), p5=False)
            report['failure_drain']={'failed_state':app.state.value,'rejections':rejected,'fresh_frontiers':fronts(fresh.live_cache)}
        elif args.case=='tiny-16385':
            ids=tokens(16385); app=make_app(lm,mx,ids); run_app(args.case,app,ids)
            assert report['plan']==[(0,16384,'FINAL_ENCODER_DECODER'),(16384,1,'ORDINARY_COMPLETE_RANGE')]
        elif args.case=='A-24577':
            ids=tokens(24577); app=make_app(lm,mx,ids); run_app(args.case,app,ids)
            assert report['plan']==[(0,16384,'ENCODER_SOURCE_ONLY'),(16384,8192,'FINAL_ENCODER_DECODER'),(24576,1,'ORDINARY_COMPLETE_RANGE')]
            assert report['segments'][0]['E']==16384 and report['segments'][0]['D']==0 and report['segments'][0]['public_frontiers_head']==[0]*4
            assert report['segments'][1]['E']==24576 and report['segments'][1]['D']==24576
        elif args.case=='A-control':
            ids=tokens(24577); cand=run_app('candidate', make_app(lm,mx,ids), ids, p5=False)
            control=run_app('control', make_app(lm,mx,ids,deferral=False), ids, p5=False)
            compare_A(cand, control, report); p5_check(lm, model, lang, args.checkpoint, cand, ids, report)
        elif args.case=='B-fresh-49155':
            ids=tokens(49155); app=make_app(lm,mx,ids); run_app(args.case,app,ids)
            assert report['segments'][0]['E']==16384 and report['segments'][0]['D']==0 and report['segments'][0]['public_frontiers_head']==[0]*4
            assert report['segments'][1]['E']==32768 and report['segments'][1]['D']==0 and report['segments'][1]['public_frontiers_head']==[0]*4
        elif args.case=='B-continued':
            seed_ids=tokens(24578); seed=make_app(lm,mx,seed_ids,deferral=False); run_app('seed-control-24578', seed, seed_ids, p5=False)
            ids=tokens(49155); app=DeferredPrefillAppend.continue_from_commit(lm, seed.commit_certificate, ids, mx=mx)
            report['seed']={'frontiers':fronts(seed.live_cache),'token_digest_sha256_int32le':digest(seed_ids)}
            run_app(args.case, app, ids)
            assert report['segments'][0]['E']==40962 and report['segments'][0]['D']==24578 and report['segments'][0]['public_frontiers_head']==[24578]*4
        elif args.case=='failure-rebuild':
            ids=tokens(24577); app=make_app(lm,mx,ids); app.begin(); first=app.execute_segment(app.plan.segments[0]); app.fail('qualification injected cancellation')
            rejected={}
            try: LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=ids)
            except Exception as e: rejected['p5']=str(e)
            try:
                from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession
                OMLXGenerationSession.from_prefilled_cache(model, app.live_cache, [])
            except Exception as e: rejected['raw_generation']=str(e)
            try: first.runner.execute_command(app.plan.segments[0].commands[0], first.arena)
            except Exception as e: rejected['old_runner']=str(e)
            rebuilt=app.rebuild_with_tokens(tokens(129)); execute_with_watchdog(rebuilt, report); validate_committed_cache(rebuilt.commit_certificate, tokens(129))
            report['failure_rebuild']={'failed_state':app.state.value,'rejections':rejected,'rebuilt_frontiers':fronts(rebuilt.live_cache)}
        report['status']='PASS'; report['stage']='complete'
    except Exception as e:
        report['reason']=type(e).__name__+': '+str(e); traceback.print_exc()
    finally:
        if model is not None: model.close()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report, indent=2))
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
