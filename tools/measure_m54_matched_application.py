"""Post-correctness matched HTTP decode measurement; no arithmetic modifications.

One lane/process, sequential fresh sessions, official checkpoint and standard
routes. Sparse Python profiling observes existing phase boundaries (inclusive
host wall times), never wraps/duplicates target or settlement execution.
"""
import argparse
import asyncio
from collections import defaultdict
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lane', choices=['off','first-party-mtp-development','candidate'], required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=3)
    args = ap.parse_args()
    # Explicit correctness receipt prerequisite; never relabel pre-PASS timings.
    evidence = Path('artifacts/m54-eof/matched-correctness.json')
    assert json.loads(evidence.read_text())['decision'] == 'MATCHED CORRECTNESS PASS'
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    if args.lane == 'candidate':
        # Existing delivered candidate's patched owner, not the unpatched OFF
        # donor checkout. This is benchmark-only explicit object injection.
        cfg = replace(cfg,omlx_path=Path(importlib.util.find_spec('omlx').origin).parent.parent)
    cfg.apply_import_paths()
    import httpx
    import uvicorn
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend
    from ds41f_mlx.serving.server import create_app
    from ds41f_mlx.runtime.accepted_prefix import AcceptedPrefixJournal
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession
    from ds41f_mlx.runtime.target_forward import TargetForwardTransaction
    from ds41f_mlx.runtime.semantic_cycle import SemanticCycleAdapter
    from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
    from ds41f_mlx.runtime.dspark_proposal import DSparkProposalProducer
    from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
    from ds41f_mlx.serving.recipe_publication import publish_recipe_turn
    from ds41f_mlx.serving.response_reservation import ResponseReservation
    backend = (InternalMTPQualificationBackend(runtime_config=cfg) if args.lane == 'candidate'
               else DeepSeekRecipeRuntimeBackend(runtime_config=cfg,execution_strategy=args.lane))
    out = dict(decision='NOT PASS',lane=args.lane,checkpoint=str(cfg.checkpoint_path),omlx=str(cfg.omlx_path),rows=[])
    t0 = time.perf_counter()
    phases = defaultdict(float)
    calls = defaultdict(int)
    frames = {}
    epoch = [None,None]
    mapping = {}
    def register(fn,label): mapping[fn.__code__] = label
    for method,label in [
        (AcceptedPrefixJournal.__init__,'M51_journal_setup_s'),
        (AcceptedPrefixJournal.advance,'target_verify_s'),
        (AcceptedPrefixJournal.complete,'target_verify_s'),
        (AcceptedPrefixJournal.settle,'M51_settlement_s'),
        (TargetGenerationSession.next_token,'protected_target_step_s'),
        (TargetForwardTransaction.forward,'target_forward_s'),
        (DSparkProposalProducer.propose,'proposal_s'),
        (SemanticCycleAdapter._safe,'semantic_preview_authorization_s'),
        (M8LiveContinuationSession.adopt_cycle_reports,'M8_accounting_s'),
        (publish_recipe_turn,'application_publication_s'),
        (ResponseReservation.append,'response_serialization_reservation_s'),
        (ResponseReservation.complete,'response_serialization_reservation_s'),
        (RecipeSemanticGuard.preview,'candidate_semantic_preview_s'),
        (RecipeSemanticGuard.observe_canonical_emit,'candidate_recipe_report_s'),
        (DeepSeekRecipeRuntimeBackend._run_recipe_turn,'normal_decode_epoch'),
        (InternalMTPQualificationBackend._next,'candidate_decode_step_s'),
        (InternalMTPQualificationBackend._settle,'candidate_settle_report_s'),
    ]: register(method,label)
    def profile(frame,event,arg):
        code = frame.f_code
        label = mapping.get(code)
        if label is None:
            if (args.lane != 'candidate' and event == 'call' and code.co_name == '_forward'
                    and code.co_filename.endswith('/model_execution/language.py')):
                raise RuntimeError('hidden diagnostic target execution')
            return
        now = time.perf_counter()
        if event == 'call':
            frames[id(frame)] = (now,label)
            calls[label] += 1
            if label in ('normal_decode_epoch','candidate_decode_step_s') and epoch[0] is None:
                epoch[0] = now
        elif event == 'return':
            item = frames.pop(id(frame),None)
            if item is not None:
                phases[label] += now-item[0]
            if label in ('normal_decode_epoch','candidate_settle_report_s'):
                epoch[1] = now
    def save(phase):
        out.update(phase=phase,elapsed_s=time.perf_counter()-t0)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,indent=2)+'\n')
        print(out['elapsed_s'],phase,flush=True)
    async def run():
        save('load')
        await backend._call(backend.load)
        sock = socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
        base = 'http://127.0.0.1:'+str(sock.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(create_app(backend=backend,runtime_config=cfg),lifespan='off',log_level='warning'))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if task.done(): await task
            await asyncio.sleep(.01)
        await backend._call(lambda:sys.setprofile(profile))
        try:
            async with httpx.AsyncClient(base_url=base,timeout=600) as client:
                # Warm-up excluded; three matched fresh-session samples. Exact
                # body is identical across all lanes, including max output/RNG.
                body = json.dumps(dict(model=cfg.model_id,messages=[
                    dict(role='system',content='A cache stores frequently used data. '*32),
                    dict(role='user',content='Write OK exactly forty times separated by spaces and nothing else.')],
                    temperature=0,reasoning_effort='none',max_tokens=64,stream=False),separators=(',',':')).encode()
                import hashlib
                out['request_sha256'] = hashlib.sha256(body).hexdigest()
                for sample in range(args.repeats+1):
                    rec = await backend.create_stateful_session()
                    phases.clear();calls.clear();frames.clear();epoch[:]=[None,None]
                    begin = time.perf_counter()
                    response = await client.post('/v1/sessions/'+rec.session_id+'/chat/completions',content=body,
                        headers={'Content-Type':'application/json','X-DS41F-Request-Sequence':'1'})
                    latency = time.perf_counter()-begin
                    assert response.status_code == 200,response.text
                    assert epoch[0] is not None and epoch[1] is not None
                    snapshot = dict(phases)
                    count_snapshot = dict(calls)
                    if args.lane == 'candidate':
                        trace = dict(rec.last_turn)
                        tokens = trace['canonical_generated']
                        stats = trace.get('mtp_stats',{})
                        acceptance = dict(offered=sum(stats.get('depth_drafted',[])),accepted=sum(stats.get('depth_accepted',[])))
                        snapshot.update(candidate_proposal_s=stats.get('mtp_head_ms',0)/1000,
                            candidate_target_verify_s=stats.get('backbone_ms',0)/1000,
                            candidate_sampling_s=stats.get('sample_ms',0)/1000,
                            candidate_cache_ops_s=stats.get('cache_ops_ms',0)/1000,
                            prefill_handoff_s=trace['prefill_handoff_s'])
                        frontier = trace['canonical_frontier']
                        assert len(trace['target_offsets']) == 40 and set(trace['target_offsets']) == {frontier}
                        assert trace['prompt_replay'] == trace['full_cache_repack'] == 0
                        assert trace['prediction_retired'] and trace['queue_empty']
                        metrics = stats
                    else:
                        trace = dict(backend.session_traces[-1])
                        turn = rec.last_turn
                        tokens = turn['generated_tokens']
                        frontier = turn['frontier_after_commit']
                        diag = rec.m11.diagnostics()['m8']
                        assert diag['all_cache_offsets_equal_frontier'] and diag['cache_layer_count'] == 40
                        assert diag['total_prompt_replay_count'] == diag['total_full_cache_repack_count'] == 0
                        metrics = getattr(rec.m11,'last_cycle_metrics',None)
                        acceptance = None if metrics is None else dict(offered=metrics['offered'],accepted=metrics['accepted'])
                        snapshot['prefill_handoff_s'] = trace.get('append_seconds',0)
                        if metrics is not None:
                            snapshot['recipe_report_s'] = metrics['report_s']
                            snapshot['adapter_execution_s'] = metrics['execution_s']
                            assert count_snapshot['target_forward_s'] == 1+metrics['planned_target_inputs']
                        else:
                            assert count_snapshot['target_forward_s'] == 1+len(tokens)
                    decode = epoch[1]-epoch[0]
                    row = dict(sample=sample,warmup=sample==0,tokens=len(tokens),generated_tokens=tokens,
                        frontier=frontier,response=response.json(),http_latency_s=latency,
                        end_to_end_decode_s=decode,end_to_end_decode_tok_s=len(tokens)/decode,
                        request_tok_s=len(tokens)/latency,acceptance=acceptance,
                        phase_s=snapshot,phase_calls=count_snapshot,metrics=metrics,
                        replay=0,repack=0,hidden_target_reexecution=0 if args.lane!='candidate' else None)
                    out['rows'].append(row)
                    await backend.close_stateful_session(rec.session_id)
                    save('sample completed')
            out['decision'] = 'MATCHED ORDINARY HTTP MEASUREMENT COMPLETE; compare lane correctness before speed claims'
        finally:
            await backend._call(lambda:sys.setprofile(None))
            server.should_exit=True
            await task
            sock.close()
        child = None if args.lane=='candidate' else getattr(backend._model.language_model,'_ds41f_proposal_child',None)
        admission = None if args.lane=='candidate' else backend._runtime.admission
        if args.lane == 'candidate':
            # Historical candidate.close shuts down its executor. DELETE already
            # settled/retired each owner on the worker; never join that worker
            # from its own thread.
            backend.close()
        else:
            await backend._call(backend.close)
        out['resources'] = dict(parent_retired=None if admission is None else not admission.active,
            child_retired=None if child is None else not child.active,
            receipts=None if child is None else len(child.receipts),producers=None if child is None else len(child.producers),
            active_sessions=len(backend.sessions),candidate_executor_closed=backend._executor._shutdown if args.lane=='candidate' else None)
    try:
        asyncio.run(run())
    except BaseException:
        out['error']=traceback.format_exc()
        raise
    finally:
        backend.close()
        save('complete')


if __name__=='__main__': main()
