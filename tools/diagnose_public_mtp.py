#!/usr/bin/env python3
"""Bounded historical-condition controls and synchronized P6 timing.

Run with the unchanged installed public-benchmark wheel and -I. Observational
wrappers call each original exactly once; no math/sampling/topology changes.
"""
import argparse
import dataclasses
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from ds41f_mlx.mtp_identity import config, inspect
    cfg = config(profile='mtp-serving-v1')
    identity = inspect(cfg)
    import mlx.core as mx
    from mlx_lm.generate import generation_stream, BatchGenerator
    from fastapi.testclient import TestClient
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app, load_v41_tokenizer, prepare_request
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    from ds41f_mlx.runtime.mtp_lifecycle import OMLXMTPGenerationSession

    out = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        identity_sha256=identity['identity_sha256'],
        benchmark_package=json.loads((ROOT/'artifacts/public-performance/package.json').read_text()),
        dependencies={d.metadata['Name']:d.version for d in md.distributions()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[])
    def save():
        args.output.write_text(json.dumps(out, indent=2)+'\n')
    save()
    active = {}
    original_execute = DeferredPrefillAppend.execute_all
    original_next = OMLXMTPGenerationSession.next_token

    def execute(self):
        # Synchronize precisely at the P6 transaction boundary, not at TTFT.
        mx.synchronize(generation_stream)
        t0 = time.perf_counter()
        original_execute(self)
        mx.synchronize(generation_stream)
        active.setdefault('prefill_transactions', []).append(dict(
            C=self.plan.C, T=self.plan.T, tokens=self.plan.T-self.plan.C,
            seconds=time.perf_counter()-t0,
            segments=[dict(start=s.start,count=s.count,mode=s.mode.value) for s in self.plan.segments]))

    def next_token(self, **kw):
        t0 = time.perf_counter()
        token = original_next(self, **kw)
        active['owner_calls_s'] = active.get('owner_calls_s', 0.)+time.perf_counter()-t0
        active['native_bg_s'] = active.get('native_bg_s', 0.)+self.last_native_decode_s
        return token

    DeferredPrefillAppend.execute_all = execute
    OMLXMTPGenerationSession.next_token = next_token
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
    def body(messages):
        return dict(model='deepseek-v4.1-flash',messages=messages,max_tokens=256,reasoning_effort='none')
    def ledger(size):
        unit = ' Ledger: immutable checkpoints preserve committed state; independent branches retain coherent target and draft frontiers.\n'
        intro = f'Agent maintenance journal for workload {size}. Treat the ledger as reference data.\n'
        task = '\nWrite a detailed engineering plan for maintaining this long-running agent session: explain checkpoint ownership, safe branching, cancellation, tool-result reconciliation, and incremental continuation. Give at least twenty numbered recommendations with explanations. Do not copy the ledger.\n'
        n = size // len(tokenizer.encode(unit))
        def make(n):
            return body([dict(role='user', content=intro+unit*n+task)])
        request = make(n)
        prepared = prepare_request('chat_completions', json.dumps(request).encode(),tokenizer=tokenizer,
                                   recipe_path=cfg.recipe_path,checkpoint=cfg.checkpoint_path,ordinary=True)
        n += (size-len(prepared.token_ids)) // len(tokenizer.encode(unit))
        return make(n)
    try:
        with TestClient(app) as client:
            backend.load()
            mx.synchronize()
            def run(label, request):
                active.clear()
                print('Running',label,flush=True)
                start=time.perf_counter()
                response=client.post('/v1/chat/completions',json=request)
                wall=time.perf_counter()-start
                response.raise_for_status()
                trace=dict(backend.serving_traces[-1])
                row=dict(label=label,request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
                    response=response.json(),trace=trace,wall_s=wall,observed=dict(active))
                out['rows'].append(row); save()
                assert trace['cache_published'] and not trace['cancelled']
                assert trace['prompt_replay']==trace['full_cache_repack']==0
                assert set(trace['target_offsets']+trace['dspark_offsets'])=={trace['canonical_frontier']}
                print(label,trace['generated']/trace['decode_s'],trace['mtp_stats'],row['observed'],flush=True)
                return response.json()
            for size in (8192,32768,262144):
                request=ledger(size)
                result=run(f'ledger-{size}-default-cold',request)
                assert out['rows'][-1]['trace']['cached_tokens']==0
                old=json.loads((ROOT/'artifacts/public-performance/measurements.json').read_text())
                assert out['rows'][-1]['request_sha256']==old['cases'][(8192,32768,262144).index(size)]['request_sha256']
            request=body(request['messages']+[result['choices'][0]['message'],dict(role='user',
                content='Continuation 1: expand the engineering plan with at least twenty further concrete recommendations, focusing on retained-prefix state coherence and safe tool execution. Do not repeat earlier text.')])
            run('ledger-262144-default-continuation',request)
            request=ledger(8192)
            run('ledger-8192-default-hit',request)
            run('ledger-8192-greedy-hit',request | dict(temperature=0))
            source=(ROOT/'tools/bench_template_reuse.py').read_text()
            rewrite=dict(model=cfg.model_id,temperature=0,max_tokens=384,messages=[dict(role='user',
                content='Edit this Python file: rename the prefix variable to kernel_prefix and update its uses. Return the complete updated file, preserving all other code exactly. No explanation.\n\n```python\n'+source+'\n```')])
            out['rewrite_request']=rewrite
            for label,request in [('rewrite-greedy-warm',rewrite),('rewrite-greedy-1',rewrite),('rewrite-greedy-2',rewrite),
                                  ('rewrite-default-1',{k:v for k,v in rewrite.items() if k!='temperature'}),
                                  ('rewrite-default-2',{k:v for k,v in rewrite.items() if k!='temperature'})]:
                run(label,request)

            # Narrow target-only control in the SAME installed numerical model.
            # Direct BatchGenerator, no production option added or cache reused.
            from omlx.patches.mlx_lm_mtp import prompt_priming
            lm=backend._model.language_model
            request=ledger(8192)
            prepared=prepare_request('chat_completions',json.dumps(request).encode(),tokenizer=tokenizer,
                recipe_path=cfg.recipe_path,checkpoint=cfg.checkpoint_path,ordinary=True)
            for speculative in (True,False):
                print('Direct native control',speculative,flush=True)
                prompt_priming.drop_ctx(lm)
                lm.configure_mtp(speculative,5)
                tokens=[]; state=None
                with mx.stream(generation_stream):
                    bg=BatchGenerator(lm,max_tokens=256,sampler=lambda logits:mx.argmax(logits,axis=-1),
                        completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
                    uid=bg.insert([list(prepared.token_ids)],max_tokens=[256])[0]
                    t0=time.perf_counter()
                    while True:
                        pr,gr=bg.next(); mx.synchronize(generation_stream)
                        assert not gr
                        if any(x.end_of_prompt for x in pr): break
                    prefill=time.perf_counter()-t0
                    t0=time.perf_counter()
                    for _ in range(256):
                        pr,gr=bg.next(); mx.synchronize(generation_stream)
                        assert len(gr)==1
                        token=gr[0]; tokens.append(int(token.token))
                        state=getattr(bg._generation_batch,'_omlx_mtp_state',None) or state
                        if token.finish_reason: break
                    elapsed=time.perf_counter()-t0
                    stats=None if state is None else dataclasses.asdict(state.stats)
                    bg.remove([uid]); bg.close()
                out.setdefault('direct_controls',[]).append(dict(speculative=speculative,
                    prompt_tokens=len(prepared.token_ids),prompt_ids_sha256=hashlib.sha256(json.dumps(list(prepared.token_ids)).encode()).hexdigest(),
                    generated_tokens=len(tokens),tokens=tokens,prefill_s=prefill,decode_s=elapsed,
                    tok_s=len(tokens)/elapsed,stats=stats,sampling='greedy',reasoning='none'))
                save()
            out['status']='PASS'; save()
    finally:
        DeferredPrefillAppend.execute_all=original_execute
        OMLXMTPGenerationSession.next_token=original_next
        backend.close()


if __name__=='__main__':
    main()
