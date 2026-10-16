#!/usr/bin/env python3
"""Replay the two diagnosis workloads in the exact retained historical wheel.

Use the reconstructed, separately admitted historical environment, never install
historical payloads into the current benchmark environment.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    from ds41f_mlx.mtp_identity import config,inspect
    cfg=config(profile='mtp-serving-v1'); identity=inspect(cfg)
    from fastapi.testclient import TestClient
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    from ds41f_mlx.runtime.mtp_lifecycle import OMLXMTPGenerationSession
    receipt=dict(identity=identity,dependencies={d.metadata['Name']:d.version for d in md.distributions()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=[])
    def save():args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    native=[]
    original=OMLXMTPGenerationSession.next_token
    def next_token(self,**kw):
        result=original(self,**kw);native.append(self.last_native_decode_s);return result
    OMLXMTPGenerationSession.next_token=next_token
    backend=ProductionMTPBackend(runtime_config=cfg);backend.dependency_identity=identity['identity_sha256']
    # Identical rewrite request bytes and ledger prompt reconstructed from the
    # recorded current pass, not a new prompt or historical-source substitution.
    current=json.loads((root/'artifacts/public-performance/diagnostic.json').read_text())
    rewrite=current['rewrite_request']
    # Read the actual initial ledger request via the canonical deterministic
    # fixture text. Request hash equality is checked against the current pass.
    unit=' Ledger: immutable checkpoints preserve committed state; independent branches retain coherent target and draft frontiers.\n'
    intro='Agent maintenance journal for workload 8192. Treat the ledger as reference data.\n'
    task='\nWrite a detailed engineering plan for maintaining this long-running agent session: explain checkpoint ownership, safe branching, cancellation, tool-result reconciliation, and incremental continuation. Give at least twenty numbered recommendations with explanations. Do not copy the ledger.\n'
    from ds41f_mlx.serving.server import load_v41_tokenizer,prepare_request
    tokenizer=load_v41_tokenizer(cfg.recipe_path)
    def body(n):return dict(model='deepseek-v4.1-flash',messages=[dict(role='user',content=intro+unit*n+task)],max_tokens=256,reasoning_effort='none')
    n=8192//len(tokenizer.encode(unit))
    prepared=prepare_request('chat_completions',json.dumps(body(n)).encode(),tokenizer=tokenizer,recipe_path=cfg.recipe_path,checkpoint=cfg.checkpoint_path,ordinary=True)
    n+=(8192-len(prepared.token_ids))//len(tokenizer.encode(unit));ledger=body(n)
    assert hashlib.sha256(json.dumps(ledger,sort_keys=True).encode()).hexdigest()==current['rows'][0]['request_sha256']
    try:
        with TestClient(create_app(profile='mtp-serving-v1',runtime_config=cfg,backend=backend)) as client:
            backend.load()
            for label,request in [('ledger-default-cold',ledger),('ledger-greedy-hit',ledger | dict(temperature=0)),
                    ('rewrite-greedy-warm',rewrite),('rewrite-greedy-1',rewrite),('rewrite-greedy-2',rewrite),
                    ('rewrite-default-1',{k:v for k,v in rewrite.items() if k!='temperature'})]:
                native.clear();print('Running',label,flush=True);start=time.perf_counter()
                response=client.post('/v1/chat/completions',json=request);wall=time.perf_counter()-start
                response.raise_for_status();trace=dict(backend.serving_traces[-1])
                receipt['rows'].append(dict(label=label,request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
                    response=response.json(),trace=trace,wall_s=wall,native_bg_s=sum(native)))
                save();print(label,trace['generated']/trace['decode_s'],trace['mtp_stats'],flush=True)
                assert trace['cache_published'] and trace['prompt_replay']==trace['full_cache_repack']==0
                assert set(trace['target_offsets']+trace['dspark_offsets'])=={trace['canonical_frontier']}
            receipt['status']='PASS';save()
    finally:
        OMLXMTPGenerationSession.next_token=original
        backend.close()


if __name__=='__main__':main()
