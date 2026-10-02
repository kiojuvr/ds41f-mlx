#!/usr/bin/env python3
"""One M20 200K endpoint using the M6 worker plus live continuation.

No context ladder. Run under qualification_supervisor in the explicit release
Python environment. The inherited M6 fixture/count convention is preserved:
200000 prefilled prefix tokens, followed by the held-out terminal token.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import time
from tools import run_m6_performance_qualification as gate
from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession


def main():
    import mlx.core as mx
    import psutil
    original_close = OMLXGenerationSession.close
    original_case = gate.run_case
    continuation = {}
    executing_continuation = False
    def empty(gen):
        bg=gen._bg
        return not (bg._generation_batch.uids or bg._prompt_batch.uids or bg._unprocessed_sequences)
    def memory():
        return {"rss_bytes": psutil.Process().memory_info().rss,
                "mlx_active_bytes": mx.get_active_memory(), "mlx_peak_bytes": mx.get_peak_memory(),
                "mlx_cache_bytes": mx.get_cache_memory()}
    def close(gen):
        nonlocal executing_continuation
        if executing_continuation:
            return original_close(gen)
        executing_continuation = True
        sess=None
        try:
            cache, history = gen.extract_final_state("m20_endpoint")
            before=len(history)
            assert before==200000+1+len(gen.generated_tokens)
            assert all(c.size()==before for c in cache)
            continuation.update(frontier_before=before, memory_before=memory())
            t=time.perf_counter(); original_close(gen)
            continuation.update(initial_cleanup_s=time.perf_counter()-t, initial_generator_empty=empty(gen))
            assert continuation['initial_generator_empty']
            sess=M8LiveContinuationSession.from_live_cache(model=gen.model, live_cache=cache,
                                                          token_history=history, config=gen.config)
            suffix=gate.deterministic_tokens(17)
            t=time.perf_counter();sess.begin_turn_from_suffix(suffix,max_tokens=16)
            continuation['append_bootstrap_s']=time.perf_counter()-t
            responses=[]
            t=time.perf_counter()
            while len(responses)<16:
                rep=sess.next_token()
                if rep is None:
                    raise RuntimeError('continuation stopped without a response')
                responses.append(rep.to_json())
                if rep.finish_reason is not None:
                    break
            seconds=time.perf_counter()-t
            active=sess.generation
            sess.ensure_idle('m20_endpoint_continuation')
            continuation.update(decode_s=seconds,decode_tok_s=len(responses)/seconds,
                                responses=responses,diagnostics=sess.diagnostics(),memory_after=memory(),
                                continuation_generator_empty=empty(active),
                                engram_history_present=sess.live_cache[0][6] is not None)
            assert sess.frontier==before+len(suffix)+len(responses)
            assert all(c.size()==sess.frontier for c in sess.live_cache)
            assert empty(active) and sess.total_prompt_replay_count==0 and sess.total_full_cache_repack_count==0
            assert continuation['engram_history_present']
            assert continuation['decode_tok_s']>=15
            t=time.perf_counter(); sess.close()
            continuation['session_cleanup_s']=time.perf_counter()-t
            continuation['passed']=True
            print(json.dumps({'event':'progress','stage':'200k-continuation-complete',
                              'frontier':sess.frontier,'decode_tok_s':continuation['decode_tok_s']}),flush=True)
        finally:
            if sess is not None:
                sess.close()
            original_close(gen)
    OMLXGenerationSession.close=close
    def case(model,lm,mx,args,count):
        assert count==200000, 'M20 endpoint must not run a context ladder'
        assert not lm._config.preserve_mtp and not getattr(lm._config,'ced_prefill',False)
        rec=original_case(model,lm,mx,args,count)
        rec['m20_continuation']=continuation
        rec['gates']['m20_continuation']=continuation.get('passed',False)
        if not all(rec['gates'].values()):rec['status']='FAIL'
        return rec
    gate.run_case=case
    args=['--contexts','200000',*sys.argv[1:]]
    return gate.main(args)


if __name__=='__main__':
    raise SystemExit(main())
