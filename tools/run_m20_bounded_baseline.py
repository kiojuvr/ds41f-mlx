#!/usr/bin/env python3
"""M20 instrumentation over the existing real-recipe M8 continuation gate.

Run separately in each versioned environment, never concurrently. All forwarded
arguments are the M8 CLI. No defaults or production prefill selectors changed.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib.metadata as md
import json
import time
import psutil
from tools import run_m8_long_session_qualification as gate
from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession


def main():
    import mlx.core as mx
    telemetry = {"load": [], "prefill": [], "sessions": [], "cleanup": [],
                 "policy": {"MTP": "OFF", "DSpark": "OFF", "speculation": "OFF", "CED": "OFF",
                            "prefill": "DENSE_P0_P7", "sampler": "argmax"},
                 "python": sys.version, "executable": sys.executable,
                 "packages": {n: md.version(n) for n in ("omlx", "mlx", "mlx-lm", "deepseek-recipe")}}
    proc = psutil.Process()
    def memory():
        return {"rss_bytes": proc.memory_info().rss, "mlx_active_bytes": mx.get_active_memory(),
                "mlx_peak_bytes": mx.get_peak_memory(), "mlx_cache_bytes": mx.get_cache_memory()}
    def empty(bg):
        return not (bg._generation_batch.uids or bg._prompt_batch.uids or bg._unprocessed_sequences)
    orig_load = gate.OmlxRuntime.load_model
    def load(rt):
        t = time.perf_counter(); result = orig_load(rt); mx.synchronize()
        lm = result[0].language_model
        assert not lm._config.preserve_mtp
        assert not getattr(lm._config, "ced_prefill", False)
        telemetry["load"].append({"seconds": time.perf_counter()-t, **memory()})
        return result
    gate.OmlxRuntime.load_model = load
    orig_pre = gate.DwarfStarMLXPrefillSession.prefill
    def pre(obj, ids):
        t=time.perf_counter(); result=orig_pre(obj,ids)
        telemetry["prefill"].append({"seconds": time.perf_counter()-t, "tokens": len(ids), **memory()})
        return result
    gate.DwarfStarMLXPrefillSession.prefill = pre
    orig_start = OMLXGenerationSession.start
    def start(gen, token, **kwargs):
        record={"prefix_tokens": gen.admitted_frontier, "steps": [], "before": memory()}
        gen._m20_record = record
        # Separate backend-local reference observation; copies only slot
        # containers, never repacks tensors or changes production handoff.
        # Done once, outside timed decode and before generator owns cache.
        if not telemetry["sessions"]:
            cache=[c.extract(0) for c in gen.initial_cache]
            logits=gen.language_model(mx.array([[int(token)]],mx.int64),cache=cache)
            expected=int(mx.argmax(logits[:,-1,:],axis=-1).item())
            record["backend_local_reference_token"] = expected
            del cache, logits
        t=time.perf_counter(); orig_start(gen,token,**kwargs)
        record.update(bootstrap_seconds=time.perf_counter()-t, prompt_replay_count=gen.prompt_replay_count,
                      initial_cache_released=not gen.initial_cache)
        assert record['initial_cache_released'] and gen.prompt_replay_count==0
        telemetry["sessions"].append(record)
    OMLXGenerationSession.start = start
    orig_next = OMLXGenerationSession.next_token
    def next_token(gen):
        rep=orig_next(gen)
        if rep is not None:
            rec=gen._m20_record
            if not rec['steps'] and 'backend_local_reference_token' in rec:
                rec['first_token_backend_fidelity'] = rep.token==rec['backend_local_reference_token']
                assert rec['first_token_backend_fidelity']
            rec['steps'].append({"token":rep.token,"latency_s":rep.latency_s,"frontier":gen.token_frontier,
                                 "finish_reason":rep.finish_reason, **memory()})
        return rep
    OMLXGenerationSession.next_token = next_token
    orig_close = OMLXGenerationSession.close
    def close(gen):
        t=time.perf_counter(); orig_close(gen)
        record={"seconds":time.perf_counter()-t,"generator_empty":empty(gen._bg), **memory()}
        telemetry['cleanup'].append(record)
        assert record['generator_empty']
    OMLXGenerationSession.close = close
    code=gate.main()
    # M8 args accept --out followed by a path; require it for isolated evidence.
    out=Path(sys.argv[sys.argv.index('--out')+1])
    record=json.loads(out.read_text());record['m20']=telemetry
    record['passed']=record['passed'] and not record.get('close_error') and all(c['generator_empty'] for c in telemetry['cleanup'])
    out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    return code if record['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
