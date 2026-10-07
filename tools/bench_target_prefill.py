"""Completed first-party real-model P6 scaling and optional serialized producer probe.

Diagnostic probes add barriers and are not production throughput measurements.
No environment selectors, checkpoint changes, or numerical replacements.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch


def _packed_suffix_attention(lang, mx, q, kv, pooled, idx, ci, sink, *, config, start, old_len, ratio, portable=None):
    """Unselected tool-only candidate; same admitted target primitive/127-row geometry."""
    c=config
    if (
        getattr(lang,'__name__','')=='ds41f_mlx.model_execution.language'
        and q.shape[1]>8 and c.n_heads==64 and c.head_dim==512
        and c.window_size==128 and old_len==127 and q.dtype==mx.bfloat16
        and lang.glm_fast.has_symbol('deepseek_v41_packed_attention')
    ):
        return lang.glm_fast.deepseek_v41_packed_attention(
            q.transpose(0,2,1,3),kv[:,None],pooled,ci[:,None].astype(mx.uint32),
            sink,c.head_dim**-0.5,start,max(ratio,1),c.window_size,
        ).transpose(0,2,1,3)
    return (portable or lang.packed_sparse_attention)(q,kv,pooled,idx,ci,sink,c.head_dim**-0.5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', default='64,128,292,512,1024,2048,4096,8191,8192,16384,32768')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--capture', type=Path, help='Metal capture of final repeat, requires MTL_CAPTURE_ENABLED=1')
    p.add_argument('--dense-replay', action='store_true', help='probe-only exact checkpoint dequantized GEMM diagnostic')
    p.add_argument('--probe-lengths', default='', help='serialized probes after ordinary measurements in same loaded model')
    p.add_argument('--keep-cache', action='store_true', help='retain admitted bounded allocator cache between fresh cases')
    p.add_argument('--continuation', action='store_true', help='measure a real suffix append and eight subsequent outputs')
    p.add_argument('--moe-replay', action='store_true', help='real sorted Expert replay using existing BM8/16/32 variants')
    p.add_argument('--moe-ab', default='', help='append tool-only BM8 crossover A/B lengths after baseline; no decode change')
    p.add_argument('--p6-ab', default='', help='tool-only completing-cone topology below current threshold')
    p.add_argument('--suffix-ab', default='', help='tool-only target native suffix candidate versus unchanged production at these lengths')
    p.add_argument('--decode-tokens',type=int,default=8,help='fixed raw decode steps (no EOS stop IDs), separate from continuation')
    a = p.parse_args()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config(); cfg.apply_import_paths()
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult, handoff_to_generation
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.model_execution import language as lang
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    import subprocess
    source_paths=[Path(__file__),Path('ds41f_mlx/prefill_fp8_mlx/omlx_suffix_math.py'),Path('ds41f_mlx/prefill_fp8_mlx/p6_append.py')]
    result = dict(schema='ds41f.target-prefill.v1', probe=a.probe, keep_cache=a.keep_cache, device=mx.device_info(), rows=[],
                  base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): a.output.write_text(json.dumps(result, indent=2)+'\n')
    def ready(value):
        arrays = [x for _, x in tree_flatten(value) if isinstance(x, mx.array)]
        if arrays: mx.eval(arrays)
        mx.synchronize()
    def memory():
        return dict(active=mx.get_active_memory(), cached=mx.get_cache_memory(), peak=mx.get_peak_memory())
    def digest(cache):
        import numpy as np
        h = hashlib.sha256()
        for _, x in tree_flatten([c.state for c in cache]):
            if isinstance(x, mx.array):
                h.update(str((x.shape,str(x.dtype))).encode()); h.update(np.asarray(x.view(mx.uint8)).tobytes())
        return h.hexdigest()
    try:
        t=time.perf_counter(); backend.load(); result['load_s']=time.perf_counter()-t
        model=backend._model; lm=model.language_model
        assert type(lm).__module__ == 'ds41f_mlx.model_execution.language'
        result['admission']=backend._runtime.admission.describe()
        tok=backend._runtime.processor.tokenizer
        source=tok.encode('Explain why a computer uses a cache. Compare latency, throughput, locality, and memory bandwidth. ' * 4000)
        replay_seen=set()
        prefix_references={}
        cases=[(int(n),a.probe,False,False,False) for n in a.lengths.split(',') if n]
        cases += [(int(n),False,True,False,False) for n in a.moe_ab.split(',') if n]
        cases += [(int(n),False,False,True,False) for n in a.p6_ab.split(',') if n]
        cases += [(int(n),False,False,int(n)<8192,True) for n in a.suffix_ab.split(',') if n]
        cases += [(int(n),True,False,False,False) for n in a.probe_lengths.split(',') if n]
        for n,probe,bm8_control,p6_control,suffix_control in cases:
            ids=source[:n]; assert len(ids)==n
            for repeat in range(a.repeats):
                cache=lm.make_cache(); row=dict(length=n, repeat=repeat, probe=probe, bm8_control=bm8_control, p6_control=p6_control, suffix_control=suffix_control, before=memory(), regions={})
                stack=[]
                opaque_grouped=[False]
                def wrap(cls):
                    original=cls.__call__
                    def call(obj,*args,**kw):
                        # Native one/few-row Expert consumes UNEVALUATED GatherQMM
                        # nodes as a single pipeline. Do not destroy that producer
                        # structure merely to put a stopwatch around its leaves.
                        if cls is lang.Expert and args[0].size // args[0].shape[-1] <= 8:
                            previous=opaque_grouped[0]; opaque_grouped[0]=True
                            try: return original(obj,*args,**kw)
                            finally: opaque_grouped[0]=previous
                        ready((args,kw))
                        key=cls.__name__ + (':routed' if cls is lang.Expert and len(args)>1 and args[1] is not None else ':shared' if cls is lang.Expert else '')
                        t=time.perf_counter(); frame=[0.0]; stack.append(frame)
                        out=original(obj,*args,**kw); ready(out)
                        elapsed=time.perf_counter()-t; stack.pop()
                        if stack: stack[-1][0]+=elapsed
                        r=row['regions'].setdefault(key,dict(calls=0,inclusive_s=0.,exclusive_s=0.,shapes={}))
                        r['calls']+=1; r['inclusive_s']+=elapsed; r['exclusive_s']+=elapsed-frame[0]
                        shape=str(tuple(args[0].shape)); r['shapes'][shape]=r['shapes'].get(shape,0)+1
                        if cls is lang.Expert and len(args)>1 and args[1] is not None:
                            import numpy as np
                            counts=np.bincount(np.asarray(args[1]).reshape(-1).astype(int),minlength=obj.w1.num_experts)
                            bm=32 if args[1].size>=16384 else 16
                            padded=int(np.sum((counts+bm-1)//bm)*bm)
                            r.setdefault('occupancy',[]).append(dict(routes=int(counts.sum()),experts=int((counts>0).sum()),max_rows=int(counts.max()),padded_rows=padded,useful_fraction=float(counts.sum()/padded),bm=bm,format=obj.w1.mode))
                            if a.moe_replay and len(r['occupancy'])<=4 and kw.get('sorted_indices') and obj.w1.mode=='mxfp4':
                                trials=[]; opaque_grouped[0]=True
                                try:
                                    for block,variant in ((8,0),(16,1),(32,2)):
                                        times=[]
                                        with patch.object(lang,'_block_config',lambda count,kind,b=block,v=variant:(b,v)):
                                            for _ in range(4):
                                                dt=time.perf_counter(); candidate=original(obj,*args,**kw); ready(candidate); times.append(time.perf_counter()-dt)
                                        trials.append(dict(bm=block,variant=variant,times_s=times,max_abs_error=mx.max(mx.abs(candidate.astype(mx.float32)-out.astype(mx.float32))).item()))
                                finally: opaque_grouped[0]=False
                                row.setdefault('moe_replay',[]).append(dict(shape=shape,occupancy=r['occupancy'][-1],trials=trials))
                        return out
                    return call
                def leaf(name, original):
                    def call(*args,**kw):
                        if opaque_grouped[0]: return original(*args,**kw)
                        ready((args,kw))
                        t=time.perf_counter(); out=original(*args,**kw); ready(out)
                        elapsed=time.perf_counter()-t
                        if stack: stack[-1][0]+=elapsed
                        r=row['regions'].setdefault(name,dict(calls=0,inclusive_s=0.,exclusive_s=0.,shapes={}))
                        r['calls']+=1; r['inclusive_s']+=elapsed; r['exclusive_s']+=elapsed
                        shape=str([(tuple(x.shape),str(x.dtype)) for x in args if isinstance(x,mx.array)])
                        r['shapes'][shape]=r['shapes'].get(shape,0)+1
                        replay_key=(n,shape,str(kw))
                        if a.dense_replay and name=='quantized_matmul' and replay_key not in replay_seen:
                            replay_seen.add(replay_key)
                            import numpy as np
                            x,w,scales,*biases=args
                            dt=time.perf_counter()
                            dense=mx.dequantize(w,scales,biases[0] if biases else None,group_size=kw['group_size'],bits=kw['bits'],mode=kw['mode']).astype(x.dtype)
                            ready(dense); dequant_s=time.perf_counter()-dt
                            q_times=[]; gemm_times=[]
                            for _ in range(4):
                                dt=time.perf_counter(); q=original(*args,**kw); ready(q); q_times.append(time.perf_counter()-dt)
                                dt=time.perf_counter(); g=mx.matmul(x,dense.T); ready(g); gemm_times.append(time.perf_counter()-dt)
                            error=mx.max(mx.abs(q.astype(mx.float32)-g.astype(mx.float32))).item()
                            row.setdefault('dense_replay',[]).append(dict(shape=shape,kwargs=kw,dequant_s=dequant_s,qmm_s=q_times,gemm_s=gemm_times,max_abs_error=error,dense_bytes=dense.nbytes))
                            del dense,q,g
                        return out
                    return call
                with ExitStack() as scope:
                    if suffix_control:
                        from ds41f_mlx.prefill_fp8_mlx.omlx_suffix_math import OmlxV41SuffixMath
                        original_attention=OmlxV41SuffixMath._attention
                        def suffix_attention(owner,attn,x,cache,shared,start,layer,ops,core):
                            original_packed=ops.packed_sparse_attention
                            def packed(q,window,pooled,wi,ci,sink,scale):
                                return _packed_suffix_attention(ops,core,q,window,pooled,wi,ci,sink,
                                    config=lm._config,start=start,old_len=window.shape[1]-q.shape[1],
                                    ratio=lm._config.compress_ratios[layer],portable=original_packed)
                            with patch.object(ops,'packed_sparse_attention',packed):
                                return original_attention(owner,attn,x,cache,shared,start,layer,ops,core)
                        scope.enter_context(patch.object(OmlxV41SuffixMath,'_attention',suffix_attention))
                    if bm8_control:
                        original_config=lang._block_config
                        scope.enter_context(patch.object(lang,'_block_config',lambda routes,kind:(8,0) if kind=='mxfp4' and routes<=3072 else original_config(routes,kind)))
                    if probe:
                        from omlx.custom_kernels.glm_moe_dsa import fast
                        for name in ('deepseek_v41_packed_attention','deepseek_mxfp4_gather_qmm_pair_concat_blocks','deepseek_mxfp4_gather_qmm_blocks'):
                            scope.enter_context(patch.object(fast,name,leaf(name,getattr(fast,name))))
                        for name in ('quantized_matmul','gather_qmm','einsum'):
                            scope.enter_context(patch.object(mx,name,leaf(name,getattr(mx,name))))
                        for cls in (lang.Attention,lang.MoE,lang.Expert,lang.Gate,lang.Block):
                            scope.enter_context(patch.object(cls,'__call__',wrap(cls)))
                    t=time.perf_counter()
                    app=DeferredPrefillAppend.create(lm,cache,ids,committed_frontier=0,mx=mx)
                    if p6_control:
                        from ds41f_mlx.prefill_fp8_mlx.p6_append import AppendPlan,P6SegmentPlan,SegmentMode,_segment_commands
                        assert 2541<=n<8192
                        mode=SegmentMode.FINAL_ENCODER_DECODER
                        segment=P6SegmentPlan(0,0,n,mode,_segment_commands(seq=0,count=n,mode=mode),completes_decoder=True)
                        app.plan=AppendPlan(0,n,16384,(segment,))
                    capture = a.capture is not None and repeat == a.repeats-1
                    if capture: mx.metal.start_capture(str(a.capture.resolve()))
                    app.execute_all(); row['enqueue_s']=time.perf_counter()-t
                    t1=time.perf_counter(); ready([c.state for c in cache]); row['completion_s']=time.perf_counter()-t1
                    row['prefill_s']=time.perf_counter()-t; row['tokens_per_s']=n/row['prefill_s']
                    if capture: mx.metal.stop_capture()
                row['segments']=[dict(mode=s.mode.value,start=s.start,count=s.count,commands=len(s.commands)) for s in app.plan.segments]
                row['after']=memory()
                if a.p6_ab or a.suffix_ab:
                    import numpy as np
                    snapshot={(i,j):(tuple(x.shape),str(x.dtype),np.asarray(x.view(mx.uint8)).copy()) for i,c in enumerate(cache) for j,x in enumerate(c.cache) if isinstance(x,mx.array)}
                    if not p6_control and not suffix_control: prefix_references[n]=snapshot
                    else:
                        changes=[]
                        reference=prefix_references[n]
                        for (i,j),(shape,dtype,data) in snapshot.items():
                            old_shape,old_dtype,old=reference[i,j]
                            if (shape,dtype)!=(old_shape,old_dtype):
                                changes.append(dict(layer=i,slot=j,shape=shape,reference_shape=old_shape,structural=True)); continue
                            count=int(np.count_nonzero(data!=old))
                            if count:
                                entry=dict(layer=i,slot=j,shape=shape,dtype=dtype,changed_bytes=count,total_bytes=int(data.size))
                                if j==1 and shape[-1]==528:
                                    from ds41f_mlx.model_execution.quantization import unpack_activation
                                    old_values=unpack_activation(mx.array(old.reshape(shape),mx.uint8),dtype=mx.float32)
                                    new_values=unpack_activation(cache[i][j],dtype=mx.float32)
                                    entry['window_max_abs_error']=mx.max(mx.abs(old_values-new_values)).item()
                                changes.append(entry)
                        row['prefix_state_changes']=changes
                        row['source_and_history_exact']=not any(v['slot']!=1 or v['layer']<20 for v in changes)
                live=LivePrefillResult.from_committed(app.commit_certificate,prefix_token_ids=ids)
                t=time.perf_counter()
                session=handoff_to_generation(live,model,terminal_prompt_token=source[n],config=OMLXDecodeConfig(preserve_mtp=False,omlx_path=cfg.omlx_path),max_tokens=a.decode_tokens)
                row['terminal_s']=time.perf_counter()-t
                t=time.perf_counter(); outputs=[r.token for r in session.generate(a.decode_tokens)]; row['decode_s']=time.perf_counter()-t
                settled,history=session.extract_final_state(); ready([c.state for c in settled])
                row['tokens']=outputs; row['state_digest']=digest(settled)
                assert settled is cache and all(c.size()==len(history) for c in cache)
                assert session.prompt_replay_count == app.final_execution.runner.full_cache_repack_count == 0
                session.close()
                if a.continuation:
                    from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
                    continuation=M8LiveContinuationSession.from_live_cache(model=model,live_cache=settled,token_history=history,config=OMLXDecodeConfig(preserve_mtp=False,omlx_path=cfg.omlx_path),mx=mx)
                    suffix=tok.encode(' Summarize the tradeoffs briefly.')
                    dt=time.perf_counter(); continuation.begin_turn_from_suffix(suffix,max_tokens=8); row['continuation_start_s']=time.perf_counter()-dt
                    dt=time.perf_counter(); row['continuation_tokens']=[r.token for r in continuation.generation.generate(8)]; row['continuation_decode_s']=time.perf_counter()-dt
                    continuation.ensure_idle()
                    assert continuation.live_cache is cache
                    assert continuation.total_prompt_replay_count == continuation.total_full_cache_repack_count == 0
                    row['continuation_frontier']=len(continuation.token_history)
                    row['continuation_state_digest']=digest(cache)
                    assert all(c.size()==row['continuation_frontier'] for c in cache)
                    continuation.close(); del continuation
                result['rows'].append(row); save(); print(n,repeat,round(row['prefill_s'],3),round(row['tokens_per_s'],1),flush=True)
                del app,live,session,cache,settled
                mx.synchronize()
                if not a.keep_cache: mx.clear_cache()
        result['status']='PASS'; save()
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc)); save(); raise
    finally: backend.close()

if __name__=='__main__': main()
