#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig, OMLXDecodeSession
from tools.run_m4_omlx_base_decode_qualification import build_prefill_state

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--omlx-path',default=os.environ.get('DS41F_OMLX',str(DEFAULT_OMLX))); ap.add_argument('--out',default='artifacts/m4/prefill-authority-repair/no-replay-admission.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); omlx=Path(a.omlx_path)
    if str(omlx) not in sys.path: sys.path.insert(0,str(omlx))
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator, generation_stream
    rt=OmlxRuntime(OmlxRuntimeConfig(omlx_path=omlx,checkpoint_path=ck,engram_ssd_offload=True,preserve_mtp=False)); model,_=rt.load_model(); lm=model.language_model
    def greedy(logits): return mx.argmax(logits,axis=-1)
    try:
        prefill=build_prefill_state(ck,Path('artifacts/m4/prefill-authority-repair/native'),[0,3],require_ok=True)
        session=OMLXDecodeSession.from_prefill_state(model,prefill.continuation_state,OMLXDecodeConfig(omlx_path=omlx,checkpoint_path=ck,preserve_mtp=False))
        before=[int(c.size()) for c in session.cache]
        bg=BatchGenerator(lm,max_tokens=1,sampler=greedy,completion_batch_size=1,prefill_batch_size=1,prefill_step_size=2048,stream=generation_stream)
        uids=bg.insert([[15]],max_tokens=[1],caches=[session.cache],all_tokens=[[0,3]],samplers=[greedy]); mx.synchronize(bg.stream)
        prompt_forward_tokens=0; gen=[]; final_prompt_cache=None; final_all_tokens=None
        for _ in range(4):
            pr,gr=bg.next(); mx.synchronize(bg.stream)
            prompt_forward_tokens += sum((r.progress[0] if isinstance(r.progress,tuple) else 0) for r in pr if not r.end_of_prompt)
            for r in gr:
                gen.append(int(r.token))
                if r.prompt_cache is not None: final_prompt_cache=r.prompt_cache; final_all_tokens=r.all_tokens
            if gen: break
        if final_prompt_cache is None:
            caches=bg.extract_cache(uids); final_prompt_cache,final_all_tokens=caches[uids[0]]
        after=[int(c.size()) for c in final_prompt_cache]
        rec={'schema':'ds41f.m4.corrected-state-no-replay-admission.v1','checkpoint':str(ck),'omlx_path':str(omlx),'prefill_artifact_ok':prefill.ok,'insert_prompt':[15],'all_tokens_seed':[0,3],'cache_offsets_before_insert':before,'cache_offsets_after_first_generation':after,'all_40_frontiers_before_15_are_2':all(x==2 for x in before),'first_backbone_input':15,'prompt_processing_forwarded_tokens_from_0_3':0,'observed_non_generation_prompt_forward_tokens':prompt_forward_tokens,'generated_tokens':gen,'all_tokens_after':final_all_tokens,'final_source_candidate_topk_generations_expected':{'compress_kv':20,'index_k':20,'candidates':20,'topk_idxs':36},'ok':bool(prefill.ok and all(x==2 for x in before) and prompt_forward_tokens==0 and gen and final_all_tokens[:3]==[0,3,15])}
    finally:
        rt.close()
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print('ok',rec['ok']); return 0 if rec['ok'] else 2
if __name__=='__main__': raise SystemExit(main())
