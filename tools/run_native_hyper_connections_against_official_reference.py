#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_hyper_connections_fixture import DEFAULT_CHECKPOINT, mmap, wm, shard, split_sinkhorn, hc_mixes, hc_pre, hc_post, digest, VOCAB, DIM, HC, MIX, HCD
try:
    from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
except Exception:
    compile_native_prefill_library = load_native_prefill_library = None
REF='artifacts/hyper-connections-official-reference-fixture.json'
DS4_AUTHORITY_REMOTE='https://github.com/antirez/ds4.git'; DS4_AUTHORITY_SHA='0aaea5a238fb41a35106a551e73c8409dfb751ac'

def cmp(a,e,tol,floaty=False):
    aa=np.asarray(a); ee=np.asarray(e)
    if floaty:
        d=np.abs(aa.astype(np.float32)-ee.astype(np.float32)) if aa.shape==ee.shape else np.array([1e30],np.float32); m=float(d.max()) if d.size else 0.0
        return {'shape_matches':aa.shape==ee.shape,'max_abs_diff':m,'within_tolerance':aa.shape==ee.shape and m<=tol,'bit_exact':aa.shape==ee.shape and np.array_equal(aa,ee)}
    d=np.abs(aa.astype(np.int64)-ee.astype(np.int64)) if aa.shape==ee.shape else np.array([2**31]); m=int(d.max()) if d.size else 0
    return {'shape_matches':aa.shape==ee.shape,'max_abs_diff':m,'within_tolerance':aa.shape==ee.shape and m<=tol,'bit_exact':aa.shape==ee.shape and m==0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=REF); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-hyper-connections-official-reference-validation.json'); a=ap.parse_args()
    ref=json.loads(Path(a.reference).read_text()); ck=Path(a.checkpoint); cfg=ref['config']; sh=shard(ck,'layers.24.hc_attn_fn')
    fn=np.ascontiguousarray(mmap(sh,'layers.24.hc_attn_fn',np.float32,(MIX,HCD))); base=np.ascontiguousarray(mmap(sh,'layers.24.hc_attn_base',np.float32,(MIX,))); scale=np.ascontiguousarray(mmap(sh,'layers.24.hc_attn_scale',np.float32,(3,)))
    toks=ref['source_tensors']['input']['tokens']; emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM))); x=emb[toks].reshape(1,2,HC,DIM).copy()
    exp=ref['expected']; tol=ref['operation_contract']['predeclared_tolerance']
    sem_mixes=np.asarray(exp['semantic_mixes_f32'],np.float32); sem_pre,sem_post,sem_comb=split_sinkhorn(sem_mixes,scale,base,HC,int(cfg['hc_sinkhorn_iters']),float(cfg['hc_eps']))
    flat,mean_sq,rsqrt,mixes,pre,post,comb=hc_mixes(x,fn,scale,base,float(cfg['norm_eps']),int(cfg['hc_sinkhorn_iters']),float(cfg['hc_eps']))
    pre_out=hc_pre(x,pre); synth=np.asarray(exp['synthetic_sublayer_output_bf16_uint16'],np.uint16); post_out=hc_post(synth,x,post,comb)
    comps={
      'semantic_pre':cmp(sem_pre,np.asarray(exp['semantic_pre_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'semantic_post':cmp(sem_post,np.asarray(exp['semantic_post_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'semantic_comb':cmp(sem_comb,np.asarray(exp['semantic_comb_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'flattened_hc_input':cmp(flat,np.asarray(exp['flattened_hc_input_f32'],np.float32),0,True),
      'normalization_mean_square':cmp(mean_sq,np.asarray(exp['normalization_mean_square_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'normalization_rsqrt':cmp(rsqrt,np.asarray(exp['normalization_rsqrt_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'mix_projection':cmp(mixes,np.asarray(exp['mix_projection_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'pre':cmp(pre,np.asarray(exp['pre_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'post':cmp(post,np.asarray(exp['post_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'comb':cmp(comb,np.asarray(exp['comb_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'hc_pre_output':cmp(pre_out,np.asarray(exp['hc_pre_output_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'hc_post_output':cmp(post_out,np.asarray(exp['hc_post_output_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
    }
    native_version=None; native_available=False
    if compile_native_prefill_library and load_native_prefill_library:
        try:
            native=load_native_prefill_library(compile_native_prefill_library(Path(a.native_out_dir))); native_version=native.version(); native_available=True
        except Exception:
            native_available=False
    rec={'schema':'ds41f.native-hyper-connections-official-reference-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate Boundary 6a Hyper-Connections primitive/mixing semantics; stop before Attention/FFN/MoE','checkpoint':a.checkpoint,'authority':{'reference_fixture':a.reference,'official_checkpoint_raw_bits':a.checkpoint,'native_validation_provider':'Python-orchestrated arithmetic; no new HC performance kernel/fusion'},'reference_fixture':a.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'native_version':native_version,'native_existing_primitive_path_available':native_available,'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'comparison':comps,'digests':{'native_hc_pre_output_sha256':digest(pre_out),'reference_hc_pre_output_sha256':digest(np.asarray(exp['hc_pre_output_bf16_uint16'],np.uint16)),'native_hc_post_output_sha256':digest(post_out),'reference_hc_post_output_sha256':digest(np.asarray(exp['hc_post_output_bf16_uint16'],np.uint16))},'semantic_status':{'official_reference_fixture_used':True,'actual_hc_tensors_used':True,'sinkhorn_validated':all(comps[k]['within_tolerance'] for k in ['semantic_pre','semantic_post','semantic_comb']),'pre_post_comb_validated':all(comps[k]['within_tolerance'] for k in ['pre','post','comb']),'hc_pre_validated':comps['hc_pre_output']['within_tolerance'],'hc_post_residual_mixing_validated':comps['hc_post_output']['within_tolerance'],'attention_ffn_moe_entered':False,'model_semantics_validated':False},'non_claims':ref['non_claims']+['no Attention-to-HC integration','no FFN/MoE','no full Block','no logits/full model','no performance claim']}
    rec['gates']={'reference_not_omlx_derived':ref.get('not_omlx_derived') is True,'reference_classification_expected':ref.get('classification')=='official_reference_derived_independent_arithmetic_contract','actual_hc_tensor_provenance_recorded':bool(ref.get('source_tensors')),'sinkhorn_arithmetic_validated':rec['semantic_status']['sinkhorn_validated'],'pre_post_comb_validated':rec['semantic_status']['pre_post_comb_validated'],'hc_pre_validated':rec['semantic_status']['hc_pre_validated'],'hc_post_residual_mixing_validated':rec['semantic_status']['hc_post_residual_mixing_validated'],'attention_ffn_moe_not_entered':not rec['semantic_status']['attention_ffn_moe_entered'],'full_model_semantics_not_claimed':not rec['semantic_status']['model_semantics_validated']}
    rec['ok']=all(rec['gates'].values())
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
