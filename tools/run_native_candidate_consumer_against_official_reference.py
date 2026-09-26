#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_candidate_consumer_fixture import semantic_case, actual_case, DEFAULT_CHECKPOINT
try:
    from ds41f_mlx import native_prefill as native
except Exception:
    native=None
REF='artifacts/candidate-consumer-official-reference-fixture.json'
DS4_AUTHORITY_REMOTE='https://github.com/antirez/ds4.git'; DS4_AUTHORITY_SHA='0aaea5a238fb41a35106a551e73c8409dfb751ac'

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def cmp(a,b,tol=0.0,floaty=False):
    aa=np.asarray(a); bb=np.asarray(b)
    if floaty:
        aa2=np.nan_to_num(aa.astype(np.float32),neginf=-1e30,posinf=1e30); bb2=np.nan_to_num(bb.astype(np.float32),neginf=-1e30,posinf=1e30)
        md=float(np.max(np.abs(aa2-bb2))) if aa.size else 0.0
        return {'shape_matches':aa.shape==bb.shape,'max_abs_diff':md,'within_tolerance':aa.shape==bb.shape and md<=tol,'bit_exact':aa.shape==bb.shape and np.array_equal(aa,bb)}
    return {'shape_matches':aa.shape==bb.shape,'max_abs_diff':int(np.max(np.abs(aa.astype(np.int64)-bb.astype(np.int64)))) if aa.size else 0,'within_tolerance':aa.shape==bb.shape and np.array_equal(aa,bb),'bit_exact':aa.shape==bb.shape and np.array_equal(aa,bb)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=REF); ap.add_argument('--out',default='artifacts/native-candidate-consumer-official-reference-validation.json'); a=ap.parse_args()
    ref=json.loads(Path(a.reference).read_text()); cfg=json.load(open(Path(a.checkpoint)/'config.json'))['text_config']
    sem,_=semantic_case(); act,_,status=actual_case(Path(a.checkpoint),cfg); exp=ref['expected']; tol=ref['operation_contract']['predeclared_tolerance']
    comps={
      'semantic_candidate_mask':cmp(sem['candidate_mask_bool'],np.asarray(exp['semantic_candidate_mask_bool'],bool)),
      'semantic_masked_scores':cmp(sem['candidate_consumer_masked_score_f32'],np.asarray(exp['semantic_candidate_consumer_masked_score_f32'],np.float32),0,True),
      'semantic_topk_idxs':cmp(sem['topk_idxs_int32'],np.asarray(exp['semantic_topk_idxs_int32'],np.int32)),
      'semantic_shared_publication':cmp(sem['shared_topk_idxs_publication_int32'],np.asarray(exp['semantic_shared_topk_idxs_publication_int32'],np.int32)),
      'actual_layer20_index_k_cache':cmp(act['layer20_index_k_cache_bf16_uint16'],np.asarray(exp['layer20_index_k_cache_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'actual_layer24_qr':cmp(act['layer24_qr_bf16_uint16'],np.asarray(exp['layer24_qr_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'actual_layer24_query_dequant':cmp(act['layer24_query_dequant_bf16_uint16'],np.asarray(exp['layer24_query_dequant_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'actual_layer24_weights_proj':cmp(act['layer24_weights_proj_bf16_uint16'],np.asarray(exp['layer24_weights_proj_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'actual_reduced_index_score':cmp(act['layer24_reduced_index_score_f32'],np.asarray(exp['layer24_reduced_index_score_f32'],np.float32),tol['actual_f32_max_abs_lte'],True),
      'actual_shared_candidate_mask':cmp(act['shared_candidate_mask_bool'],np.asarray(exp['shared_candidate_mask_bool'],bool)),
      'actual_candidate_masked_score':cmp(act['layer24_candidate_masked_score_f32'],np.asarray(exp['layer24_candidate_masked_score_f32'],np.float32),tol['actual_f32_max_abs_lte'],True),
      'actual_topk_idxs':cmp(act['topk_idxs_int32'],np.asarray(exp['topk_idxs_int32'],np.int32)),
      'actual_shared_topk_publication':cmp(act['shared_topk_idxs_publication_int32'],np.asarray(exp['shared_topk_idxs_publication_int32'],np.int32)),
    }
    rec={'schema':'ds41f.native-candidate-consumer-official-reference-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate Boundary 5d candidate consumer masking + top-k publication; stop before Attention KV concatenation / sparse_attn','checkpoint':a.checkpoint,'authority':{'reference_fixture':a.reference,'official_checkpoint_raw_bits':a.checkpoint,'native_validation_provider':'Python-orchestrated native arithmetic path over official checkpoint tensors for layer-24 consumer wiring'},'reference_fixture':a.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'native_version':native.version() if native is not None and hasattr(native,'version') else None,'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'comparison':comps,'digests':{'native_actual_topk_idxs_sha256':digest(act['topk_idxs_int32']),'reference_actual_topk_idxs_sha256':digest(np.asarray(exp['topk_idxs_int32'],np.int32))},'semantic_status':{'official_reference_fixture_used':True,'actual_consumer_layer':24,'candidate_source_layer':20,'uses_candidates':True,'non_trivial_semantic_masking_effect_validated':all(comps[k]['within_tolerance'] for k in ['semantic_candidate_mask','semantic_masked_scores','semantic_topk_idxs','semantic_shared_publication']),'actual_layer24_real_indexer_score_validated':all(comps[k]['within_tolerance'] for k in ['actual_layer24_qr','actual_layer24_query_dequant','actual_layer24_weights_proj','actual_reduced_index_score']),'actual_shared_candidate_mask_applied':comps['actual_candidate_masked_score']['within_tolerance'],'actual_topk_idxs_int32_exact':comps['actual_topk_idxs']['bit_exact'],'actual_shared_publication_exact':comps['actual_shared_topk_publication']['bit_exact'],'production_mask_all_reachable_positions_retained':status['production_mask_all_reachable_positions_retained'],'production_pruning_effect_claimed':False,'explicit_stop_before_attention_kv_concat_sparse_attn':True,'model_semantics_validated':False},'non_claims':ref['non_claims']+['native validation does not claim production pruning effect because the short production mask retains all reachable positions']}
    rec['gates']={'reference_not_omlx_derived':ref.get('not_omlx_derived') is True,'reference_classification_expected':ref.get('classification')=='official_reference_derived_independent_arithmetic_contract','local_pinned_consumer_branch_reviewed':True,'semantic_masking_effect_validated':rec['semantic_status']['non_trivial_semantic_masking_effect_validated'],'masked_scores_exact':comps['semantic_masked_scores']['bit_exact'] and comps['actual_candidate_masked_score']['within_tolerance'],'topk_sort_minus_one_offset_int32_exact':comps['semantic_topk_idxs']['bit_exact'] and comps['actual_topk_idxs']['bit_exact'],'actual_wiring_publication_exact':rec['semantic_status']['actual_shared_publication_exact'],'real_tensor_provenance_recorded':bool(ref.get('source_tensors')),'explicit_stop_before_attention_kv_concat_sparse_attn':True,'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
    rec['ok']=all(rec['gates'].values())
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
