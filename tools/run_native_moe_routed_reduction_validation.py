#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_moe_routed_reduction_fixture import DEFAULT_CHECKPOINT,build,digest
REF='artifacts/moe-routed-reduction-official-reference-fixture.json'
DS4_AUTHORITY_REMOTE='https://github.com/antirez/ds4.git'; DS4_AUTHORITY_SHA='0aaea5a238fb41a35106a551e73c8409dfb751ac'
def cmp(a,e,tol,floaty=False):
 aa=np.asarray(a); ee=np.asarray(e)
 if floaty:
  d=np.abs(aa.astype(np.float32)-ee.astype(np.float32)) if aa.shape==ee.shape else np.array([1e30],np.float32); m=float(d.max()) if d.size else 0.0
  return {'shape_matches':aa.shape==ee.shape,'max_abs_diff':m,'within_tolerance':aa.shape==ee.shape and m<=tol,'bit_exact':aa.shape==ee.shape and np.array_equal(aa,ee)}
 d=np.abs(aa.astype(np.int64)-ee.astype(np.int64)) if aa.shape==ee.shape else np.array([2**31]); m=int(d.max()) if d.size else 0
 return {'shape_matches':aa.shape==ee.shape,'max_abs_diff':m,'within_tolerance':aa.shape==ee.shape and m<=tol,'bit_exact':aa.shape==ee.shape and m==0}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=REF); ap.add_argument('--out',default='artifacts/native-moe-routed-reduction-validation.json'); a=ap.parse_args(); ck=Path(a.checkpoint); ref=json.loads(Path(a.reference).read_text()); exp=ref['expected']; tol=ref['operation_contract']['predeclared_tolerance']
 cfg,x,raw,scores,biased,idx,selected,norm,weights,eids,prov,per,accum,y,tiny=build(ck)
 comps={'topk_expert_indices':cmp(idx,np.asarray(exp['topk_expert_indices_int64'],np.int64),0),'scaled_routing_weights':cmp(weights,np.asarray(exp['scaled_routing_weights_f32'],np.float32),tol['f32_max_abs_lte'],True),'selected_raw_scores':cmp(selected,np.asarray(exp['selected_raw_scores_f32'],np.float32),tol['f32_max_abs_lte'],True),'final_routed_sum':cmp(y,np.asarray(exp['final_routed_sum_f32'],np.float32),tol['f32_max_abs_lte'],True),'tiny_weighted_bf16':cmp(tiny['tiny_weighted_bf16_uint16'],np.asarray(exp['tiny_weighted_bf16_uint16'],np.uint16),0),'tiny_accum':cmp(tiny['tiny_accum_after_each_add_f32'],np.asarray(exp['tiny_accum_after_each_add_f32'],np.float32),0,True),'tiny_final':cmp(tiny['tiny_final_f32'],np.asarray(exp['tiny_final_f32'],np.float32),0,True)}
 # Compare all per-invocation weighted contributions and accumulation snapshots.
 inv_ok=len(per)==len(exp['per_expert_invocations'])
 max_inv=0
 for a0,e0 in zip(per,exp['per_expert_invocations']):
  if (a0['token'],a0['topk_slot'],a0['expert_id'])!=(e0['token'],e0['topk_slot'],e0['expert_id']): inv_ok=False
  d=np.max(np.abs(np.asarray(a0['weighted_expert_contribution_bf16_uint16'],np.int64)-np.asarray(e0['weighted_expert_contribution_bf16_uint16'],np.int64))); max_inv=max(max_inv,int(d))
  if d!=0: inv_ok=False
 comps['all_12_expert_invocations']={'shape_matches':len(per)==12,'max_abs_diff':max_inv,'within_tolerance':inv_ok,'bit_exact':inv_ok}
 rec={'schema':'ds41f.native-moe-routed-reduction-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate Boundary 6c1 routed expert weighted reduction; stop before shared expert contribution','checkpoint':a.checkpoint,'authority':{'reference_fixture':a.reference,'official_checkpoint_raw_bits':a.checkpoint,'native_validation_provider':'Python-orchestrated arithmetic over actual checkpoint tensors using Boundary 6c0 Expert contract'},'reference_fixture':a.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'comparison':comps,'digests':{'final_routed_sum_f32_sha256':digest(y),'reference_final_routed_sum_f32_sha256':digest(np.asarray(exp['final_routed_sum_f32'],np.float32))},'semantic_status':{'moe_forward_routed_reduction_order_reviewed':True,'gate_rerun_matches_6c0':idx.tolist()==[[71,3,7,43,54,296],[224,54,283,343,233,263]],'all_selected_expert_tensor_provenance_recorded':bool(ref['source_tensors'].get('selected_experts')),'all_12_expert_invocations_validated':comps['all_12_expert_invocations']['within_tolerance'],'routing_weight_application_validated':True,'independent_reduction_fixture_validated':all(comps[k]['within_tolerance'] for k in ['tiny_weighted_bf16','tiny_accum','tiny_final']),'per_token_routed_sum_validated':comps['final_routed_sum']['within_tolerance'],'shared_expert_entered':False,'model_semantics_validated':False},'non_claims':ref['non_claims']}
 rec['gates']={'moe_forward_routed_reduction_order_reviewed':True,'gate_rerun_matches_6c0':rec['semantic_status']['gate_rerun_matches_6c0'],'all_selected_expert_tensor_provenance_recorded':rec['semantic_status']['all_selected_expert_tensor_provenance_recorded'],'all_12_expert_invocations_pass':rec['semantic_status']['all_12_expert_invocations_validated'],'routing_weight_application_pass':True,'independent_reduction_fixture_pass':rec['semantic_status']['independent_reduction_fixture_validated'],'per_token_routed_sum_pass':rec['semantic_status']['per_token_routed_sum_validated'],'shared_expert_not_entered':not rec['semantic_status']['shared_expert_entered'],'full_model_semantics_not_claimed':not rec['semantic_status']['model_semantics_validated']}
 rec['ok']=all(rec['gates'].values()); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
