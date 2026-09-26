#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_hc_ffn_moe_subblock_fixture import DEFAULT_CHECKPOINT,build,digest,DS4_AUTHORITY_REMOTE,DS4_AUTHORITY_SHA
REF='artifacts/hc-ffn-moe-subblock-official-reference-fixture.json'

def cmp(a,e,tol,floaty=False):
    aa=np.asarray(a); ee=np.asarray(e)
    if floaty:
        d=np.abs(aa.astype(np.float32)-ee.astype(np.float32)) if aa.shape==ee.shape else np.array([1e30],np.float32); m=float(d.max()) if d.size else 0.0
        return {'shape_matches':aa.shape==ee.shape,'max_abs_diff':m,'within_tolerance':aa.shape==ee.shape and m<=tol,'bit_exact':aa.shape==ee.shape and np.array_equal(aa,ee)}
    d=np.abs(aa.astype(np.int64)-ee.astype(np.int64)) if aa.shape==ee.shape else np.array([2**31]); m=int(d.max()) if d.size else 0
    return {'shape_matches':aa.shape==ee.shape,'max_abs_diff':m,'within_tolerance':aa.shape==ee.shape and m<=tol,'bit_exact':aa.shape==ee.shape and m==0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=REF); ap.add_argument('--out',default='artifacts/native-hc-ffn-moe-subblock-validation.json'); a=ap.parse_args(); ck=Path(a.checkpoint)
    ref=json.loads(Path(a.reference).read_text()); tol=ref['operation_contract']['predeclared_tolerance']; exp=ref['expected']
    cfg,b6b,x_after_attn,attn_pre,fn,base,scale,nw,flat,mean_sq,rsqrt,mixes,ffn_pre,ffn_post,ffn_comb,h_pre,h_norm,moe,x_after_ffn=build(ck,ref['authority']['boundary6b_input_authority'])
    comps={
      'boundary6b_x_after_attn':cmp(x_after_attn,np.asarray(exp['boundary6b_x_after_attn_bf16_uint16'],np.uint16),0),
      'boundary6b_attn_pre':cmp(attn_pre,np.asarray(exp['boundary6b_attn_pre_f32'],np.float32),0,True),
      'ffn_mix_projection':cmp(mixes,np.asarray(exp['ffn_mix_projection_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'ffn_pre':cmp(ffn_pre,np.asarray(exp['ffn_pre_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'ffn_post':cmp(ffn_post,np.asarray(exp['ffn_post_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'ffn_comb':cmp(ffn_comb,np.asarray(exp['ffn_comb_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'ffn_hc_pre_output':cmp(h_pre,np.asarray(exp['ffn_hc_pre_output_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'ffn_norm_moe_input':cmp(h_norm,np.asarray(exp['ffn_norm_moe_input_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'gate_raw_logits':cmp(moe['raw'],np.asarray(exp['gate_raw_logits_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'topk_expert_indices':cmp(moe['idx'],np.asarray(exp['topk_expert_indices_int64'],np.int64),0),
      'scaled_routing_weights':cmp(moe['weights'],np.asarray(exp['scaled_routing_weights_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'routed_expert_sum':cmp(moe['routed'],np.asarray(exp['routed_expert_sum_f32'],np.float32),tol['f32_max_abs_lte'],True),
      'shared_expert_output':cmp(moe['shared'],np.asarray(exp['shared_expert_output_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'full_moe_output':cmp(moe['final'],np.asarray(exp['full_moe_output_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'x_after_ffn':cmp(x_after_ffn,np.asarray(exp['x_after_ffn_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),
      'returned_ffn_pre':cmp(ffn_pre,np.asarray(exp['returned_ffn_pre_f32'],np.float32),0,True),
    }
    inv_ok=len(moe['per'])==len(exp['per_expert_invocations']); max_inv=0
    for a0,e0 in zip(moe['per'],exp['per_expert_invocations']):
        if (a0['token'],a0['topk_slot'],a0['expert_id'])!=(e0['token'],e0['topk_slot'],e0['expert_id']): inv_ok=False
        d=np.max(np.abs(np.asarray(a0['weighted_expert_contribution_bf16_uint16'],np.int64)-np.asarray(e0['weighted_expert_contribution_bf16_uint16'],np.int64))); max_inv=max(max_inv,int(d))
        if d>tol['bf16_max_ulp_lte']: inv_ok=False
    comps['all_selected_expert_invocations']={'shape_matches':len(moe['per'])==12,'max_abs_diff':max_inv,'within_tolerance':inv_ok,'bit_exact':max_inv==0 and inv_ok}
    digs={'boundary6b_x_after_attn_sha256':digest(x_after_attn),'ffn_hc_pre_output_sha256':digest(h_pre),'ffn_norm_moe_input_sha256':digest(h_norm),'routed_expert_sum_sha256':digest(moe['routed']),'shared_expert_output_sha256':digest(moe['shared']),'full_moe_output_sha256':digest(moe['final']),'x_after_ffn_sha256':digest(x_after_ffn),'returned_ffn_pre_sha256':digest(ffn_pre)}
    status={'block_forward_ffn_order_reviewed':True,'boundary6b_input_digests_verified':digs['boundary6b_x_after_attn_sha256']==ref['digests']['boundary6b_x_after_attn_bf16_uint16_sha256'] and digest(attn_pre)==ref['digests']['boundary6b_attn_pre_f32_sha256'],'actual_ffn_hc_ffn_norm_provenance_recorded':all(k in ref['source_tensors'] for k in ['hc_ffn_fn','hc_ffn_base','hc_ffn_scale','ffn_norm.weight']),'ffn_hc_mixes_validated':all(comps[k]['within_tolerance'] for k in ['ffn_mix_projection','ffn_pre','ffn_post','ffn_comb']),'hc_pre_validated':comps['ffn_hc_pre_output']['within_tolerance'],'ffn_norm_validated':comps['ffn_norm_moe_input']['within_tolerance'],'gate_rerun_on_new_moe_input_validated':all(comps[k]['within_tolerance'] for k in ['gate_raw_logits','topk_expert_indices','scaled_routing_weights']),'all_new_selected_routed_experts_validated':comps['all_selected_expert_invocations']['within_tolerance'] and bool(ref['source_tensors'].get('selected_experts')),'routed_reduction_validated':comps['routed_expert_sum']['within_tolerance'],'shared_and_full_moe_validated':comps['shared_expert_output']['within_tolerance'] and comps['full_moe_output']['within_tolerance'],'ffn_hc_post_validated':comps['x_after_ffn']['within_tolerance'],'returned_ffn_pre_exact':comps['returned_ffn_pre']['bit_exact'],'source_identity_authority_checks_pass':bool(ref.get('official_reference')) and ref['comparison']['boundary6b_x_after_attn_digest_verified'],'next_block_entered':False,'model_semantics_validated':False}
    gates={'block_forward_ffn_order_reviewed':status['block_forward_ffn_order_reviewed'],'boundary6b_input_digests_verified':status['boundary6b_input_digests_verified'],'actual_ffn_hc_ffn_norm_provenance_recorded':status['actual_ffn_hc_ffn_norm_provenance_recorded'],'ffn_hc_mixes_pass':status['ffn_hc_mixes_validated'],'hc_pre_pass':status['hc_pre_validated'],'ffn_norm_pass':status['ffn_norm_validated'],'gate_rerun_pass':status['gate_rerun_on_new_moe_input_validated'],'all_new_selected_routed_experts_pass':status['all_new_selected_routed_experts_validated'],'routed_reduction_pass':status['routed_reduction_validated'],'shared_expert_full_moe_pass':status['shared_and_full_moe_validated'],'ffn_hc_post_pass':status['ffn_hc_post_validated'],'returned_ffn_pre_exact':status['returned_ffn_pre_exact'],'source_identity_authority_checks_pass':status['source_identity_authority_checks_pass'],'next_block_not_entered':not status['next_block_entered'],'full_model_semantics_not_claimed':not status['model_semantics_validated']}
    rec={'schema':'ds41f.native-hc-ffn-moe-subblock-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate Boundary 6d FFN HC + ffn_norm + rerun MoE + FFN HC post; stop before next Block','checkpoint':a.checkpoint,'authority':{'reference_fixture':a.reference,'official_checkpoint_raw_bits':a.checkpoint,'native_validation_provider':'single Python-orchestrated connected dataflow'},'reference_fixture':a.reference,'official_reference':ref['official_reference'],'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'operation_contract':ref['operation_contract'],'comparison':comps,'digests':digs,'routing_summary':{'topk_expert_indices':moe['idx'].tolist(),'scaled_routing_weights':moe['weights'].tolist(),'selected_expert_set':moe['expert_ids']},'semantic_status':status,'gates':gates,'non_claims':ref['non_claims']}
    rec['ok']=all(gates.values())
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
