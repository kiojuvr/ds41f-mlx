#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_m4_trajectory_seed_causal_audit import block0_remainder_from_xattn, hc_post, build_prefill_state, cmp_bf16, cmp_f32
from tools.run_native_first_incremental_block1_layer2_entry_validation import arr_digest

OUT=ROOT/'artifacts/m4/reduction-trajectory-behavioral-stability/result.json'
ACT=ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz'
EXP=ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz'
CUDA_JSON=ROOT/'artifacts/m4/block0-wob-official-cuda-oracle/cuda-oracle-result.json'
CUDA_NPY=ROOT/'artifacts/m4/block0-wob-official-cuda-oracle/cuda_wob_output_bf16_u16.npy'

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def bits(a,i=3758): return f'0x{int(a.reshape(1,1,-1)[0,0,i]):04x}'
def bf16_to_f32(x): return (np.asarray(x,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def elem(a,i=3758): return {'index':[0,0,i],'bits':bits(a,i),'value':float(bf16_to_f32(a.reshape(1,1,-1)[0,0,i]))}
def pairwise(vals, cmpfn=cmp_bf16):
    return {'C_vs_E':cmpfn(vals['C'],vals['E']),'C_vs_M':cmpfn(vals['C'],vals['M']),'M_vs_E':cmpfn(vals['M'],vals['E'])}
def eq3(vals): return bool(np.array_equal(vals['E'],vals['C']) and np.array_equal(vals['E'],vals['M']))
def top_summary_not_run():
    return {'executed':False,'reason':'This runner intentionally stops at Block1/layer2-entry continuous/discrete diagnostics. Full logits and bounded multi-token continuation require extending the reviewed source-derived full decode path with a Block0-Attention-output injection point; no production or CUDA full-model runtime was used.'}

def main():
    act=np.load(ACT); exp=np.load(EXP); cuda=json.loads(CUDA_JSON.read_text()); C=np.load(CUDA_NPY)
    if sha(C)!='13c5dc43f40d4b4d0ba3771bf3b8fcdd492478f940de9ef3274f9f8c3f032d45': raise SystemExit('CUDA output digest mismatch')
    if cuda['fixture']['npz_sha256']!='4bc7ab067127decc22c28cdfa25814882f7314d59e320cc5c6cadc998e806cd2': raise SystemExit('fixture sha mismatch')
    E=np.ascontiguousarray(exp['block0_attention_output'],dtype=np.uint16); M=np.ascontiguousarray(act['block0_attention_output'],dtype=np.uint16)
    prefill,_=build_prefill_state(); hash_ids=exp['engram1_hash_ids']
    branches={}
    for name,attn in [('E',E),('C',C),('M',M)]:
        post=hc_post(attn, exp['block0_entry_h'], exp['block0_attn_ao'], exp['block0_attn_ac'])
        branches[name]=block0_remainder_from_xattn(post, exp['block0_attn_ap'], hash_ids, prefill)
        branches[name]['post_attention_h']=post
    boundaries={
      'block0_attention_output': {'E':E,'C':C,'M':M},
      'block0_post_attention_h': {k:v['post_attention_h'] for k,v in branches.items()},
      'block0_ffn_pre_norm': {k:v['ffn_pre_norm'] for k,v in branches.items()},
      'block0_moe_output': {k:v['moe_output'] for k,v in branches.items()},
      'block0_final_x_out': {k:v['x_out'] for k,v in branches.items()},
      'engram1_output': {k:v['engram1_output'] for k,v in branches.items()},
      'block1_attention_input': {k:v['block1']['attention_input'] for k,v in branches.items()},
      'block1_q_rotary': {k:v['block1']['attn_path']['q'] for k,v in branches.items()},
      'block1_sparse_output': {k:v['block1']['sparse'] for k,v in branches.items()},
      'block1_attention_output': {k:v['block1']['attention_output'] for k,v in branches.items()},
      'block1_moe_input': {k:v['block1']['moe_input'] for k,v in branches.items()},
      'block1_x_out': {k:v['block1']['x_out'] for k,v in branches.items()},
    }
    comparisons={k:pairwise(v) for k,v in boundaries.items()}
    first_amp=None
    for k in boundaries:
        c=comparisons[k]['C_vs_E']
        if c.get('max_bf16_ulp',0)>32 or c.get('count_gt_32_ulp',0)>0:
            first_amp={'boundary':k,'C_vs_E':c}; break
    disc={
      'block0_moe_route_ids':{k:v['moe_route_ids'] for k,v in branches.items()},
      'block0_moe_route_weights':{k:v['moe_route_weights'] for k,v in branches.items()},
      'block1_topk_window_indices':{k:v['block1']['attn_path']['topk'].tolist() for k,v in branches.items()},
      'block1_moe_route_ids':{k:v['block1']['moe']['idx'].tolist() for k,v in branches.items()},
      'block1_moe_expert_sets':{k:v['block1']['moe']['expert_ids'] for k,v in branches.items()},
    }
    discrete_eq={k:(v['E']==v['C']==v['M']) for k,v in disc.items()}
    # Route weights are continuous diagnostics; route ids, selected sets, and sparse/top-k indices are discrete decisions.
    discrete_gate_keys=['block0_moe_route_ids','block1_topk_window_indices','block1_moe_route_ids','block1_moe_expert_sets']
    first_disc=next((k for k in discrete_gate_keys if not discrete_eq[k]),None)
    rec={
      'schema':'ds41f.m4.reduction_trajectory_behavioral_stability.v1',
      'status':'PARTIAL_FIRST_TOKEN_SOURCE_DERIVED_DIAGNOSTIC_FULL_LOGITS_PENDING',
      'cuda_oracle_import':{'json':str(CUDA_JSON),'output_npy':str(CUDA_NPY),'fixture_sha256':cuda['fixture']['npz_sha256'],'cuda_output_digest':sha(C),'verified':True,'external_final_classification':cuda['final_classification'],'environment':cuda['cuda_environment'],'official_source_identities':cuda['official_source_identities'],'activation_matches_fixture':cuda['oracle_A_complete_official_linear']['activation_matches_fixture'],'oracle_A_B_agree':cuda['oracle_B_fixed_quantized_inputs']['agrees_with_oracle_A']},
      'trajectory_identities':{'E':{'name':'source-derived arithmetic reference','digest':sha(E),'element_3758':elem(E)},'C':{'name':'official CUDA/TileLang wo_b output','digest':sha(C),'element_3758':elem(C)},'M':{'name':'production MLX M=1 output','digest':sha(M),'element_3758':elem(M)}},
      'block0_attention_pairwise':comparisons['block0_attention_output'],
      'continuous_boundary_pairwise':comparisons,
      'first_continuous_amplification_boundary_gt32ulp_C_vs_E':first_amp,
      'discrete_decisions_compared':disc,
      'discrete_decision_equality':discrete_eq,
      'first_discrete_divergence':first_disc,
      'block0_moe_route_equality':discrete_eq['block0_moe_route_ids'],
      'engram1_output_pairwise':comparisons['engram1_output'],
      'block1_sparse_index_equality':discrete_eq['block1_topk_window_indices'],
      'block1_moe_route_equality':discrete_eq['block1_moe_route_ids'],
      'full_logits':top_summary_not_run(),
      'topk_logits':top_summary_not_run(),
      'bounded_continuation':{'executed':False,'length':0,'reason':'First-token logits gate was not reached in this local source-derived injection runner.'},
      'first_token_behavioral_stability_classification':'NOT_EVALUATED_FULL_LOGITS_PENDING',
      'final_classification':'THREE_TRAJECTORY_BEHAVIORAL_STABILITY_INCOMPLETE_FULL_LOGITS_PENDING',
      'precision_policy_decision':'Interim: official CUDA establishes multiple valid FP8 reduction trajectories; connected hidden-state bit identity and whole-logits SHA identity are non-authoritative while the behavioral/logit stability gate remains pending.',
      'metal_exactification_status':'RETIRED_AS_ACTIVE_FRONTIER_PENDING_BEHAVIORAL_STABILITY; do not implement canonical Metal wo_b solely to match NumPy or CUDA bits.',
      'next_frontier':'extend reviewed source-derived decode/logits harness with explicit Block0 Attention output injection for E/C/M and run first-token logits plus bounded lockstep continuation',
      'ok':False}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'classification':rec['final_classification'],'first_discrete_divergence':first_disc,'cuda_digest':sha(C)},indent=2))
if __name__=='__main__': main()
