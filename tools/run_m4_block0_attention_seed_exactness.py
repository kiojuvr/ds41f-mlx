#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/m4/block0-attention-seed-exactness/result.json'
ACT=ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz'
EXP=ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz'
OMLX=Path('/Users/kioju/omlx-0.7.0.dev2')

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def bf16(x): return (np.asarray(x,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def ordered(a):
    x=a.astype(np.uint16).astype(np.int32); return np.where((x&0x8000)!=0,0x8000-x,x).astype(np.int32)
def cmp(a,b):
    d=np.abs(bf16(a)-bf16(b)); u=np.abs(ordered(a)-ordered(b)); return {'actual_sha256':sha(a),'expected_sha256':sha(b),'max_bf16_ulp':int(u.max()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'differing_count':int((a!=b).sum()),'count_gt_1_ulp':int((u>1).sum())}
def elem(a,b,idx):
    idx=tuple(idx); return {'index':list(idx),'actual_value':float(bf16(a[idx])),'expected_value':float(bf16(b[idx])),'actual_bits':f'0x{int(a[idx]):04x}','expected_bits':f'0x{int(b[idx]):04x}','ulp':int(abs(ordered(a[idx])-ordered(b[idx]))),'abs':float(abs(bf16(a[idx])-bf16(b[idx])))}
def git(path,args): return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def grep(path,pats):
    out=[]
    for i,l in enumerate(path.read_text(errors='replace').splitlines(),1):
        if any(p in l for p in pats): out.append({'line':i,'text':l.strip()})
    return out

def main():
    act=np.load(ACT); exp=np.load(EXP); causal=json.loads((ROOT/'artifacts/m4/trajectory-seed-causal-audit/result.json').read_text())
    idx=[0,0,3758]
    input_cmp=cmp(act['block0_attention_input'],exp['block0_attn_pre_norm_output'])
    final_cmp=cmp(act['block0_attention_output'],exp['block0_attention_output'])
    deterministic={'available_repetitions':1,'same_existing_capture_value':'0x%04x'%int(act['block0_attention_output'][tuple(idx)]),'status':'SINGLE_CAPTURE_DETERMINISTIC_WITHIN_RECORDED_RUN; additional production reruns not performed in this task'}
    internals=[
      {'stage':'input h / attention_input','status':'EXACT','comparison':input_cmp,'production_boundary':'captured actual block0_attention_input','official_boundary':'expected block0_attn_pre_norm_output'},
      {'stage':'shared input activation quantization for wq_a/wkv','status':'NOT_CAPTURED_PRODUCTION_INTERNAL','classification':'UNRESOLVED','production_boundary':'oMLX Attention._input_projections quantize_activation(x) when wq_a and wkv are QuantizedProjection(quantize_input=True)','official_boundary':'reviewed fp8_linear actq over the same BF16 attention_input'},
      {'stage':'wq_a output','status':'NOT_CAPTURED_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'q_norm output','status':'NOT_CAPTURED_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'wq_b input activation quantization','status':'NOT_CAPTURED_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'q_b pre-RoPE output','status':'NOT_CAPTURED_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'q_rotary','status':'NOT_CAPTURED_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'wkv projection output','status':'NOT_CAPTURED_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'kv_norm / kv_rotary / packed window KV','status':'NOT_CAPTURED_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'sparse attention output','status':'NOT_CAPTURED_FOR_BLOCK0_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'inverse RoPE output','status':'NOT_CAPTURED_FOR_BLOCK0_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'wo_a grouped projection input/output','status':'NOT_CAPTURED_FOR_BLOCK0_PRODUCTION_INTERNAL','classification':'UNRESOLVED'},
      {'stage':'wo_b activation quantization / quantized matmul / final BF16 Attention output','status':'DIFF_WITHIN_CURRENT_CONTRACT','classification':'FIRST_OBSERVED_CAUSAL_NONEXACT_BOUNDARY','comparison':final_cmp,'element':elem(act['block0_attention_output'],exp['block0_attention_output'],idx)}]
    rec={'schema':'ds41f.m4.block0_attention_seed_exactness.v1','production_source_identity':{'omlx_path':str(OMLX),'git':git(OMLX,['rev-parse','HEAD']) if OMLX.exists() else None,'expected_git':'b390b31e0c6831225fed0f24d278eb1db7fcb68b','language_py_sha256':hashlib.sha256((OMLX/'omlx/patches/deepseek_v41/language.py').read_bytes()).hexdigest(),'quantization_py_sha256':hashlib.sha256((OMLX/'omlx/patches/deepseek_v41/quantization.py').read_bytes()).hexdigest(),'kernels_py_sha256':hashlib.sha256((OMLX/'omlx/patches/deepseek_v41/kernels.py').read_bytes()).hexdigest()},'official_source_identity':{'checkpoint':'/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash','model_py_sha256':'4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'},'block0_attention_input':{'actual_digest':sha(act['block0_attention_input']),'expected_digest':sha(exp['block0_attn_pre_norm_output']),'equality_confirmed':bool(np.array_equal(act['block0_attention_input'],exp['block0_attn_pre_norm_output'])),'comparison':input_cmp},'internal_boundary_sequence':internals,'first_nonexact_internal_boundary':'first observed non-exact production-vs-official boundary is final Block0 Attention output; earlier requested production internals were not present in the existing capture and were not re-captured here','final_seed':elem(act['block0_attention_output'],exp['block0_attention_output'],idx),'root_cause_classification':'UNRESOLVED_ACTUAL_ATTENTION_INTERNALS_NOT_CAPTURED; known to arise inside Block0 Attention between exact input and final output, with production path using QuantizedProjection/quantize_activation, packed_sparse_attention, inverse rope, grouped mx.einsum, wo_b QuantizedProjection and BF16 output','activation_quantization_involved':'POSSIBLE_BUT_NOT_LOCALIZED; QuantizedProjection quantizes shared input for wq_a/wkv and quantizes wq_b/wo_b inputs, but no actual codes/scales were captured for Block0','reduction_fma_cast_involved':'POSSIBLE_BUT_NOT_LOCALIZED; sparse attention, mx.einsum, quantized_matmul reduction order, and final BF16 cast all remain candidate causes until a production-internal capture is added','deterministic_reproduction':deterministic,'upstream_mlx_runtime_evidence':{'searched_local_runtime_files':['language.py','quantization.py','kernels.py'],'relevant_source_lines':{'attention':grep(OMLX/'omlx/patches/deepseek_v41/language.py',['def _input_projections','quantize_activation(x)','packed_sparse_attention','mx.einsum','self.wo_b']),'quantization':grep(OMLX/'omlx/patches/deepseek_v41/quantization.py',['def quantize_activation','class QuantizedProjection','mx.quantized_matmul','project_quantized'])},'external_issue_pr_search':'not performed; no internet/tooling available in this harness'},'exact_path_candidates':[{'name':'official-source-derived NumPy/reference Block0 Attention path','preserves_official_semantics':True,'changes_seed_to_expected':True,'output_at_3758':'0x3f6c','production_topology_preserved':False,'bounded_performance_implication':'major/diagnostic-only; not a practical production path as-is'},{'name':'future production-internal exact/canonical MLX path','preserves_official_semantics':'unknown until first actual internal mismatch is captured','changes_seed_to_expected':'unknown','production_topology_preserved':'unknown'}],'seed_removal_downstream_trajectory_result':{'experiment':'Use exact official Attention output (expected path / EEEE) instead of actual 1-ULP seed; compare with prior causal S and actual trajectory','seed_removed':True,'attention_output_3758_after_removal':'0x3f6c','downstream_30k_branch_eliminated_relative_to_E':True,'evidence':'The causal audit proved AEEE (only actual attention_output) recreates actual downstream trajectory. Therefore removing that one A seed returns to E; direct production exact fallback not implemented.','actual_seeded_progression_from_prior_artifact':causal['seeded_progression']},'final_classification':'BLOCK0_ATTENTION_EXACTNESS_LOCALIZATION_INCOMPLETE','high_sensitivity_exact_boundary_recommendation':'CANDIDATE_HIGH_SENSITIVITY_EXACT_BOUNDARY, but promotion requires a real production-internal exact alternative that preserves topology or an explicit performance tradeoff decision','next_frontier':'bounded Block0 Attention production-internal capture for length=1: quantize_activation codes/scales, q/wkv/q_norm/q_b/RoPE/window/sparse/inverse_rope/wo_a/wo_b intermediates, then test a real MLX exact/canonical alternative if localized','non_claims':['does not widen tolerance','does not resume Layer2','does not implement DwarfStar','does not prove root cause','does not provide a production exact fallback']}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'input_exact':rec['block0_attention_input']['equality_confirmed'],'first_observed':rec['first_nonexact_internal_boundary'],'classification':rec['final_classification']},indent=2))
if __name__=='__main__': main()
