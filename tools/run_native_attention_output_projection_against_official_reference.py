#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE, DS4_AUTHORITY_SHA, compile_native_prefill_library, load_native_prefill_library  # noqa
from tools.run_official_attention_output_projection_fixture import inv_rot, deq_weight_bf16, digest, BLOCK, WOAOUT, WOAIN, DIM, GROUPS, O_RANK
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'; REF='artifacts/attention-output-projection-official-reference-fixture.json'
def header(p):
 with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape): h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def cmp(a,e,tol):
 a=np.ascontiguousarray(a,dtype=np.uint16); e=np.ascontiguousarray(e,dtype=np.uint16); d=np.abs(a.astype(np.int32)-e.astype(np.int32)); m=int(d.max()) if d.size else 0
 return {'shape_matches':list(a.shape)==list(e.shape),'bit_exact':digest(a)==digest(e),'mismatch_count':int(np.count_nonzero(d)),'max_bf16_ulp_error':m,'max_bf16_ulp_lte':int(tol),'within_tolerance':m<=int(tol),'native_sha256':digest(a),'reference_sha256':digest(e)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=REF); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-attention-output-projection-official-reference-validation.json'); a=ap.parse_args()
 ref=json.loads(Path(a.reference).read_text()); ck=Path(a.checkpoint); shard=ck/'model-00003-of-00048.safetensors'; exp=ref['expected']; sparse=np.asarray(exp['sparse_attn_input_bf16_uint16'],np.uint16)
 inv=inv_rot(sparse); grouped_in=inv.reshape(1,sparse.shape[1],GROUPS,WOAIN)
 woa=np.ascontiguousarray(mmap(shard,'layers.0.attn.wo_a.weight',np.uint8,(WOAOUT,WOAIN))); woas=np.ascontiguousarray(mmap(shard,'layers.0.attn.wo_a.scale',np.uint8,(WOAOUT//BLOCK,WOAIN//BLOCK))); woa_bf16=deq_weight_bf16(woa,woas)
 wob=np.ascontiguousarray(mmap(shard,'layers.0.attn.wo_b.weight',np.uint8,(DIM,WOAOUT))); wobs=np.ascontiguousarray(mmap(shard,'layers.0.attn.wo_b.scale',np.uint8,(DIM//BLOCK,WOAOUT//BLOCK)))
 native=load_native_prefill_library(compile_native_prefill_library(Path(a.native_out_dir)))
 woa_parts=[]; lin_results=[]
 for g in range(GROUPS):
  xg=np.ascontiguousarray(grouped_in[:,:,g,:].reshape(sparse.shape[1],WOAIN)); wg=np.ascontiguousarray(woa_bf16[g*O_RANK:(g+1)*O_RANK,:]); outg,r=native.official_bf16_linear_f32(xg,wg); lin_results.append(r); woa_parts.append(outg.astype(np.float32))
 # convert grouped F32 outputs to BF16 RNE locally for official dtype boundary
 from tools.run_official_attention_output_projection_fixture import f32_to_bf16_rne
 woa_f32=np.stack(woa_parts,axis=1).reshape(sparse.shape[1],GROUPS,O_RANK)[None,:,:,:]; woa_out=f32_to_bf16_rne(woa_f32); flat=woa_out.reshape(1,sparse.shape[1],WOAOUT)
 final_flat,rwob=native.official_fp8_linear_bf16(flat.reshape(sparse.shape[1],WOAOUT),wob,wobs,BLOCK); final=final_flat.reshape(1,sparse.shape[1],DIM)
 tol_i=ref['operation_contract']['predeclared_tolerance']['intermediate_max_bf16_ulp_lte']; tol_f=ref['operation_contract']['predeclared_tolerance']['final_max_bf16_ulp_lte']
 comps={'inverse_rotary_output':cmp(inv,np.asarray(exp['inverse_rotary_output_bf16_uint16'],np.uint16),tol_i),'grouped_reshape':cmp(grouped_in,np.asarray(exp['grouped_reshape_bf16_uint16'],np.uint16),0),'grouped_wo_a_output':cmp(woa_out,np.asarray(exp['grouped_wo_a_output_bf16_uint16'],np.uint16),tol_i),'flattened_wo_a_output':cmp(flat,np.asarray(exp['flattened_wo_a_output_bf16_uint16'],np.uint16),tol_i),'final_wo_b_output':cmp(final,np.asarray(exp['final_wo_b_output_bf16_uint16'],np.uint16),tol_f)}
 rec={'schema':'ds41f.native-attention-output-projection-official-reference-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate Boundary 4 attention output projection and stop before Block/HC','checkpoint':a.checkpoint,'authority':{'reference_fixture':a.reference,'official_checkpoint_raw_bits':a.checkpoint,'native_validation_provider':'Metal-backed BF16 grouped wo_a linears and FP8 wo_b linear'},'reference_fixture':a.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'native_version':native.version(),'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'native_result':{'wo_a_group_linears':lin_results,'wo_b_fp8_linear':rwob},'comparison':comps,'semantic_status':{'official_reference_fixture_used':True,'inverse_rotary_validated':comps['inverse_rotary_output']['within_tolerance'],'grouped_wo_a_validated':comps['grouped_wo_a_output']['within_tolerance'],'wo_b_validated':comps['final_wo_b_output']['within_tolerance'],'explicit_stop_before_block_hc':True,'model_semantics_validated':False,'validated_stage':'attention output projection only'},'non_claims':ref['non_claims']+['does not execute Block or HC residual integration']}
 rec['gates']={'reference_not_omlx_derived':ref.get('not_omlx_derived') is True,'reference_classification_expected':ref.get('classification')=='official_reference_derived_independent_arithmetic_contract','inverse_rotary_validation_passed':comps['inverse_rotary_output']['within_tolerance'],'grouped_wo_a_validation_passed':comps['grouped_wo_a_output']['within_tolerance'],'wo_b_projection_passed':comps['final_wo_b_output']['within_tolerance'],'native_metal_executed_where_applicable':all(r.get('metal_enabled') is True for r in lin_results+[rwob]),'explicit_stop_before_block_hc':True,'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
 rec['ok']=all(rec['gates'].values())
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
