#!/usr/bin/env python3
"""Validate native sparse_attn against Boundary 3 fixture."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE, DS4_AUTHORITY_SHA, compile_native_prefill_library, load_native_prefill_library  # noqa
DEFAULT_REFERENCE='artifacts/sparse-attn-official-reference-fixture.json'; DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def cmp(a,e,tol):
    a=np.ascontiguousarray(a,dtype=np.uint16); e=np.ascontiguousarray(e,dtype=np.uint16); d=np.abs(a.astype(np.int32)-e.astype(np.int32)); m=int(d.max()) if d.size else 0
    return {'shape_matches':list(a.shape)==list(e.shape),'bit_exact':digest(a)==digest(e),'mismatch_count':int(np.count_nonzero(d)),'max_bf16_ulp_error':m,'max_bf16_ulp_lte':int(tol),'within_tolerance':m<=int(tol),'native_sha256':digest(a),'reference_sha256':digest(e)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=DEFAULT_REFERENCE); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-sparse-attn-official-reference-validation.json'); args=ap.parse_args()
    ref=json.loads(Path(args.reference).read_text()); assert ref.get('not_omlx_derived') is True
    b1=json.loads(Path(ref['inputs']['q_source']).read_text()); b2=json.loads(Path(ref['inputs']['window_kv_source']).read_text())
    q=np.asarray(b1['expected']['rotary_applied_q_bf16_uint16'],dtype=np.uint16); kv=np.asarray(b2['expected']['window_kv_bf16_uint16'],dtype=np.uint16); idx=np.asarray(ref['expected']['selected_kv_indices_int32'],dtype=np.int32); sink_digest=ref['source_tensors']['attn_sink']['digest']
    # attn_sink bounded values are not embedded; load official checkpoint raw tensor.
    import struct
    ck=Path(args.checkpoint); p=ck/'model-00003-of-00048.safetensors'
    with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); base=8+n
    off=h['layers.0.attn.attn_sink']['data_offsets'][0]; sink=np.ascontiguousarray(np.memmap(p,mode='r',dtype=np.float32,offset=base+off,shape=(64,)))
    if digest(sink)!=sink_digest: raise ValueError('attn_sink digest mismatch')
    native=load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    out,nres=native.official_sparse_attn_bf16(q,kv,sink,idx,float(ref['inputs']['softmax_scale']))
    expected=np.asarray(ref['expected']['attention_output_bf16_uint16'],dtype=np.uint16); tol=ref['operation_contract']['predeclared_tolerance']['max_bf16_ulp_lte']; c=cmp(out,expected,tol)
    rec={'schema':'ds41f.native-sparse-attn-official-reference-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate native Metal sparse_attn against Boundary 3 fixture','checkpoint':args.checkpoint,'authority':{'reference_fixture':args.reference,'official_checkpoint_raw_bits':args.checkpoint,'native_validation_provider':'Metal-backed ds41f_official_sparse_attn_bf16'},'reference_fixture':args.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'native_version':native.version(),'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'native_result':nres,'digests':{'native_output_bf16_sha256':digest(out),'reference_output_bf16_sha256':digest(expected)},'comparison':{'attention_output':c,'minus_one_mask_exact_in_fixture':ref['comparison']['minus_one_mask_present'],'sink_denominator_effect_validated_by_fixture':ref['comparison']['sink_denominator_effect_present']},'semantic_status':{'official_reference_fixture_used':True,'q_from_boundary1':True,'window_kv_topk_from_boundary2':True,'native_sparse_attn_validated_against_contract':c['within_tolerance'] and c['shape_matches'],'explicit_stop_before_inverse_rotary_output_projection':True,'model_semantics_validated':False,'validated_stage':'sparse_attn only'},'non_claims':ref['non_claims']+['does not execute inverse rotary, wo_a, or wo_b']}
    rec['gates']={'reference_not_omlx_derived':ref.get('not_omlx_derived') is True,'reference_classification_expected':ref.get('classification')=='official_reference_derived_independent_arithmetic_contract','q_kv_topk_from_validated_boundaries':True,'minus_one_mask_semantics_present':ref['comparison']['minus_one_mask_present'],'sink_denominator_semantics_validated':ref['comparison']['sink_denominator_effect_present'],'native_metal_executed':nres.get('metal_enabled') is True,'output_within_predeclared_tolerance':c['within_tolerance'] and c['shape_matches'],'explicit_stop_before_inverse_rotary_output_projection':True,'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
    rec['ok']=all(rec['gates'].values())
    outp=Path(args.out); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(outp); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
