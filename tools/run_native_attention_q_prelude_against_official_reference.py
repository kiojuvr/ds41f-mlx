#!/usr/bin/env python3
"""Validate native Attention Q-prelude composition against official fixture."""
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE, DS4_AUTHORITY_SHA, compile_native_prefill_library, load_native_prefill_library  # noqa
DEFAULT_CHECKPOINT="/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"; DEFAULT_REFERENCE="artifacts/attention-q-prelude-official-reference-fixture.json"
VOCAB_SIZE=129280; DIM=5120; Q_RANK=1280; N_HEADS=64; HEAD_DIM=512; ROPE_DIM=64; WQB_OUT=32768; BLOCK=32

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def header(path:Path):
    with path.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(path,name,dtype,shape):
    h,b=header(path); off=h[name]['data_offsets'][0]; return np.memmap(path,mode='r',dtype=dtype,offset=b+off,shape=shape)
def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)
def f32_to_bf16_rne(x):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); l=(u>>16)&1
    return ((u+np.uint32(0x7fff)+l)>>16).astype(np.uint16)
def cmp_u16(actual, expected, tol):
    a=np.ascontiguousarray(actual,dtype=np.uint16); e=np.ascontiguousarray(expected,dtype=np.uint16); d=np.abs(a.astype(np.int32)-e.astype(np.int32))
    return {'shape_matches':list(a.shape)==list(e.shape),'bit_exact':digest(a)==digest(e),'mismatch_count':int(np.count_nonzero(d)),'max_bf16_ulp_error':int(d.max()) if d.size else 0,'max_bf16_ulp_lte':int(tol),'within_tolerance':(int(d.max()) if d.size else 0)<=int(tol),'native_sha256':digest(a),'reference_sha256':digest(e)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=DEFAULT_REFERENCE); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-attention-q-prelude-official-reference-validation.json'); args=ap.parse_args()
    ref=json.loads(Path(args.reference).read_text())
    if ref.get('not_omlx_derived') is not True: raise ValueError('reference fixture must be not_omlx_derived')
    ck=Path(args.checkpoint); tokens=np.array(ref['inputs']['tokens'],dtype=np.int64); shard=ck/'model-00003-of-00048.safetensors'; embed_shard=ck/'model-00002-of-00048.safetensors'
    embed=mmap(embed_shard,'embed.weight',np.uint16,(VOCAB_SIZE,DIM)); x=np.ascontiguousarray(embed[tokens],dtype=np.uint16)
    wqa=np.ascontiguousarray(mmap(shard,'layers.0.attn.wq_a.weight',np.uint8,(Q_RANK,DIM))); wqa_s=np.ascontiguousarray(mmap(shard,'layers.0.attn.wq_a.scale',np.uint8,(Q_RANK//BLOCK,DIM//BLOCK)))
    qnw=np.ascontiguousarray(mmap(shard,'layers.0.attn.q_norm.weight',np.uint16,(Q_RANK,)))
    wqb=np.ascontiguousarray(mmap(shard,'layers.0.attn.wq_b.weight',np.uint8,(WQB_OUT,Q_RANK))); wqb_s=np.ascontiguousarray(mmap(shard,'layers.0.attn.wq_b.scale',np.uint8,(WQB_OUT//BLOCK,Q_RANK//BLOCK)))
    native=load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    wqa_out,r1=native.official_fp8_linear_bf16(x,wqa,wqa_s,BLOCK)
    qn_out,r2=native.official_rmsnorm_bf16(wqa_out,qnw,1.0e-6)
    wqb_out,r3=native.official_fp8_linear_bf16(qn_out,wqb,wqb_s,BLOCK)
    reshaped=np.ascontiguousarray(wqb_out.reshape(1,int(tokens.size),N_HEADS,HEAD_DIM))
    q_rot=np.array(reshaped,copy=True)
    tail_f32=bf16_to_f32(q_rot[..., -ROPE_DIM:])
    _,_,rot_tail,_,r4=native.official_rotary_f32(tail_f32,{'original_seq_len':0,'base':10000.0,'factor':16.0,'beta_fast':32,'beta_slow':1})
    q_rot[..., -ROPE_DIM:]=f32_to_bf16_rne(rot_tail)
    exp=ref['expected']; tol_i=ref['operation_contract']['predeclared_tolerance']['intermediate_max_bf16_ulp_lte']; tol_f=ref['operation_contract']['predeclared_tolerance']['final_rotary_max_bf16_ulp_lte']
    comps={
      'input':cmp_u16(x,np.asarray(exp['input_bf16_uint16'],dtype=np.uint16),0),
      'wq_a_output':cmp_u16(wqa_out,np.asarray(exp['wq_a_output_bf16_uint16'],dtype=np.uint16),tol_i),
      'q_norm_output':cmp_u16(qn_out,np.asarray(exp['q_norm_output_bf16_uint16'],dtype=np.uint16),tol_i),
      'wq_b_output':cmp_u16(wqb_out,np.asarray(exp['wq_b_output_bf16_uint16'],dtype=np.uint16),tol_i),
      'reshaped_q':cmp_u16(reshaped,np.asarray(exp['reshaped_q_bf16_uint16'],dtype=np.uint16),tol_i),
      'rotary_applied_q':cmp_u16(q_rot,np.asarray(exp['rotary_applied_q_bf16_uint16'],dtype=np.uint16),tol_f)}
    rec={'schema':'ds41f.native-attention-q-prelude-official-reference-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate native layer-0 Attention Q-prelude through rotary and stop before sparse_attn','checkpoint':str(ck),'authority':{'reference_fixture':args.reference,'official_checkpoint_raw_bits':str(ck),'native_validation_provider':'Metal-backed validated FP8 linear/RMSNorm/rotary primitive composition'},'reference_fixture':args.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'native_version':native.version(),'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'native_result':{'wq_a_fp8_linear':r1,'q_norm_rmsnorm':r2,'wq_b_fp8_linear':r3,'rotary_tail':r4},'comparison':comps,'semantic_status':{'official_reference_fixture_used':True,'real_shape_wq_a_used':True,'real_shape_wq_b_used':True,'multi_block_fp8_scale_indexing_exercised':True,'explicit_stop_before_sparse_attn':True,'native_attention_q_prelude_validated_against_contract':all(c['within_tolerance'] and c['shape_matches'] for c in comps.values()),'model_semantics_validated':False,'validated_stage':'Attention Q-prelude through rotary only'},'non_claims':ref['non_claims']+['does not execute sparse_attn or any K/V/cache path']}
    rec['gates']={'reference_not_omlx_derived':ref.get('not_omlx_derived') is True,'reference_classification_expected':ref.get('classification')=='official_reference_derived_independent_arithmetic_contract','native_metal_executed':all(r.get('metal_enabled') is True for r in [r1,r2,r3,r4]),'real_shape_wq_a_used':r1.get('in_dim')==DIM and r1.get('out_dim')==Q_RANK,'real_shape_wq_b_used':r3.get('in_dim')==Q_RANK and r3.get('out_dim')==WQB_OUT,'multi_block_scale_indexing_exercised':True,'all_intermediates_within_tolerance':all(c['within_tolerance'] and c['shape_matches'] for c in comps.values()),'explicit_stop_before_sparse_attn':rec['semantic_status']['explicit_stop_before_sparse_attn'],'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
    rec['ok']=all(rec['gates'].values())
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
