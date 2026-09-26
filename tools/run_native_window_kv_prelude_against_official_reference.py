#!/usr/bin/env python3
"""Validate native layer-0 window-KV prelude against official fixture."""
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE, DS4_AUTHORITY_SHA, compile_native_prefill_library, load_native_prefill_library  # noqa
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'; DEFAULT_REFERENCE='artifacts/window-kv-prelude-official-reference-fixture.json'
VOCAB_SIZE=129280; DIM=5120; KV_DIM=512; ROPE_DIM=64; BLOCK=32; WINDOW=128

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def header(p:Path):
    with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape):
    h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)
def f32_to_bf16_rne(x):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); l=(u>>16)&1; return ((u+np.uint32(0x7fff)+l)>>16).astype(np.uint16)
def cmp_u(a,e,tol):
    a=np.ascontiguousarray(a); e=np.ascontiguousarray(e); d=np.abs(a.astype(np.int64)-e.astype(np.int64)); m=int(d.max()) if d.size else 0
    return {'shape_matches':list(a.shape)==list(e.shape),'bit_exact':digest(a)==digest(e),'mismatch_count':int(np.count_nonzero(d)),'max_error':m,'max_error_lte':int(tol),'within_tolerance':m<=int(tol),'native_sha256':digest(a),'reference_sha256':digest(e)}
def topk(seqlen):
    end=np.arange(seqlen,dtype=np.int32)[:,None]; idx=(end-WINDOW+1).clip(0)+np.arange(min(seqlen,WINDOW),dtype=np.int32); return np.where(idx>end,-1,idx).astype(np.int32)[None,:,:]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=DEFAULT_REFERENCE); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-window-kv-prelude-official-reference-validation.json'); args=ap.parse_args()
    ref=json.loads(Path(args.reference).read_text()); assert ref.get('not_omlx_derived') is True
    ck=Path(args.checkpoint); tokens=np.array(ref['inputs']['tokens'],dtype=np.int64); shard=ck/'model-00003-of-00048.safetensors'; emb_shard=ck/'model-00002-of-00048.safetensors'
    x=np.ascontiguousarray(mmap(emb_shard,'embed.weight',np.uint16,(VOCAB_SIZE,DIM))[tokens]); w=np.ascontiguousarray(mmap(shard,'layers.0.attn.wkv.weight',np.uint8,(KV_DIM,DIM))); ws=np.ascontiguousarray(mmap(shard,'layers.0.attn.wkv.scale',np.uint8,(KV_DIM//BLOCK,DIM//BLOCK))); nw=np.ascontiguousarray(mmap(shard,'layers.0.attn.kv_norm.weight',np.uint16,(KV_DIM,)))
    native=load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    wkv,r1=native.official_fp8_linear_bf16(x,w,ws,BLOCK)
    kvn,r2=native.official_rmsnorm_bf16(wkv,nw,1e-6)
    tail=bf16_to_f32(kvn.reshape(1,int(tokens.size),KV_DIM)[..., -ROPE_DIM:]).reshape(1,int(tokens.size),1,ROPE_DIM); _,_,rot_tail,_,r3=native.official_rotary_f32(tail,{'original_seq_len':0,'base':10000.0,'factor':16.0,'beta_fast':32,'beta_slow':1})
    rot=np.array(kvn.reshape(1,int(tokens.size),KV_DIM),copy=True); rot[..., -ROPE_DIM:]=f32_to_bf16_rne(rot_tail.reshape(1,int(tokens.size),ROPE_DIM))
    q,sc,quant_flat,r4=native.official_act_quant_bf16(rot.reshape(int(tokens.size),KV_DIM),BLOCK)
    q=q.reshape(1,int(tokens.size),KV_DIM); sc=sc.reshape(1,int(tokens.size),KV_DIM//BLOCK); window_kv=quant_flat.reshape(1,int(tokens.size),KV_DIM)
    cache=np.zeros((1,WINDOW,KV_DIM),dtype=np.uint16); cache[:,:int(tokens.size),:]=window_kv
    idx=topk(int(tokens.size))
    exp=ref['expected']; tol=ref['operation_contract']['predeclared_tolerance']
    comps={'input':cmp_u(x,np.asarray(exp['input_bf16_uint16'],dtype=np.uint16),0),'wkv_output':cmp_u(wkv,np.asarray(exp['wkv_output_bf16_uint16'],dtype=np.uint16),tol['intermediate_max_bf16_ulp_lte']),'kv_norm_output':cmp_u(kvn,np.asarray(exp['kv_norm_output_bf16_uint16'],dtype=np.uint16),tol['intermediate_max_bf16_ulp_lte']),'rotary_kv':cmp_u(rot,np.asarray(exp['rotary_kv_bf16_uint16'],dtype=np.uint16),tol['intermediate_max_bf16_ulp_lte']),'quantized_window_kv_fp8':cmp_u(q,np.asarray(exp['quantized_window_kv_fp8_uint8'],dtype=np.uint8),0),'quantized_window_kv_scale':cmp_u(sc,np.asarray(exp['quantized_window_kv_scale_e8m0_uint8'],dtype=np.uint8),0),'window_kv':cmp_u(window_kv,np.asarray(exp['window_kv_bf16_uint16'],dtype=np.uint16),tol['act_quant_max_bf16_ulp_lte']),'cache_after_publication':cmp_u(cache,np.asarray(exp['cache_after_publication_bf16_uint16'],dtype=np.uint16),0),'topk_idxs':cmp_u(idx,np.asarray(exp['topk_idxs_int32'],dtype=np.int32),0)}
    rec={'schema':'ds41f.native-window-kv-prelude-official-reference-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate native layer-0 window-KV prelude and stop before sparse_attn','checkpoint':str(ck),'authority':{'reference_fixture':args.reference,'official_checkpoint_raw_bits':str(ck),'native_validation_provider':'Metal-backed FP8 linear/RMSNorm/rotary/act_quant plus exact cache/topk checks'},'reference_fixture':args.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'native_version':native.version(),'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'native_result':{'wkv_fp8_linear':r1,'kv_norm_rmsnorm':r2,'rotary_tail':r3,'act_quant_window_kv':r4},'comparison':comps,'semantic_status':{'official_reference_fixture_used':True,'real_shape_wkv_used':True,'kv_rotary_and_act_quant_validated':all(comps[k]['within_tolerance'] for k in ['rotary_kv','quantized_window_kv_fp8','quantized_window_kv_scale','window_kv']),'cache_publication_exact':comps['cache_after_publication']['bit_exact'],'topk_int32_exact':comps['topk_idxs']['bit_exact'],'explicit_stop_before_sparse_attn':True,'model_semantics_validated':False,'validated_stage':'window-KV prelude only'},'non_claims':ref['non_claims']+['does not execute sparse_attn']}
    rec['gates']={'reference_not_omlx_derived':ref.get('not_omlx_derived') is True,'reference_classification_expected':ref.get('classification')=='official_reference_derived_independent_arithmetic_contract','native_metal_executed':all(r.get('metal_enabled') is True for r in [r1,r2,r3,r4]),'real_shape_wkv_path_validated':r1.get('in_dim')==DIM and r1.get('out_dim')==KV_DIM,'kv_rotary_act_quant_validated':rec['semantic_status']['kv_rotary_and_act_quant_validated'],'cache_publication_exact':rec['semantic_status']['cache_publication_exact'],'topk_int32_exact':rec['semantic_status']['topk_int32_exact'],'explicit_stop_before_sparse_attn':True,'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
    rec['ok']=all(rec['gates'].values())
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
