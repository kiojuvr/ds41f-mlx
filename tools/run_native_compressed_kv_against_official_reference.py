#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,struct,sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE,DS4_AUTHORITY_SHA,compile_native_prefill_library,load_native_prefill_library
from tools.run_official_compressed_kv_fixture import f32_to_bf16,bf16_to_f32,fp4_quant_inplace,digest
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'; REF='artifacts/compressed-kv-official-reference-fixture.json'
VOCAB=129280; DIM=5120; HD=512; RD=64; RATIO=2; BLOCK=16
def header(p):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape): h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def cmp(a,e,tol=0,is_float=False):
 a=np.ascontiguousarray(a); e=np.ascontiguousarray(e)
 if is_float:
  d=np.abs(a.astype(np.float32)-e.astype(np.float32)); m=float(d.max()) if d.size else 0.0; ok=m<=tol
 else:
  d=np.abs(a.astype(np.int64)-e.astype(np.int64)); m=int(d.max()) if d.size else 0; ok=m<=tol
 return {'shape_matches':list(a.shape)==list(e.shape),'bit_exact':digest(a)==digest(e),'max_error':m,'max_error_lte':tol,'within_tolerance':ok,'native_sha256':digest(a),'reference_sha256':digest(e),'mismatch_count':int(np.count_nonzero(d))}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=REF); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-compressed-kv-official-reference-validation.json'); a=ap.parse_args(); ref=json.loads(Path(a.reference).read_text()); ck=Path(a.checkpoint); toks=np.array(ref['inputs']['tokens'],np.int64)
 emb=mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)); sh=ck/'model-00005-of-00048.safetensors'; x=np.ascontiguousarray(emb[toks],np.uint16); wkv=np.ascontiguousarray(mmap(sh,'layers.2.attn.compressor.wkv.weight',np.uint16,(HD,DIM))); wg=np.ascontiguousarray(mmap(sh,'layers.2.attn.compressor.wgate.weight',np.uint16,(HD,DIM))); nw=np.ascontiguousarray(mmap(sh,'layers.2.attn.compressor.norm.weight',np.uint16,(HD,)))
 native=load_native_prefill_library(compile_native_prefill_library(Path(a.native_out_dir)))
 kv,r1=native.official_bf16_linear_f32(x,wkv); score,r2=native.official_bf16_linear_f32(x,wg); kv_g=kv.reshape(1,1,RATIO,HD); sc_g=score.reshape(1,1,RATIO,HD); ex=np.exp(sc_g-np.max(sc_g,axis=2,keepdims=True)); weights=(ex/np.sum(ex,axis=2,keepdims=True)).astype(np.float32); pooled=np.sum(kv_g*weights,axis=2).reshape(1,HD).astype(np.float32); latent,r3=native.official_rmsnorm_bf16(f32_to_bf16(pooled),nw,1e-6); latent=latent.reshape(1,1,HD)
 tail=bf16_to_f32(latent[...,-RD:]).reshape(1,1,1,RD); _,_,rot_tail,_,r4=native.official_rotary_f32(tail,{'original_seq_len':0,'base':160000.0,'factor':16.0,'beta_fast':32,'beta_slow':1}); rotary=np.array(latent,copy=True); rotary[...,-RD:]=f32_to_bf16(rot_tail.reshape(1,1,RD))
 fp4_bytes,fp4_sc,deq=fp4_quant_inplace(rotary.reshape(1,HD)); deq=deq.reshape(1,1,HD); cache=np.array(deq,copy=True)
 exp=ref['expected']; tol=ref['operation_contract']['predeclared_tolerance']; comps={'wkv_output':cmp(kv,np.asarray(exp['wkv_output_f32'],np.float32),tol['f32_max_abs_lte'],True),'compressor_score':cmp(score,np.asarray(exp['compressor_score_f32'],np.float32),tol['f32_max_abs_lte'],True),'softmax_pooling_weights':cmp(weights,np.asarray(exp['softmax_pooling_weights_f32'],np.float32),tol['f32_max_abs_lte'],True),'pooled_latent':cmp(pooled,np.asarray(exp['pooled_latent_f32'],np.float32),tol['f32_max_abs_lte'],True),'compressor_rmsnorm':cmp(latent,np.asarray(exp['compressor_rmsnorm_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),'compressed_rotary':cmp(rotary,np.asarray(exp['compressed_rotary_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),'fp4_quantized_bytes':cmp(fp4_bytes.reshape(1,1,HD//2),np.asarray(exp['fp4_quantized_bytes_uint8'],np.uint8),0),'fp4_scales':cmp(fp4_sc.reshape(1,1,HD//BLOCK),np.asarray(exp['fp4_scales_e4m3_uint8'],np.uint8),0),'dequantized_compressed_kv':cmp(deq,np.asarray(exp['dequantized_compressed_kv_bf16_uint16'],np.uint16),0),'cache_publication':cmp(cache,np.asarray(exp['compress_kv_cache_publication_bf16_uint16'],np.uint16),0)}
 rec={'schema':'ds41f.native-compressed-kv-official-reference-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate Boundary 5a Compressor + compressed-KV publication; stop before Indexer','checkpoint':a.checkpoint,'authority':{'reference_fixture':a.reference,'official_checkpoint_raw_bits':a.checkpoint,'native_validation_provider':'Metal-backed BF16 linears/RMSNorm/rotary plus exact FP4/cache checks'},'reference_fixture':a.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'native_version':native.version(),'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'native_result':{'wkv_linear':r1,'wgate_linear':r2,'compressor_rmsnorm':r3,'compressed_rotary':r4},'comparison':comps,'semantic_status':{'official_reference_fixture_used':True,'actual_compressing_layer':ref['inputs']['layer'],'compress_ratio':ref['inputs']['compress_ratio'],'indexer_entered':False,'cache_publication_exact':comps['cache_publication']['bit_exact'],'model_semantics_validated':False},'non_claims':ref['non_claims']+['does not execute Indexer or candidate selection']}
 rec['gates']={'reference_not_omlx_derived':ref.get('not_omlx_derived') is True,'reference_classification_expected':ref.get('classification')=='official_reference_derived_independent_arithmetic_contract','pooling_weights_latent_validated':all(comps[k]['within_tolerance'] for k in ['softmax_pooling_weights','pooled_latent']),'compressed_rope_validated':comps['compressed_rotary']['within_tolerance'],'fp4_bytes_scales_validated':comps['fp4_quantized_bytes']['bit_exact'] and comps['fp4_scales']['bit_exact'],'cache_publication_exact':comps['cache_publication']['bit_exact'],'native_metal_executed_where_applicable':all(r.get('metal_enabled') is True for r in [r1,r2,r3,r4]),'indexer_not_entered':True,'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
 rec['ok']=all(rec['gates'].values())
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
