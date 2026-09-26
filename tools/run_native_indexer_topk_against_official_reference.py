#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE,DS4_AUTHORITY_SHA,compile_native_prefill_library,load_native_prefill_library
from tools.run_official_compressed_kv_fixture import bf16_to_f32,f32_to_bf16,fp4_quant_inplace,digest
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'; REF='artifacts/indexer-topk-official-reference-fixture.json'
VOCAB=129280; DIM=5120; QR=1280; H=32; ID=128; HD=512; RD=64; BLOCK=32; RATIO=2
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
def topk_sort(scores,compress_lens,offset=2):
 out=np.empty((1,2,1),np.int32)
 for s in range(2):
  idx=int(np.argmax(scores[0,s])); out[0,s,0]=idx+offset if idx<compress_lens[s,0] else -1
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--reference',default=REF); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-indexer-topk-official-reference-validation.json'); a=ap.parse_args(); ref=json.loads(Path(a.reference).read_text()); ck=Path(a.checkpoint); sh=ck/'model-00005-of-00048.safetensors'; toks=np.array(ref['inputs'].get('tokens',[0,3]),np.int64) if 'tokens' in ref['inputs'] else np.array([0,3],np.int64)
 emb=mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)); x=np.ascontiguousarray(emb[toks],np.uint16); native=load_native_prefill_library(compile_native_prefill_library(Path(a.native_out_dir)))
 # K construction from Boundary 5a latent
 latent=np.asarray(json.loads(Path(ref['authority']['input_authority']).read_text())['expected']['compressor_rmsnorm_bf16_uint16'],np.uint16)
 wk=np.ascontiguousarray(mmap(sh,'layers.2.attn.indexer.wk.weight',np.uint16,(ID,HD))); kn=np.ascontiguousarray(mmap(sh,'layers.2.attn.indexer.k_norm.weight',np.uint16,(ID,)))
 kf,r1=native.official_bf16_linear_f32(latent.reshape(1,HD),wk); klin=f32_to_bf16(kf).reshape(1,1,ID); knorm_flat,r2=native.official_rmsnorm_bf16(klin.reshape(1,ID),kn,1e-6); knorm=knorm_flat.reshape(1,1,ID); tail=bf16_to_f32(knorm.reshape(1,1,1,ID)[...,-RD:]); _,_,rtail,_,r3=native.official_rotary_f32(tail,{'original_seq_len':0,'base':10000.0,'factor':16.0,'beta_fast':32,'beta_slow':1}); krot=np.array(knorm.reshape(1,1,1,ID),copy=True); krot[...,-RD:]=f32_to_bf16(rtail); krot=krot.reshape(1,1,ID); kb,ks,kdeq=fp4_quant_inplace(krot.reshape(1,ID)); index_k=kdeq.reshape(1,1,ID)
 # qr and q projection
 wqa=np.ascontiguousarray(mmap(sh,'layers.2.attn.wq_a.weight',np.uint8,(QR,DIM))); wqas=np.ascontiguousarray(mmap(sh,'layers.2.attn.wq_a.scale',np.uint8,(QR//32,DIM//32))); qnw=np.ascontiguousarray(mmap(sh,'layers.2.attn.q_norm.weight',np.uint16,(QR,)))
 wqa_out,r4=native.official_fp8_linear_bf16(x,wqa,wqas,32); qr,r5=native.official_rmsnorm_bf16(wqa_out,qnw,1e-6)
 wqb=np.ascontiguousarray(mmap(sh,'layers.2.attn.indexer.wq_b.weight',np.uint8,(H*ID,QR))); wqbs=np.ascontiguousarray(mmap(sh,'layers.2.attn.indexer.wq_b.scale',np.uint8,((H*ID)//32,QR//32)))
 qproj_flat,r6=native.official_fp8_linear_bf16(qr,wqb,wqbs,32); qproj=qproj_flat.reshape(1,2,H,ID); qtail=bf16_to_f32(qproj[...,-RD:]); _,_,qrtail,_,r7=native.official_rotary_f32(qtail,{'original_seq_len':0,'base':10000.0,'factor':16.0,'beta_fast':32,'beta_slow':1}); qrot=np.array(qproj,copy=True); qrot[...,-RD:]=f32_to_bf16(qrtail); qb,qs,qd=fp4_quant_inplace(qrot.reshape(2*H,ID)); qd=qd.reshape(1,2,H,ID)
 wp=np.ascontiguousarray(mmap(sh,'layers.2.attn.indexer.weights_proj.weight',np.uint16,(H,DIM))); wf,r8=native.official_bf16_linear_f32(x,wp); weights=f32_to_bf16(wf).reshape(1,2,H); weights_f=bf16_to_f32(weights)*np.float32((ID**-0.5)*(H**-0.5))
 raw=np.einsum('bshd,btd->bsht',bf16_to_f32(qd),bf16_to_f32(index_k)).astype(np.float32); weighted=np.maximum(raw,0).astype(np.float32)*weights_f[:,:,:,None]; score=np.sum(weighted,axis=2).astype(np.float32); compress_lens=(np.arange(1,3,dtype=np.int32)//RATIO).reshape(2,1); masked=np.array(score,copy=True); masked[0,0,0]=-np.inf; topk=topk_sort(masked,compress_lens,2)
 exp=ref['expected']; tol=ref['operation_contract']['predeclared_tolerance']; comps={'index_k_cache':cmp(index_k,np.asarray(exp['index_k_cache_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),'qr':cmp(qr,np.asarray(exp['qr_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),'query_projection':cmp(qproj,np.asarray(exp['query_projection_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),'query_dequant':cmp(qd,np.asarray(exp['query_dequant_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),'weights_proj':cmp(weights,np.asarray(exp['weights_proj_bf16_uint16'],np.uint16),tol['bf16_max_ulp_lte']),'per_head_index_score':cmp(raw,np.asarray(exp['per_head_index_score_f32'],np.float32),tol['f32_max_abs_lte'],True),'reduced_index_score':cmp(score,np.asarray(exp['reduced_index_score_f32'],np.float32),tol['f32_max_abs_lte'],True),'causal_masked_score':cmp(np.nan_to_num(masked,neginf=-1e30),np.nan_to_num(np.asarray(exp['causal_masked_score_f32'],np.float32),neginf=-1e30),tol['f32_max_abs_lte'],True),'topk_idxs':cmp(topk,np.asarray(exp['topk_idxs_int32'],np.int32),0),'shared_publication':cmp(topk,np.asarray(exp['shared_topk_idxs_publication_int32'],np.int32),0)}
 rec={'schema':'ds41f.native-indexer-topk-official-reference-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'validate Boundary 5b layer-2 Indexer + top-k publication; stop before candidates','checkpoint':a.checkpoint,'authority':{'reference_fixture':a.reference,'official_checkpoint_raw_bits':a.checkpoint,'native_validation_provider':'Metal-backed linears/RMSNorm/rotary plus exact fp4/topk checks'},'reference_fixture':a.reference,'official_reference':ref['official_reference'],'operation_contract':ref['operation_contract'],'native_version':native.version(),'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'native_result':{'wk_linear':r1,'k_norm':r2,'k_rotary':r3,'wq_a':r4,'q_norm':r5,'wq_b':r6,'q_rotary':r7,'weights_proj':r8},'comparison':comps,'semantic_status':{'official_reference_fixture_used':True,'actual_indexer_layer':2,'candidate_filtering_entered':False,'topk_idxs_int32_exact':comps['topk_idxs']['bit_exact'],'shared_publication_exact':comps['shared_publication']['bit_exact'],'model_semantics_validated':False},'non_claims':ref['non_claims']+['does not execute select_candidate_blocks or candidate filtering']}
 rec['gates']={'reference_not_omlx_derived':ref.get('not_omlx_derived') is True,'reference_classification_expected':ref.get('classification')=='official_reference_derived_independent_arithmetic_contract','index_k_query_scoring_validated':all(comps[k]['within_tolerance'] for k in ['index_k_cache','qr','query_projection','query_dequant','weights_proj','per_head_index_score','reduced_index_score']),'causal_mask_exact':comps['causal_masked_score']['within_tolerance'],'topk_sort_minus_one_exact':comps['topk_idxs']['bit_exact'],'shared_publication_exact':comps['shared_publication']['bit_exact'],'native_metal_executed_where_applicable':all(r.get('metal_enabled') is True for r in [r1,r2,r3,r4,r5,r6,r7,r8]),'candidate_filtering_not_entered':True,'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
 rec['ok']=all(rec['gates'].values()); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
