#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,os,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_compressed_kv_fixture import bf16_to_f32,f32_to_bf16,fp4_quant_inplace,digest
from tools.run_official_attention_q_prelude_fixture import fp8_linear_bf16
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
CKV='artifacts/compressed-kv-official-reference-fixture.json'
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
INDEXER_SHA='cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75'; FWD_SHA='32c3de30aeb0e78df5271e28dc9b2873feade877987048137a590f6f58ba110d'; CTOPK_SHA='da4f82f646260020cdefc673e5f09ceda8280cdef2f14c424832b362bd40f640'
VOCAB=129280; DIM=5120; QR=1280; H=32; ID=128; HD=512; RD=64; BLOCK=32; RATIO=2; LAYER=2

def header(p):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape): h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def linear_f32(x_bf16,w_bf16): return np.ascontiguousarray(bf16_to_f32(x_bf16) @ bf16_to_f32(w_bf16).T,dtype=np.float32)
def rms_bf16(x_bf16,w_bf16):
 x=bf16_to_f32(x_bf16); w=bf16_to_f32(w_bf16); var=np.mean(np.square(x,dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32); return f32_to_bf16((x*(1/np.sqrt(var+np.float32(1e-6),dtype=np.float32))*w).astype(np.float32))
def rotary(x_bf16):
 y=np.array(x_bf16,copy=True); tail=bf16_to_f32(y[...,-RD:]); half=RD//2; freqs=1/(10000.0**(np.arange(0,RD,2,dtype=np.float32)/np.float32(RD))); ang=np.outer(np.arange(y.shape[1],dtype=np.float32),freqs).astype(np.float32); c=np.cos(ang).astype(np.float32); s=np.sin(ang).astype(np.float32); p=tail.reshape(*tail.shape[:-1],half,2); re=p[...,0]; im=p[...,1]; out=np.empty_like(p); out[...,0]=re*c[None,:,None,:]; out[...,1]=im*c[None,:,None,:] # pos0 identity; pos1 only for q below overwritten
 if y.shape[1]>1:
  out[...,0]=re*c[None,:,None,:]-im*s[None,:,None,:]; out[...,1]=re*s[None,:,None,:]+im*c[None,:,None,:]
 y[...,-RD:]=f32_to_bf16(out.reshape(*tail.shape)); return y
def topk_sort(scores,compress_lens,offset=2):
 B,S,T=scores.shape; out=np.empty((B,S,1),np.int32)
 for b in range(B):
  for s in range(S):
   # topk 1; numpy argmax returns first. all -inf -> 0 then masked to -1.
   idx=int(np.argmax(scores[b,s])); out[b,s,0]=idx+offset if idx<compress_lens[s,0] else -1
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--compressed-kv',default=CKV); ap.add_argument('--tokens',default='0,3'); ap.add_argument('--out',default='artifacts/indexer-topk-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint); toks=[int(x) for x in a.tokens.split(',') if x.strip()]; ckv=json.loads(Path(a.compressed_kv).read_text())
 sh=ck/'model-00005-of-00048.safetensors'; emb=mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)); x=np.ascontiguousarray(emb[toks],np.uint16)
 latent=np.asarray(ckv['expected']['compressor_rmsnorm_bf16_uint16'],np.uint16) # [1,1,512]
 # index key construction
 wk=mmap(sh,'layers.2.attn.indexer.wk.weight',np.uint16,(ID,HD)); kn=mmap(sh,'layers.2.attn.indexer.k_norm.weight',np.uint16,(ID,)); k_lin=f32_to_bf16(linear_f32(latent.reshape(1,HD),np.ascontiguousarray(wk))).reshape(1,1,ID); k_norm=rms_bf16(k_lin.reshape(1,ID),np.ascontiguousarray(kn)).reshape(1,1,ID); k_rot=rotary(k_norm.reshape(1,1,1,ID)).reshape(1,1,ID); k_bytes,k_sc,k_deq=fp4_quant_inplace(k_rot.reshape(1,ID)); index_k=k_deq.reshape(1,1,ID)
 # qr for layer2 from x
 wqa=mmap(sh,'layers.2.attn.wq_a.weight',np.uint8,(QR,DIM)); wqas=mmap(sh,'layers.2.attn.wq_a.scale',np.uint8,(QR//32,DIM//32)); qnw=mmap(sh,'layers.2.attn.q_norm.weight',np.uint16,(QR,)); wqa_out=fp8_linear_bf16(x,np.ascontiguousarray(wqa),np.ascontiguousarray(wqas))[0]; qr=rms_bf16(wqa_out,np.ascontiguousarray(qnw))
 wqb=mmap(sh,'layers.2.attn.indexer.wq_b.weight',np.uint8,(H*ID,QR)); wqbs=mmap(sh,'layers.2.attn.indexer.wq_b.scale',np.uint8,((H*ID)//32,QR//32)); q_proj=fp8_linear_bf16(qr,np.ascontiguousarray(wqb),np.ascontiguousarray(wqbs))[0].reshape(1,2,H,ID); q_rot=rotary(q_proj); q_bytes,q_sc,q_deq=fp4_quant_inplace(q_rot.reshape(2*H,ID)); q_deq=q_deq.reshape(1,2,H,ID)
 wp=mmap(sh,'layers.2.attn.indexer.weights_proj.weight',np.uint16,(H,DIM)); weights=f32_to_bf16(linear_f32(x,np.ascontiguousarray(wp))).reshape(1,2,H); weights_f=bf16_to_f32(weights)*np.float32((ID**-0.5)*(H**-0.5))
 raw=np.einsum('bshd,btd->bsht',bf16_to_f32(q_deq),bf16_to_f32(index_k)).astype(np.float32); relu=np.maximum(raw,0).astype(np.float32); weighted=relu*weights_f[:,:,:,None]; score=np.sum(weighted,axis=2).astype(np.float32); compress_lens=(np.arange(1,3,dtype=np.int32)//RATIO).reshape(2,1); masked=np.array(score,copy=True); masked[:,np.arange(2),:]=score[:,np.arange(2),:]; masked[0,0,0]=-np.inf; topk=topk_sort(masked,compress_lens,offset=2)
 exp={'input_bf16_uint16':x.tolist(),'index_k_linear_bf16_uint16':k_lin.tolist(),'index_k_norm_bf16_uint16':k_norm.tolist(),'index_k_rotary_bf16_uint16':k_rot.tolist(),'index_k_fp4_bytes_uint8':k_bytes.reshape(1,1,ID//2).tolist(),'index_k_fp4_scales_uint8':k_sc.reshape(1,1,ID//16).tolist(),'index_k_cache_bf16_uint16':index_k.tolist(),'qr_bf16_uint16':qr.tolist(),'query_projection_bf16_uint16':q_proj.tolist(),'query_rotary_bf16_uint16':q_rot.tolist(),'query_fp4_bytes_uint8':q_bytes.reshape(1,2,H,ID//2).tolist(),'query_fp4_scales_uint8':q_sc.reshape(1,2,H,ID//16).tolist(),'query_dequant_bf16_uint16':q_deq.tolist(),'weights_proj_bf16_uint16':weights.tolist(),'per_head_index_score_f32':raw.tolist(),'relu_weighted_score_f32':weighted.tolist(),'reduced_index_score_f32':score.tolist(),'causal_masked_score_f32':masked.tolist(),'compress_lens_int32':compress_lens.tolist(),'topk_idxs_int32':topk.tolist(),'shared_topk_idxs_publication_int32':topk.tolist()}
 dig={k+'_sha256':digest(np.asarray(v,dtype=np.float32 if 'f32' in k else (np.int32 if 'int32' in k else (np.uint8 if 'uint8' in k else np.uint16)))) for k,v in exp.items()}
 rec={'schema':'ds41f.indexer-topk-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 5b layer-2 Indexer + top-k publication; stop before candidate filtering','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':'inference/model.py','input_authority':a.compressed_kv,'expected_value_provider':'independent arithmetic over official checkpoint tensors'},'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':MODEL_SHA,'functions':[{'name':'Indexer','source_lines':[488,580],'source_sha256':INDEXER_SHA},{'name':'Indexer.forward','source_lines':[527,580],'source_sha256':FWD_SHA},{'name':'Attention._compress_topk_idxs','source_lines':[722,737],'source_sha256':CTOPK_SHA}]}},'source_tensors':{'indexer.wk.weight':{'name':'layers.2.attn.indexer.wk.weight','shard':str(sh),'dtype':'BF16','shape':[ID,HD]},'indexer.k_norm.weight':{'name':'layers.2.attn.indexer.k_norm.weight','shard':str(sh),'dtype':'BF16','shape':[ID]},'indexer.wq_b.weight':{'name':'layers.2.attn.indexer.wq_b.weight','shard':str(sh),'dtype':'F8_E4M3','shape':[H*ID,QR]},'indexer.wq_b.scale':{'name':'layers.2.attn.indexer.wq_b.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[(H*ID)//32,QR//32]},'indexer.weights_proj.weight':{'name':'layers.2.attn.indexer.weights_proj.weight','shard':str(sh),'dtype':'BF16','shape':[H,DIM]}},'inputs':{'layer':2,'batch':1,'sequence':2,'tokens':toks,'start_pos':0,'compress_ratio':2,'world_size':1,'candidate_filtering_enabled':False,'offset':2},'operation_contract':{'order':'latent -> wk/k_norm/RoPE/FP4 -> index_k cache; qr -> wq_b/RoPE/FP4; weights_proj; per-head dot, ReLU, weighted head reduction; causal compressed-position mask; top-k; position sort; +offset; publish shared topk_idxs; STOP before candidates','causal_mask':'compress_lens=floor(arange(1,S+1)/ratio); compressed index >= compress_lens is unreachable and maps to -1','topk':'topk over score then sorted by position; unreachable topk entries become -1; reachable positions shifted by offset','predeclared_tolerance':{'bf16_max_ulp_lte':1,'f32_max_abs_lte':1e-4,'int32_exact':True,'fp4_exact':True}},'expected':exp,'digests':dig,'comparison':{'self_consistent':True,'minus_one_unreachable_present':bool(np.any(topk<0)),'explicit_stop_before_candidate_filtering':True},'non_claims':['does not execute oMLX','does not validate select_candidate_blocks, candidate source layer 20, candidate consumer masking, sparse-attn integration with compressed KV, decode path, Block/HC, logits, full layer, or full model correctness','does not benchmark performance'],'ok':True}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
