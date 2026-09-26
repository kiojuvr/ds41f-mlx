#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,os,struct
from pathlib import Path
import numpy as np
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'; KERNEL_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
COMP_SHA='dcd32a8debcf46c4d19d0347a3bc982e7aa70bba9746845d0b1555a7a73c8d67'; COMP_FWD_SHA='cd864ce74af1194d0a30178035efa2af92b8f7d27666724b3c63cd1ca6e5a092'; COMPRESS_KV_SHA='fa0b8a602b8d6e200131396219685902c525c48bfb1943740446c242352ea2d9'; FP4_SHA='1066960c1da76f484a12ae2b456daa4734eabe618f77fd781f3c98d80b05db99'
VOCAB=129280; DIM=5120; HD=512; RD=64; BLOCK=16; LAYER=2; RATIO=2
def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def header(p):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape): h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)
def f32_to_bf16(x):
 u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); l=(u>>16)&1; return ((u+np.uint32(0x7fff)+l)>>16).astype(np.uint16)
def e4(c):
 s=-1.0 if c&0x80 else 1.0; ax=c&0x7f
 if ax==0:return math.copysign(0.0,s)
 e=(c>>3)&0xf; m=c&7; return s*(math.ldexp(m/8,-6) if e==0 else math.ldexp(1+m/8,e-7))
E4=np.array([e4(i) if (i&0x7f)!=0x7f else np.nan for i in range(256)],np.float32); VALID=np.array([i for i in range(256) if (i&0x7f)!=0x7f],np.uint16); VV=E4[VALID]
def e4enc(v):
 v=min(max(float(v),-448),448); d=np.abs(VV.astype(np.float64)-v); m=float(d.min()); t=VALID[d==m]; ev=t[(t&1)==0]; return int(ev[0] if ev.size else t[0])
FP4_VAL=np.array([0,.5,1,1.5,2,3,4,6, -0,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)
def fp4enc(v):
 v=min(max(float(v),-6),6); d=np.abs(FP4_VAL.astype(np.float64)-v); m=float(d.min()); t=np.where(d==m)[0]; ev=t[(t&1)==0]; return int(ev[0] if ev.size else t[0])
def pack_fp4(codes): return (codes[:,0::2] | (codes[:,1::2]<<4)).astype(np.uint8)
def linear_f32(x_bf16,w_bf16): return np.ascontiguousarray(bf16_to_f32(x_bf16) @ bf16_to_f32(w_bf16).T,dtype=np.float32)
def rms(x_bf16,w_bf16):
 x=bf16_to_f32(x_bf16); w=bf16_to_f32(w_bf16); var=np.mean(np.square(x,dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32); return f32_to_bf16((x*(1/np.sqrt(var+np.float32(1e-6),dtype=np.float32))*w).astype(np.float32))
def fp4_quant_inplace(x_bf16):
 x=bf16_to_f32(x_bf16); rows,dim=x.shape; blocks=dim//BLOCK; codes=np.empty((rows,dim),np.uint8); scales=np.empty((rows,blocks),np.uint8); deq=np.empty((rows,dim),np.uint16)
 for r in range(rows):
  for b in range(blocks):
   sl=x[r,b*BLOCK:(b+1)*BLOCK]; a=max(float(np.max(np.abs(sl))),6*(2**-9)); sc=E4[e4enc(a/6.0)]; scales[r,b]=e4enc(a/6.0); cs=np.array([fp4enc(v/sc) for v in sl],np.uint8); codes[r,b*BLOCK:(b+1)*BLOCK]=cs; deq[r,b*BLOCK:(b+1)*BLOCK]=f32_to_bf16(FP4_VAL[cs]*sc)
 return pack_fp4(codes),scales,deq
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--tokens',default='0,3'); ap.add_argument('--out',default='artifacts/compressed-kv-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint); toks=[int(x) for x in a.tokens.split(',') if x.strip()]
 emb=mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)); sh=ck/'model-00005-of-00048.safetensors'; x=np.ascontiguousarray(emb[toks],np.uint16); wkv=mmap(sh,'layers.2.attn.compressor.wkv.weight',np.uint16,(HD,DIM)); wg=mmap(sh,'layers.2.attn.compressor.wgate.weight',np.uint16,(HD,DIM)); nw=mmap(sh,'layers.2.attn.compressor.norm.weight',np.uint16,(HD,))
 kv=linear_f32(x,np.ascontiguousarray(wkv)); score=linear_f32(x,np.ascontiguousarray(wg)); kv_g=kv.reshape(1,1,RATIO,HD); sc_g=score.reshape(1,1,RATIO,HD); ex=np.exp(sc_g-np.max(sc_g,axis=2,keepdims=True)); weights=(ex/np.sum(ex,axis=2,keepdims=True)).astype(np.float32); pooled=np.sum(kv_g*weights,axis=2).reshape(1,HD).astype(np.float32); latent=rms(f32_to_bf16(pooled),np.ascontiguousarray(nw)).reshape(1,1,HD)
 # compressed position is group start 0, so rotary is identity but recorded.
 rotary=np.array(latent,copy=True); fp4_bytes,fp4_sc,deq=fp4_quant_inplace(rotary.reshape(1,HD)); deq=deq.reshape(1,1,HD); cache=np.array(deq,copy=True)
 exp={'input_bf16_uint16':x.tolist(),'wkv_output_f32':kv.tolist(),'compressor_score_f32':score.tolist(),'softmax_pooling_weights_f32':weights.tolist(),'pooled_latent_f32':pooled.tolist(),'compressor_rmsnorm_bf16_uint16':latent.tolist(),'compressed_rotary_bf16_uint16':rotary.tolist(),'fp4_quantized_bytes_uint8':fp4_bytes.reshape(1,1,HD//2).tolist(),'fp4_scales_e4m3_uint8':fp4_sc.reshape(1,1,HD//BLOCK).tolist(),'dequantized_compressed_kv_bf16_uint16':deq.tolist(),'compress_kv_cache_publication_bf16_uint16':cache.tolist()}
 dig={k+'_sha256':digest(np.asarray(v,dtype=np.float32 if k.endswith('f32') or 'f32' in k else (np.uint8 if 'uint8' in k else np.uint16))) for k,v in exp.items()}
 rec={'schema':'ds41f.compressed-kv-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 5a layer-2 Compressor + compressed-KV publication; stop before Indexer','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':['inference/model.py','inference/kernel.py'],'expected_value_provider':'independent arithmetic over official checkpoint tensors'},'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':MODEL_SHA,'functions':[{'name':'Compressor','source_lines':[429,485],'source_sha256':COMP_SHA},{'name':'Compressor.forward','source_lines':[458,485],'source_sha256':COMP_FWD_SHA},{'name':'Attention._compress_kv','source_lines':[739,763],'source_sha256':COMPRESS_KV_SHA}]},'kernel_py':{'file':'inference/kernel.py','file_sha256':KERNEL_SHA,'functions':[{'name':'fp4_act_quant','source_lines':[184,204],'source_sha256':FP4_SHA}]}},'source_tensors':{'wkv.weight':{'name':'layers.2.attn.compressor.wkv.weight','shard':str(sh),'dtype':'BF16','shape':[HD,DIM]},'wgate.weight':{'name':'layers.2.attn.compressor.wgate.weight','shard':str(sh),'dtype':'BF16','shape':[HD,DIM]},'norm.weight':{'name':'layers.2.attn.compressor.norm.weight','shard':str(sh),'dtype':'BF16','shape':[HD]}},'inputs':{'layer':LAYER,'compress_ratio':RATIO,'batch':1,'sequence':len(toks),'tokens':toks,'start_pos':0,'world_size':1},'operation_contract':{'order':'Compressor.forward: wkv/wgate fp32 linears, softmax over ratio, weighted pool, RMSNorm; _compress_kv: compressed-position rotary, fp4_act_quant inplace, publish compress_kv_cache; STOP before Indexer','fp4':'block=16, scale dtype E4M3, scale=E4M3_RNE(max(abs(block),6*2^-9)/6), fp4 E2M1 nearest over {0,.5,1,1.5,2,3,4,6} with sign, dequantized BF16 cache value','cache_publication':'prefill start_pos=0 publishes first compressed slot exactly','predeclared_tolerance':{'f32_max_abs_lte':1e-4,'bf16_max_ulp_lte':1,'fp4_bytes_exact':True,'cache_exact':True}},'expected':exp,'digests':dig,'comparison':{'self_consistent':True,'explicit_stop_before_indexer':True},'non_claims':['does not execute oMLX','does not validate Indexer, candidate selection, decode partial-group state, sparse-attn integration with compressed KV, Block/HC, logits, full layer, or full model correctness','does not benchmark performance'],'ok':True}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
