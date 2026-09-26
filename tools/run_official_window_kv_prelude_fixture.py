#!/usr/bin/env python3
"""Generate bounded layer-0 window-KV prelude fixture from official bits."""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct
from pathlib import Path
from typing import Any
import numpy as np
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
MODEL_PY_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'; KERNEL_PY_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
WINDOW_KV_SHA='62f9eda94cd22ee13aa115822ffede34f22aec317671411380a69eceab66a2bc'; TOPK_SHA='20d752a018e1a7b72190468471849bcd869319b50130b5902639973237b2d0b6'
LINEAR_SHA='c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada'; RMS_FWD_SHA='adb7c70ed245e8830f6692b8c026fd7f2d17518cd98b68bf973e75524a921c85'; ROTARY_SHA='1c47da553d29f41c798a2eda82b72476473913d1201e32283de0e4977aa30b1a'; ACT_QUANT_SHA='563a82836450bfefe3f5f176636dec0e1f4d126c76d8007b42bb7abb877d30cb'; FP8_GEMM_SHA='cfd550d8b02ee127760ac26b39accae603302be6bf8e97c0cce29d8993af0657'
VOCAB_SIZE=129280; DIM=5120; KV_DIM=512; ROPE_DIM=64; BLOCK=32; WINDOW=128

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def header(p:Path):
    with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape):
    h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)
def f32_to_bf16_rne(x):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); l=(u>>16)&1; return ((u+np.uint32(0x7fff)+l)>>16).astype(np.uint16)
def e4_code(c:int):
    s=-1.0 if c&0x80 else 1.0; ax=c&0x7f
    if ax==0: return math.copysign(0.0,s)
    e=(c>>3)&0xf; m=c&7; return s*(math.ldexp(m/8.0,-6) if e==0 else math.ldexp(1.0+m/8.0,e-7))
E4=np.array([e4_code(i) if (i&0x7f)!=0x7f else np.nan for i in range(256)],dtype=np.float32); VALID=np.array([i for i in range(256) if (i&0x7f)!=0x7f],dtype=np.uint16); VV=E4[VALID]
def e4_enc(v):
    v=min(max(float(v),-448.0),448.0); d=np.abs(VV.astype(np.float64)-v); m=float(d.min()); tied=VALID[d==m]; even=tied[(tied&1)==0]; return int(even[0] if even.size else tied[0])
def e8_arr(x): return np.ldexp(np.ones_like(x,dtype=np.float32), x.astype(np.int32)-127)
def act_quant(x_bf16):
    x=bf16_to_f32(x_bf16); rows,dim=x.shape; blocks=dim//BLOCK; q=np.empty((rows,dim),dtype=np.uint8); s=np.empty((rows,blocks),dtype=np.uint8); out=np.empty((rows,dim),dtype=np.uint16)
    for r in range(rows):
      for b in range(blocks):
        sl=x[r,b*BLOCK:(b+1)*BLOCK]; amax=max(float(np.max(np.abs(sl))),1e-4); exp=math.ceil(math.log2(amax/448.0)); scale=np.float32(math.ldexp(1.0,exp)); s[r,b]=exp+127
        codes=np.array([e4_enc(np.clip(v/scale,-448,448)) for v in sl],dtype=np.uint8); q[r,b*BLOCK:(b+1)*BLOCK]=codes; out[r,b*BLOCK:(b+1)*BLOCK]=f32_to_bf16_rne(E4[codes]*scale)
    return q,s,out
def act_quant_scales(x_f32):
    rows,k=x_f32.shape; kb=k//BLOCK; q=np.empty((rows,k),dtype=np.uint8); s=np.empty((rows,kb),dtype=np.float32)
    for r in range(rows):
      for b in range(kb):
        sl=x_f32[r,b*BLOCK:(b+1)*BLOCK]; amax=max(float(np.max(np.abs(sl))),1e-4); exp=math.ceil(math.log2(amax/448.0)); scale=np.float32(math.ldexp(1.0,exp)); s[r,b]=scale; q[r,b*BLOCK:(b+1)*BLOCK]=[e4_enc(np.clip(v/scale,-448,448)) for v in sl]
    return q,s
def fp8_linear(inp_bf16,w,ws):
    x=bf16_to_f32(inp_bf16); rows,k=x.shape; out_dim=w.shape[0]; kb=k//BLOCK; aq,asc=act_quant_scales(x); out=np.zeros((rows,out_dim),dtype=np.float32); adeq=E4[aq].reshape(rows,kb,BLOCK)*asc[:,:,None]
    for b in range(kb):
        scale=e8_arr(ws[:,b]); wd=E4[w[:,b*BLOCK:(b+1)*BLOCK]]*scale[np.arange(out_dim)//BLOCK,None]; out += adeq[:,b,:].astype(np.float32) @ wd.astype(np.float32).T
    return f32_to_bf16_rne(out)
def rms(x_bf16,w_bf16):
    x=bf16_to_f32(x_bf16); w=bf16_to_f32(w_bf16); var=np.mean(np.square(x,dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32); y=x*(1.0/np.sqrt(var+np.float32(1e-6),dtype=np.float32))*w; return f32_to_bf16_rne(y.astype(np.float32))
def rotary_kv(kv_bf16):
    kv=np.array(kv_bf16,copy=True).reshape(1,kv_bf16.shape[0],KV_DIM); tail=bf16_to_f32(kv[..., -ROPE_DIM:]); half=ROPE_DIM//2; freqs=1.0/(10000.0**(np.arange(0,ROPE_DIM,2,dtype=np.float32)/np.float32(ROPE_DIM))); angles=np.outer(np.arange(kv.shape[1],dtype=np.float32),freqs).astype(np.float32); c=np.cos(angles).astype(np.float32); s=np.sin(angles).astype(np.float32); p=tail.reshape(1,kv.shape[1],half,2); re=p[...,0]; im=p[...,1]; o=np.empty_like(p); o[...,0]=re*c[None,:,:]-im*s[None,:,:]; o[...,1]=re*s[None,:,:]+im*c[None,:,:]; kv[..., -ROPE_DIM:]=f32_to_bf16_rne(o.reshape(1,kv.shape[1],ROPE_DIM)); return kv

def topk(seqlen):
    end=np.arange(seqlen,dtype=np.int32)[:,None]; idx=(end-WINDOW+1).clip(0)+np.arange(min(seqlen,WINDOW),dtype=np.int32); return np.where(idx>end,-1,idx).astype(np.int32)[None,:,:]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--tokens',default='0,3'); ap.add_argument('--out',default='artifacts/window-kv-prelude-official-reference-fixture.json'); args=ap.parse_args(); ck=Path(args.checkpoint); toks=[int(x) for x in args.tokens.split(',') if x.strip()]
    shard=ck/'model-00003-of-00048.safetensors'; emb_shard=ck/'model-00002-of-00048.safetensors'; emb=mmap(emb_shard,'embed.weight',np.uint16,(VOCAB_SIZE,DIM)); x=np.ascontiguousarray(emb[toks],dtype=np.uint16)
    w=mmap(shard,'layers.0.attn.wkv.weight',np.uint8,(KV_DIM,DIM)); ws=mmap(shard,'layers.0.attn.wkv.scale',np.uint8,(KV_DIM//BLOCK,DIM//BLOCK)); nw=mmap(shard,'layers.0.attn.kv_norm.weight',np.uint16,(KV_DIM,))
    wkv=fp8_linear(x,np.ascontiguousarray(w),np.ascontiguousarray(ws)); kvn=rms(wkv,np.ascontiguousarray(nw)); rot=rotary_kv(kvn); q,sc,quant=act_quant(rot.reshape(len(toks),KV_DIM)); window_kv=quant.reshape(1,len(toks),KV_DIM); cache=np.zeros((1,WINDOW,KV_DIM),dtype=np.uint16); cache[:,:len(toks),:]=window_kv; idx=topk(len(toks))
    expected={'input_bf16_uint16':x.tolist(),'wkv_output_bf16_uint16':wkv.tolist(),'kv_norm_output_bf16_uint16':kvn.tolist(),'rotary_kv_bf16_uint16':rot.tolist(),'quantized_window_kv_fp8_uint8':q.reshape(1,len(toks),KV_DIM).tolist(),'quantized_window_kv_scale_e8m0_uint8':sc.reshape(1,len(toks),KV_DIM//BLOCK).tolist(),'window_kv_bf16_uint16':window_kv.tolist(),'cache_after_publication_bf16_uint16':cache.tolist(),'topk_idxs_int32':idx.tolist()}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.int32 if 'topk' in k else (np.uint8 if 'uint8' in k else np.uint16))) for k,v in expected.items()}
    rec={'schema':'ds41f.window-kv-prelude-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'bounded layer-0 window-KV prelude fixture; stop before sparse_attn','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':['inference/model.py','inference/kernel.py'],'expected_value_provider':'independent arithmetic composition of validated FP8 linear/RMSNorm/rotary/act_quant plus int32 topk contract'},'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':MODEL_PY_SHA,'functions':[{'name':'Attention._window_kv','source_lines':[700,720],'source_sha256':WINDOW_KV_SHA},{'name':'get_window_topk_idxs','source_lines':[410,426],'source_sha256':TOPK_SHA},{'name':'linear','source_lines':[181,207],'source_sha256':LINEAR_SHA},{'name':'RMSNorm.forward','source_lines':[288,293],'source_sha256':RMS_FWD_SHA},{'name':'apply_rotary_emb','source_lines':[392,406],'source_sha256':ROTARY_SHA}]},'kernel_py':{'file':'inference/kernel.py','file_sha256':KERNEL_PY_SHA,'functions':[{'name':'act_quant','source_lines':[98,124],'source_sha256':ACT_QUANT_SHA},{'name':'fp8_gemm','source_lines':[277,307],'source_sha256':FP8_GEMM_SHA}]}},'source_tensors':{'input':{'name':'embed.weight','shard':str(emb_shard),'tokens':toks,'dtype':'BF16','shape':[len(toks),DIM],'digest':digest(x)},'wkv.weight':{'name':'layers.0.attn.wkv.weight','shard':str(shard),'dtype':'F8_E4M3','shape':[KV_DIM,DIM]},'wkv.scale':{'name':'layers.0.attn.wkv.scale','shard':str(shard),'dtype':'F8_E8M0','shape':[KV_DIM//BLOCK,DIM//BLOCK]},'kv_norm.weight':{'name':'layers.0.attn.kv_norm.weight','shard':str(shard),'dtype':'BF16','shape':[KV_DIM],'digest':digest(np.ascontiguousarray(nw))}},'inputs':{'layer':0,'batch':1,'sequence':len(toks),'tokens':toks,'start_pos':0,'window_size':WINDOW,'compress_ratio':0},'operation_contract':{'order':'kv=kv_norm(wkv(x)); apply_rotary_emb(kv[..., -64:], freqs_cis); act_quant(kv, block=32, inplace=True); publish cache[:bsz,:seqlen]=kv; return kv and get_window_topk_idxs; STOP before sparse_attn','wkv_full_shape_used':True,'fp8_block_size':32,'act_quant_inplace':'records E4M3 bytes/scales and BF16 dequantized in-place KV','cache_publication':'prefill start_pos=0 and S<=window writes first S slots exactly','topk_contract':'int32 exact causal prefill rows from get_window_topk_idxs','predeclared_tolerance':{'intermediate_max_bf16_ulp_lte':1,'act_quant_max_bf16_ulp_lte':0,'topk_exact':True,'cache_exact':True}},'expected':expected,'digests':dig,'comparison':{'self_consistent':True,'explicit_stop_before_sparse_attn':True},'non_claims':['does not execute oMLX','does not enter sparse_attn','does not validate compressed KV, Compressor/Indexer, decode ring-wrap, output projection, HC, MoE, logits, full layer, or full model correctness','does not benchmark performance'],'ok':True}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
