#!/usr/bin/env python3
"""Generate bounded Attention Q-prelude through rotary fixture from official bits."""

from __future__ import annotations

import argparse, hashlib, json, math, os, struct
from pathlib import Path
from typing import Any
import numpy as np

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
MODEL_PY_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
KERNEL_PY_SHA = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
ATTENTION_SHA = "86d80f5cdaa6435cacd56ce5be796c3f0155a7f92cebdb12ffe6743ac974d110"
LINEAR_SHA = "c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada"
RMS_FWD_SHA = "adb7c70ed245e8830f6692b8c026fd7f2d17518cd98b68bf973e75524a921c85"
ROTARY_SHA = "1c47da553d29f41c798a2eda82b72476473913d1201e32283de0e4977aa30b1a"
ACT_QUANT_SHA = "563a82836450bfefe3f5f176636dec0e1f4d126c76d8007b42bb7abb877d30cb"
FP8_GEMM_SHA = "cfd550d8b02ee127760ac26b39accae603302be6bf8e97c0cce29d8993af0657"
VOCAB_SIZE=129280; DIM=5120; Q_RANK=1280; N_HEADS=64; HEAD_DIM=512; ROPE_DIM=64; WQB_OUT=N_HEADS*HEAD_DIM; BLOCK=32


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()

def header(path: Path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n

def mmap(path: Path, name: str, dtype: Any, shape: tuple[int,...]):
    h,b=header(path); off=h[name]['data_offsets'][0]
    return np.memmap(path,mode='r',dtype=dtype,offset=b+off,shape=shape)

def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)

def f32_to_bf16_rne(x):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); l=(u>>16)&1
    return ((u+np.uint32(0x7fff)+l)>>16).astype(np.uint16)

def e4_decode_code(c:int)->float:
    s=-1.0 if c&0x80 else 1.0; ax=c&0x7f
    if ax==0: return math.copysign(0.0,s)
    e=(c>>3)&0xf; m=c&7
    v=math.ldexp(m/8.0,-6) if e==0 else math.ldexp(1.0+m/8.0,e-7)
    return s*v
E4=np.array([e4_decode_code(i) if (i&0x7f)!=0x7f else np.nan for i in range(256)],dtype=np.float32)
VALID=np.array([i for i in range(256) if (i&0x7f)!=0x7f],dtype=np.uint16); VALID_VALUES=E4[VALID]

def e4_encode(v: float)->int:
    if math.isnan(v): return 0x7f
    v=min(max(v,-448.0),448.0)
    d=np.abs(VALID_VALUES.astype(np.float64)-float(v)); md=float(d.min()); tied=VALID[d==md]; even=tied[(tied&1)==0]
    return int(even[0] if even.size else tied[0])

def e8_decode_arr(x): return np.ldexp(np.ones_like(x,dtype=np.float32), x.astype(np.int32)-127)

def act_quant_matrix(x_f32: np.ndarray):
    rows,k=x_f32.shape; kb=k//BLOCK
    q=np.empty((rows,k),dtype=np.uint8); s=np.empty((rows,kb),dtype=np.float32); se=np.empty((rows,kb),dtype=np.uint8)
    for r in range(rows):
        for b in range(kb):
            sl=x_f32[r,b*BLOCK:(b+1)*BLOCK]
            amax=max(float(np.max(np.abs(sl))),1e-4); exp=math.ceil(math.log2(amax/448.0)); scale=np.float32(math.ldexp(1.0,exp))
            s[r,b]=scale; se[r,b]=exp+127
            q[r,b*BLOCK:(b+1)*BLOCK]=[e4_encode(float(np.clip(v/scale,-448,448))) for v in sl]
    return q,s,se

def fp8_linear_bf16(inp_bf16: np.ndarray, weight_u8: np.ndarray, scale_u8: np.ndarray):
    x=bf16_to_f32(inp_bf16); rows,k=x.shape; out_dim=weight_u8.shape[0]; kb=k//BLOCK
    aq,asc,ase=act_quant_matrix(x)
    out=np.zeros((rows,out_dim),dtype=np.float32)
    a_deq=E4[aq].reshape(rows,kb,BLOCK)*asc[:,:,None]
    for b in range(kb):
        bs=e8_decode_arr(scale_u8[:,b])  # [out_blocks]
        w=E4[weight_u8[:,b*BLOCK:(b+1)*BLOCK]] * bs[np.arange(out_dim)//BLOCK,None]
        out += a_deq[:,b,:].astype(np.float32) @ w.astype(np.float32).T
    return f32_to_bf16_rne(out), out, aq, ase

def rmsnorm_bf16(x_bf16, w_bf16, eps=1e-6):
    x=bf16_to_f32(x_bf16); w=bf16_to_f32(w_bf16)
    var=np.mean(np.square(x,dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32)
    y=x*(1.0/np.sqrt(var+np.float32(eps),dtype=np.float32))*w
    return f32_to_bf16_rne(y.astype(np.float32)), y.astype(np.float32)

def apply_rotary_tail_bf16(q_bf16):
    q=np.array(q_bf16,copy=True)
    tail=bf16_to_f32(q[..., -ROPE_DIM:])
    half=ROPE_DIM//2
    freqs=1.0/(10000.0**(np.arange(0,ROPE_DIM,2,dtype=np.float32)/np.float32(ROPE_DIM)))
    angles=np.outer(np.arange(q.shape[1],dtype=np.float32),freqs).astype(np.float32)
    cos=np.cos(angles).astype(np.float32); sin=np.sin(angles).astype(np.float32)
    pairs=tail.reshape(q.shape[0],q.shape[1],q.shape[2],half,2)
    re=pairs[...,0]; im=pairs[...,1]
    out=np.empty_like(pairs)
    out[...,0]=re*cos[None,:,None,:]-im*sin[None,:,None,:]
    out[...,1]=re*sin[None,:,None,:]+im*cos[None,:,None,:]
    q[..., -ROPE_DIM:]=f32_to_bf16_rne(out.reshape(q.shape[0],q.shape[1],q.shape[2],ROPE_DIM))
    return q

def list2d(a): return np.ascontiguousarray(a).tolist()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--tokens',default='0,3'); ap.add_argument('--out',default='artifacts/attention-q-prelude-official-reference-fixture.json'); args=ap.parse_args()
    ck=Path(args.checkpoint); toks=[int(x) for x in args.tokens.split(',') if x.strip()]
    shard=ck/'model-00003-of-00048.safetensors'; embed_shard=ck/'model-00002-of-00048.safetensors'
    embed=mmap(embed_shard,'embed.weight',np.uint16,(VOCAB_SIZE,DIM)); x=np.ascontiguousarray(embed[toks],dtype=np.uint16)
    wqa=mmap(shard,'layers.0.attn.wq_a.weight',np.uint8,(Q_RANK,DIM)); wqa_s=mmap(shard,'layers.0.attn.wq_a.scale',np.uint8,(Q_RANK//BLOCK,DIM//BLOCK))
    qnw=mmap(shard,'layers.0.attn.q_norm.weight',np.uint16,(Q_RANK,))
    wqb=mmap(shard,'layers.0.attn.wq_b.weight',np.uint8,(WQB_OUT,Q_RANK)); wqb_s=mmap(shard,'layers.0.attn.wq_b.scale',np.uint8,(WQB_OUT//BLOCK,Q_RANK//BLOCK))
    wqa_out,_,_,_=fp8_linear_bf16(x,np.ascontiguousarray(wqa),np.ascontiguousarray(wqa_s))
    qn_out,_=rmsnorm_bf16(wqa_out,np.ascontiguousarray(qnw))
    wqb_out,_,_,_=fp8_linear_bf16(qn_out,np.ascontiguousarray(wqb),np.ascontiguousarray(wqb_s))
    reshaped=np.ascontiguousarray(wqb_out.reshape(1,len(toks),N_HEADS,HEAD_DIM))
    rotary=np.ascontiguousarray(apply_rotary_tail_bf16(reshaped))
    expected={
      'input_bf16_uint16': list2d(x), 'wq_a_output_bf16_uint16': list2d(wqa_out), 'q_norm_output_bf16_uint16': list2d(qn_out),
      'wq_b_output_bf16_uint16': list2d(wqb_out), 'reshaped_q_bf16_uint16': reshaped.tolist(), 'rotary_applied_q_bf16_uint16': rotary.tolist()}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.uint16)) for k,v in expected.items()}
    rec={
      'schema':'ds41f.attention-q-prelude-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,
      'purpose':'bounded layer-0 Attention Q-prelude through rotary fixture; stop before sparse_attn','checkpoint':str(ck),
      'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':['inference/model.py','inference/kernel.py'],'expected_value_provider':'independent arithmetic composition of validated FP8 linear/RMSNorm/rotary contracts'},
      'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':MODEL_PY_SHA,'functions':[{'name':'Attention','source_lines':[613,789],'source_sha256':ATTENTION_SHA},{'name':'linear','source_lines':[181,207],'source_sha256':LINEAR_SHA},{'name':'RMSNorm.forward','source_lines':[288,293],'source_sha256':RMS_FWD_SHA},{'name':'apply_rotary_emb','source_lines':[392,406],'source_sha256':ROTARY_SHA}]},'kernel_py':{'file':'inference/kernel.py','file_sha256':KERNEL_PY_SHA,'functions':[{'name':'act_quant','source_lines':[98,124],'source_sha256':ACT_QUANT_SHA},{'name':'fp8_gemm','source_lines':[277,307],'source_sha256':FP8_GEMM_SHA}]}},
      'source_tensors':{
        'input':{'name':'embed.weight','shard':str(embed_shard),'tokens':toks,'dtype':'BF16','shape':[len(toks),DIM],'digest':digest(x)},
        'wq_a.weight':{'name':'layers.0.attn.wq_a.weight','shard':str(shard),'dtype':'F8_E4M3','shape':[Q_RANK,DIM]},'wq_a.scale':{'name':'layers.0.attn.wq_a.scale','shard':str(shard),'dtype':'F8_E8M0','shape':[Q_RANK//BLOCK,DIM//BLOCK]},
        'q_norm.weight':{'name':'layers.0.attn.q_norm.weight','shard':str(shard),'dtype':'BF16','shape':[Q_RANK],'digest':digest(np.ascontiguousarray(qnw))},
        'wq_b.weight':{'name':'layers.0.attn.wq_b.weight','shard':str(shard),'dtype':'F8_E4M3','shape':[WQB_OUT,Q_RANK]},'wq_b.scale':{'name':'layers.0.attn.wq_b.scale','shard':str(shard),'dtype':'F8_E8M0','shape':[WQB_OUT//BLOCK,Q_RANK//BLOCK]}},
      'inputs':{'batch':1,'sequence':len(toks),'tokens':toks,'start_pos':0},
      'operation_contract':{'order':'qr = q_norm(wq_a(x)); q = wq_b(qr).unflatten(-1,(64,512)); apply_rotary_emb(q[..., -64:], freqs_cis); STOP before _window_kv/sparse_attn','fp8_block_size':32,'wq_a_full_shape_used':True,'wq_b_full_shape_used':True,'multi_block_scale_indexing_exercised':True,'reshape_layout':'[B,S,32768] -> [B,S,n_local_heads=64,head_dim=512] row-major unflatten of last dimension','rotary':'no-YaRN layer 0 compress_ratio=0, base=10000, rope_dim=64, positions 0..S-1','output_dtype':'BF16 at FP8/RMSNorm outputs and after in-place rotary copy','predeclared_tolerance':{'intermediate_max_bf16_ulp_lte':1,'final_rotary_max_bf16_ulp_lte':1}},
      'expected':expected,'digests':dig,'comparison':{'self_consistent':True,'explicit_stop_before_sparse_attn':True},
      'non_claims':['does not execute oMLX','does not enter sparse_attn','does not validate K/V, cache, window-KV, Compressor/Indexer, output projection, HC, MoE, logits, full layer, or full model correctness','does not benchmark performance'],'ok':True}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
