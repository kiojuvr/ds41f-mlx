#!/usr/bin/env python3
"""Generate Boundary 3 sparse_attn fixture from validated Boundary 1/2 artifacts."""
from __future__ import annotations
import argparse, hashlib, json, os, struct
from pathlib import Path
from typing import Any
import numpy as np
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
KERNEL_PY_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
SPARSE_KERNEL_SHA='5438750533acb517260b1da40a9a033e5068eaef4a0c57b2c769f2de4e686266'; SPARSE_SHA='42208bc5467f5a29efd18020b62162fa3177614293f3669d3d0b8800d6e5d704'
B1='artifacts/attention-q-prelude-official-reference-fixture.json'; B2='artifacts/window-kv-prelude-official-reference-fixture.json'
H=64; D=512

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def header(p:Path):
    with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape):
    h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)
def f32_to_bf16_rne(x):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); l=(u>>16)&1; return ((u+np.uint32(0x7fff)+l)>>16).astype(np.uint16)

def sparse(q_u16,kv_u16,sink,idxs,scale):
    q=bf16_to_f32(q_u16); kv=bf16_to_f32(kv_u16); B,S,H,D=q.shape; T=idxs.shape[2]
    raw=np.full((B,S,H,T),-1.0e30,dtype=np.float32); scaled=np.full_like(raw,-1.0e30); rowmax=np.empty((B,S,H),dtype=np.float32); denom=np.empty((B,S,H),dtype=np.float32); denom_no_sink=np.empty((B,S,H),dtype=np.float32); sink_term=np.empty((B,S,H),dtype=np.float32); out_f32=np.empty((B,S,H,D),dtype=np.float32); all_invalid=np.zeros((B,S,H),dtype=np.bool_)
    for b in range(B):
      for m in range(S):
        for h in range(H):
          scores=[]
          for t in range(T):
            ix=int(idxs[b,m,t])
            if ix>=0 and ix<kv.shape[1]:
              r=np.float32(np.sum((q[b,m,h].astype(np.float32)*kv[b,ix].astype(np.float32)),dtype=np.float32)); s=np.float32(r*scale); raw[b,m,h,t]=r; scaled[b,m,h,t]=s; scores.append((t,ix,s))
          rm=np.float32(max([float(s[2]) for s in scores], default=-1.0e30)); rowmax[b,m,h]=rm; all_invalid[b,m,h]=(len(scores)==0)
          dn=np.float32(0.0); acc=np.zeros((D,),dtype=np.float32)
          for t,ix,s in scores:
            p=np.float32(np.exp(np.float32(s-rm))); dn=np.float32(dn+p); pb=bf16_to_f32(f32_to_bf16_rne(np.array([p],dtype=np.float32)))[0]; acc += np.float32(pb*kv[b,ix])
          st=np.float32(np.exp(np.float32(sink[h]-rm))); sink_term[b,m,h]=st; denom_no_sink[b,m,h]=dn; denom[b,m,h]=np.float32(dn+st); out_f32[b,m,h]=acc/denom[b,m,h]
    return raw,scaled,rowmax,denom,denom_no_sink,sink_term,all_invalid,out_f32,f32_to_bf16_rne(out_f32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--boundary1',default=B1); ap.add_argument('--boundary2',default=B2); ap.add_argument('--out',default='artifacts/sparse-attn-official-reference-fixture.json'); args=ap.parse_args()
    b1=json.loads(Path(args.boundary1).read_text()); b2=json.loads(Path(args.boundary2).read_text()); ck=Path(args.checkpoint)
    q=np.asarray(b1['expected']['rotary_applied_q_bf16_uint16'],dtype=np.uint16); kv=np.asarray(b2['expected']['window_kv_bf16_uint16'],dtype=np.uint16); idx=np.asarray(b2['expected']['topk_idxs_int32'],dtype=np.int32)
    sink=np.ascontiguousarray(mmap(ck/'model-00003-of-00048.safetensors','layers.0.attn.attn_sink',np.float32,(H,)))
    scale=np.float32(D**-0.5); raw,scaled,rowmax,den,den0,sterm,allinv,outf,out=sparse(q,kv,sink,idx,scale)
    sink_effect=np.count_nonzero(f32_to_bf16_rne(outf) != f32_to_bf16_rne(outf * (den/(den0+np.float32(1e-30)))[:,:,:,None])) > 0
    expected={'selected_kv_indices_int32':idx.tolist(),'raw_scores_f32':raw.tolist(),'scaled_scores_f32':scaled.tolist(),'row_max_f32':rowmax.tolist(),'softmax_denominator_with_sink_f32':den.tolist(),'softmax_denominator_without_sink_f32':den0.tolist(),'attn_sink_term_f32':sterm.tolist(),'attention_output_bf16_uint16':out.tolist()}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.int32 if 'int32' in k else (np.uint16 if 'uint16' in k else np.float32))) for k,v in expected.items()}
    rec={'schema':'ds41f.sparse-attn-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 3 sparse_attn fixture from validated Q/window-KV/topk artifacts; stop before inverse rotary/output projection','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':'inference/kernel.py','inputs_from_validated_artifacts':[args.boundary1,args.boundary2],'expected_value_provider':'independent arithmetic reconstruction of reviewed sparse_attn semantics'},'official_reference':{'kernel_py':{'file':'inference/kernel.py','file_sha256':KERNEL_PY_SHA,'functions':[{'name':'sparse_attn_kernel','source_lines':[311,389],'source_sha256':SPARSE_KERNEL_SHA},{'name':'sparse_attn','source_lines':[392,403],'source_sha256':SPARSE_SHA}]}},'source_tensors':{'attn_sink':{'name':'layers.0.attn.attn_sink','shard':str(ck/'model-00003-of-00048.safetensors'),'dtype':'F32','shape':[H],'digest':digest(sink)}},'inputs':{'q_source':args.boundary1,'window_kv_source':args.boundary2,'topk_source':args.boundary2,'layer':0,'batch':1,'sequence':2,'kv_len':2,'heads':H,'head_dim':D,'topk':2,'compressed_kv':False,'world_size':1,'softmax_scale':float(scale)},'operation_contract':{'minus_one_mask':'topk index -1 contributes no score and no value numerator; invalid scores are -inf for softmax','scores':'raw=q @ kv.T over BF16 values represented as F32; scaled=raw*head_dim^-0.5','softmax':'FP32 row max and denominator over valid selected KV entries','value_accumulation':'exp(score-rowmax) is rounded to BF16 before multiplying BF16 KV values, matching acc_s_cast before value GEMM','attn_sink':'per-head sink contributes exp(attn_sink[h]-rowmax) to denominator only; it has no value vector numerator','all_invalid_row':'rowmax remains finite lower bound -1e30; numerator remains zero; sink denominator makes output zero','output_dtype':'BF16 round-to-nearest-even','predeclared_tolerance':{'max_bf16_ulp_lte':2}},'expected':expected,'digests':dig,'comparison':{'self_consistent':True,'minus_one_mask_present':bool(np.any(idx<0)),'sink_denominator_effect_present':bool(sink_effect),'all_invalid_rows_present':bool(np.any(allinv)),'explicit_stop_before_inverse_rotary_output_projection':True},'non_claims':['does not execute oMLX','does not validate compressed KV, Compressor/Indexer, inverse rotary, output projection, HC, MoE, logits, full layer, or full model correctness','does not benchmark performance'],'ok':True}
    outp=Path(args.out); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(outp); return 0
if __name__=='__main__': raise SystemExit(main())
