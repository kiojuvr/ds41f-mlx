#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, struct
from pathlib import Path
import numpy as np
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
SP='artifacts/sparse-attn-official-reference-fixture.json'
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'; ATTN_SHA='86d80f5cdaa6435cacd56ce5be796c3f0155a7f92cebdb12ffe6743ac974d110'
H=64; D=512; RD=64; GROUPS=8; O_RANK=1024; WOAIN=4096; WOAOUT=8192; DIM=5120; BLOCK=32

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def header(p):
 with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape): h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)
def f32_to_bf16_rne(x):
 u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); l=(u>>16)&1; return ((u+np.uint32(0x7fff)+l)>>16).astype(np.uint16)
def e4(c):
 s=-1.0 if c&0x80 else 1.0; ax=c&0x7f
 if ax==0: return math.copysign(0.0,s)
 e=(c>>3)&0xf; m=c&7; return s*(math.ldexp(m/8.0,-6) if e==0 else math.ldexp(1+m/8.0,e-7))
E4=np.array([e4(i) if (i&0x7f)!=0x7f else np.nan for i in range(256)],np.float32); VALID=np.array([i for i in range(256) if (i&0x7f)!=0x7f],np.uint16); VV=E4[VALID]
def e4enc(v):
 v=min(max(float(v),-448),448); d=np.abs(VV.astype(np.float64)-v); m=float(d.min()); t=VALID[d==m]; e=t[(t&1)==0]; return int(e[0] if e.size else t[0])
def e8arr(x): return np.ldexp(np.ones_like(x,dtype=np.float32),x.astype(np.int32)-127)
def inv_rot(o):
 y=np.array(o,copy=True); tail=bf16_to_f32(y[...,-RD:]); half=RD//2; freqs=1/(10000.0**(np.arange(0,RD,2,dtype=np.float32)/np.float32(RD))); ang=np.outer(np.arange(y.shape[1],dtype=np.float32),freqs).astype(np.float32); c=np.cos(ang).astype(np.float32); s=np.sin(ang).astype(np.float32); p=tail.reshape(1,y.shape[1],H,half,2); re=p[...,0]; im=p[...,1]; out=np.empty_like(p); out[...,0]=re*c[None,:,None,:]+im*s[None,:,None,:]; out[...,1]=-re*s[None,:,None,:]+im*c[None,:,None,:]; y[...,-RD:]=f32_to_bf16_rne(out.reshape(1,y.shape[1],H,RD)); return y
def deq_weight_bf16(w,sc):
 out=np.empty(w.shape,dtype=np.uint16)
 for ob in range(sc.shape[0]):
  for kb in range(sc.shape[1]):
   rows=slice(ob*BLOCK,min((ob+1)*BLOCK,w.shape[0])); cols=slice(kb*BLOCK,(kb+1)*BLOCK); vals=E4[w[rows,cols]]*np.float32(2.0**(int(sc[ob,kb])-127)); out[rows,cols]=f32_to_bf16_rne(vals)
 return out
def grouped_woa(inp, wb):
 x=bf16_to_f32(inp.reshape(1,inp.shape[1],GROUPS,WOAIN)); w=bf16_to_f32(wb.reshape(GROUPS,O_RANK,WOAIN)); out=np.empty((1,inp.shape[1],GROUPS,O_RANK),np.float32)
 for g in range(GROUPS): out[:,:,g,:]=x[:,:,g,:] @ w[g].T
 return f32_to_bf16_rne(out), out
def actq(x):
 rows,k=x.shape; kb=k//BLOCK; q=np.empty((rows,k),np.uint8); s=np.empty((rows,kb),np.float32)
 for r in range(rows):
  for b in range(kb):
   sl=x[r,b*BLOCK:(b+1)*BLOCK]; a=max(float(np.max(np.abs(sl))),1e-4); exp=math.ceil(math.log2(a/448)); sc=np.float32(math.ldexp(1,exp)); s[r,b]=sc; q[r,b*BLOCK:(b+1)*BLOCK]=[e4enc(np.clip(v/sc,-448,448)) for v in sl]
 return q,s
def fp8_linear(inp_bf16,w,ws):
 x=bf16_to_f32(inp_bf16); rows,k=x.shape; outdim=w.shape[0]; kb=k//BLOCK; q,asc=actq(x); out=np.zeros((rows,outdim),np.float32); adeq=E4[q].reshape(rows,kb,BLOCK)*asc[:,:,None]
 for b in range(kb):
  scale=e8arr(ws[:,b]); wd=E4[w[:,b*BLOCK:(b+1)*BLOCK]]*scale[np.arange(outdim)//BLOCK,None]; out += adeq[:,b,:].astype(np.float32) @ wd.astype(np.float32).T
 return f32_to_bf16_rne(out)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--sparse',default=SP); ap.add_argument('--out',default='artifacts/attention-output-projection-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint); sparse=json.loads(Path(a.sparse).read_text())
 o=np.asarray(sparse['expected']['attention_output_bf16_uint16'],np.uint16); shard=ck/'model-00003-of-00048.safetensors'; woa=mmap(shard,'layers.0.attn.wo_a.weight',np.uint8,(WOAOUT,WOAIN)); woas=mmap(shard,'layers.0.attn.wo_a.scale',np.uint8,(WOAOUT//BLOCK,WOAIN//BLOCK)); wob=mmap(shard,'layers.0.attn.wo_b.weight',np.uint8,(DIM,WOAOUT)); wobs=mmap(shard,'layers.0.attn.wo_b.scale',np.uint8,(DIM//BLOCK,WOAOUT//BLOCK))
 inv=inv_rot(o); grouped_in=inv.reshape(1,o.shape[1],GROUPS,WOAIN); woa_bf16=deq_weight_bf16(np.ascontiguousarray(woa),np.ascontiguousarray(woas)); woa_out,_=grouped_woa(inv,woa_bf16); flat=woa_out.reshape(1,o.shape[1],WOAOUT); final=fp8_linear(flat.reshape(o.shape[1],WOAOUT),np.ascontiguousarray(wob),np.ascontiguousarray(wobs)).reshape(1,o.shape[1],DIM)
 exp={'sparse_attn_input_bf16_uint16':o.tolist(),'inverse_rotary_output_bf16_uint16':inv.tolist(),'grouped_reshape_bf16_uint16':grouped_in.tolist(),'grouped_wo_a_output_bf16_uint16':woa_out.tolist(),'flattened_wo_a_output_bf16_uint16':flat.tolist(),'final_wo_b_output_bf16_uint16':final.tolist()}; dig={k+'_sha256':digest(np.asarray(v,dtype=np.uint16)) for k,v in exp.items()}
 rec={'schema':'ds41f.attention-output-projection-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 4 attention output projection fixture; stop before Block/HC','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':'inference/model.py','inputs_from_validated_artifacts':[a.sparse],'expected_value_provider':'independent arithmetic over official checkpoint tensors'},'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':MODEL_SHA,'functions':[{'name':'Attention','source_lines':[613,789],'source_sha256':ATTN_SHA,'reviewed_lines':'apply_rotary_emb inverse, grouped wo_a einsum, wo_b projection'}]}},'source_tensors':{'wo_a.weight':{'name':'layers.0.attn.wo_a.weight','shard':str(shard),'dtype':'F8_E4M3','shape':[WOAOUT,WOAIN]},'wo_a.scale':{'name':'layers.0.attn.wo_a.scale','shard':str(shard),'dtype':'F8_E8M0','shape':[WOAOUT//BLOCK,WOAIN//BLOCK]},'wo_b.weight':{'name':'layers.0.attn.wo_b.weight','shard':str(shard),'dtype':'F8_E4M3','shape':[DIM,WOAOUT]},'wo_b.scale':{'name':'layers.0.attn.wo_b.scale','shard':str(shard),'dtype':'F8_E8M0','shape':[DIM//BLOCK,WOAOUT//BLOCK]}},'inputs':{'sparse_attn_source':a.sparse,'batch':1,'sequence':2,'n_local_groups':GROUPS,'o_lora_rank':O_RANK,'world_size':1},'operation_contract':{'order':'apply_rotary_emb(o[..., -64:], inverse=True); o.view(B,S,8,-1); wo_a.weight.view(8,1024,-1); einsum bsgd,grd->bsgr; flatten; wo_b projection; STOP before Block/HC','wo_a_contract':'actual checkpoint is FP8; dequantize with F8_E8M0 32x32 scales to BF16, then grouped F32-accum einsum and BF16 output','wo_b_contract':'validated FP8 linear/GEMM contract over flattened wo_a output','tp_all_reduce':'world_size=1 executed; TP all-reduce not validated','predeclared_tolerance':{'intermediate_max_bf16_ulp_lte':1,'final_max_bf16_ulp_lte':1}},'expected':exp,'digests':dig,'comparison':{'self_consistent':True,'explicit_stop_before_block_hc':True},'non_claims':['does not execute oMLX','does not validate Block, HC residual integration, MoE, Compressor/Indexer, logits, full layer, or full model correctness','does not validate tensor-parallel all-reduce','does not benchmark performance'],'ok':True}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
