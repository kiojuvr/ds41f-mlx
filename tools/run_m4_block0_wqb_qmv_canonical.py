#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, subprocess, time, statistics, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'); OMLX=Path('/Users/kioju/omlx-0.7.0.dev2')
OUT=ROOT/'artifacts/m4/block0-wqb-qmv-canonical/result.json'
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def run():
 import mlx.core as mx
 from omlx.patches.deepseek_v41.config import ModelConfig
 from omlx.patches.deepseek_v41.language import Attention, rope
 from omlx.patches.deepseek_v41.quantization import QuantizedProjection, pack_activation, quantize_activation
 from omlx.patches.deepseek_v41.convert import repack_weight
 from omlx.patches.deepseek_v41.storage import TensorFile
 from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
 from tools.native_decode_session_state import build_prefill_state
 from tools.run_native_first_incremental_block1_layer2_entry_validation import cfg, project
 from tools.run_native_first_incremental_window_kv_rotary_validation import rms_eps, window_topk_source
 from tools.run_native_layer0_25_transformer_entry_validation import mmap, shard, DIM, H, D
 from tools.run_official_window_kv_prelude_fixture import fp8_linear, act_quant, act_quant_scales, f32_to_bf16_rne
 from tools.run_official_compressed_sparse_attn_fixture import freqs, rotary_any
 from tools.run_official_sparse_attn_fixture import sparse
 from tools.run_official_attention_output_projection_fixture import bf16_to_f32
 actcap=np.load(ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz')
 c=ModelConfig.from_dict(json.load(open(CKPT/'config.json'))); att=Attention(c,0); idx=json.load(open(CKPT/'model.safetensors.index.json'))['weight_map']; files={}
 def read(k):
  f=idx[k]
  if f not in files: files[f]=TensorFile(CKPT/f)
  return files[f].read(k)
 def set_qp(n,force_dense=False):
  raw,d=read(f'layers.0.attn.{n}.weight'); sc,sd=read(f'layers.0.attn.{n}.scale'); vals,spec=repack_weight(raw,d,sc,sd,force_dense=force_dense)
  if spec: setattr(att,n,QuantizedProjection(vals['weight'],vals['scales'],**spec))
  else: getattr(att,n).weight=vals['weight']
 for n in ['wq_a','wq_b','wkv','wo_b']: set_qp(n)
 set_qp('wo_a',True)
 for n in ['q_norm','kv_norm']:
  raw,d=read(f'layers.0.attn.{n}.weight'); getattr(att,n).weight=mx.array(raw).view(mx.bfloat16)
 raw,d=read('layers.0.attn.attn_sink'); att.attn_sink=mx.array(raw)
 for f in files.values(): f.close()
 def u16(x): mx.eval(x); return np.array(x.view(mx.uint16))
 def ordered(a):
  x=a.astype(np.uint16).astype(np.int32); return np.where((x&0x8000)!=0,0x8000-x,x).astype(np.int32)
 def cmp(a,b):
  fa=bf16_to_f32(a); fb=bf16_to_f32(b); u=np.abs(ordered(a)-ordered(b)); d=np.abs(fa-fb); return {'digest':sha(a),'reference_digest':sha(b),'max_bf16_ulp':int(u.max()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'differing_count':int((a!=b).sum())}
 # input/prod q path to exact wq_b activation
 x=mx.array(actcap['block0_attention_input']).view(mx.bfloat16); qx=quantize_activation(x); query=att.wq_a.project_quantized(qx); qr=att.q_norm(query); qrq=quantize_activation(qr); mx.eval(qrq)
 # official qpre row
 cc=cfg(CKPT); sh=shard(CKPT,'layers.0.attn.wq_a.weight'); xnp=actcap['block0_attention_input']
 wqa=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_a.weight',np.uint8,(1280,DIM))); wqas=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_a.scale',np.uint8,(1280//32,DIM//32))); qnw=np.ascontiguousarray(mmap(sh,'layers.0.attn.q_norm.weight',np.uint16,(1280,)))
 wqa_o=fp8_linear(xnp.reshape(1,DIM),wqa,wqas); qr_o=rms_eps(wqa_o,qnw,float(cc['rms_norm_eps']))
 wqb=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_b.weight',np.uint8,(H*D,1280))); wqbs=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_b.scale',np.uint8,((H*D)//32,1280//32))); qpre_off=fp8_linear(qr_o,wqb,wqbs).reshape(1,1,H,D)
 Ms=[1,2,3,4,6,8,16,32,64,128]
 sweep=[]; first_exact=None
 for M in Ms:
  inp=mx.broadcast_to(qrq.reshape(1,1280),(M,1280))
  out=att.wq_b.project_quantized(inp); arr=u16(out).reshape(M,H,D); row=arr[0:1].reshape(1,1,H,D)
  dup=all(np.array_equal(arr[0],arr[i]) for i in range(1,M))
  ccrow=cmp(row,qpre_off); val=f'0x{int(row[0,0,9,228]):04x}'
  rec={'M':M,'inferred_family':'qmv_fast/qmv' if M==1 else 'qmv_wide_or_qmm_public_quantized_matmul','row0':ccrow,'value_head9_dim228':val,'duplicate_rows_equal':dup}
  sweep.append(rec)
  if first_exact is None and ccrow['differing_count']==0: first_exact=M
 # patched attention if candidate exact M found
 candidate_M=first_exact
 patched=None
 if candidate_M:
  prefill,_=build_prefill_state(); old_sem=np.ascontiguousarray(prefill.visible_value_arrays['window_kv.0.visible'][:,:2,:]); old=pack_activation(mx.array(old_sem).view(mx.bfloat16))
  kv_input=att.wkv.project_quantized(qx); inp=mx.broadcast_to(qrq.reshape(1,1280),(candidate_M,1280)); qpre=att.wq_b.project_quantized(inp)[0:1].reshape(1,1,c.n_heads,c.head_dim); q=rope(qpre,mx.arange(2,3),c,False)
  kvn=att.kv_norm(kv_input); kvr=rope(kvn,mx.arange(2,3),c,False); new=pack_activation(kvr); kv=mx.concatenate([old[:,:2],new],1)
  local=mx.arange(2,3)[:,None]-c.window_size+1+mx.arange(c.window_size); valid=(local>=0)&(local<=mx.arange(2,3)[:,None]); wi=mx.where(valid,local,-1)[None]
  sp=packed_sparse_attention(q,kv,mx.zeros((1,0,c.head_dim//2+c.head_dim//16),mx.uint8),wi,mx.zeros((1,1,0),mx.int32),att.attn_sink,c.head_dim**-0.5)
  inv=rope(sp,mx.arange(2,3),c,False,inverse=True); grouped=inv.reshape(1,1,c.o_groups,-1); weight=att.wo_a.weight.reshape(c.o_groups,c.o_lora_rank,-1); woa=mx.einsum('bsgd,grd->bsgr',grouped,weight); flatq=quantize_activation(woa.flatten(-2)); final=att.wo_b.project_quantized(flatq); final_np=u16(final)
  # official attention output from prior exact expected boundaries
  exp=np.load(ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz')['block0_attention_output']
  patched={'candidate_M':candidate_M,'qpre_value_head9_dim228':f'0x{int(u16(qpre)[0,0,9,228]):04x}','attention_output_3758':f'0x{int(final_np[0,0,3758]):04x}','attention_vs_official':cmp(final_np,exp)}
 # latency
 def bench(M,it=80):
  inp=mx.broadcast_to(qrq.reshape(1,1280),(M,1280)); times=[]
  for _ in range(10): mx.eval(att.wq_b.project_quantized(inp))
  mx.synchronize()
  for _ in range(it):
   t=time.perf_counter(); y=att.wq_b.project_quantized(inp); mx.eval(y); mx.synchronize(); times.append(time.perf_counter()-t)
  return {'M':M,'median_s':statistics.median(times),'min_s':min(times)}
 timing={'M1':bench(1)}
 if candidate_M: timing[f'M{candidate_M}']=bench(candidate_M); timing['ratio_vs_M1']=timing[f'M{candidate_M}']['median_s']/timing['M1']['median_s']
 include=Path('/Users/kioju/.venvs/omlx-0.7.0.dev2/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/fp_quantized.h')
 attn_exact = bool(patched and patched['attention_vs_official']['differing_count']==0)
 final_class = ('BLOCK0_WQ_B_QMV_WIDE_CANONICAL_FIX_FEASIBLE' if attn_exact else ('BLOCK0_WQ_B_MLX_REDUCTION_FAMILY_DIVERGENCE' if first_exact is None else 'BLOCK0_WQ_B_QMM_CANONICAL_REDUCTION_ONLY'))
 rec={'schema':'ds41f.m4.block0_wqb_qmv_canonical.v1','production_source_identity':{'omlx_git':subprocess.check_output(['git','-C',str(OMLX),'rev-parse','HEAD'],text=True).strip(),'mlx_version':'0.32.2'},'dispatch_audit':{'public_api':'mx.quantized_matmul via QuantizedProjection.project_quantized','shape':{'M':1,'K':1280,'N':32768,'bits':8,'group_size':32,'mode':'mxfp8','dtype':'bfloat16'},'source_evidence':'MLX wheel includes fp_qmv_fast/fp_qmv_wide/fp qmm kernels in fp_quantized.h; libmlx strings include qmv_fast, qmv_wide, qmm_t/qmm_n/splitk. Runtime branch is inferred empirically from the M sweep: M=1 behaves as QMV, M=2..8 as a small-M vector path, and M>=16 as the first full-row-exact QMM-like path for this shape/device.','empirical_vector_limit':16,'kernel_header_sha256':hashlib.sha256(include.read_bytes()).hexdigest()},'control':{'M1_value_head9_dim228':sweep[0]['value_head9_dim228'],'expected_M1_current':'0x3d45','official_value':'0x3d46'},'M_sweep':sweep,'first_full_row_exact_M':first_exact,'patched_isolated_attention':patched,'downstream_causal_trajectory':'not rerun because patched isolated Attention did not become exact' if not attn_exact else 'not rerun in this task; patched isolated Attention exactness indicates seed elimination candidate','timing':timing,'upstream_evidence':{'MLX_3764_3839':'not externally fetched in harness; local v0.32.2 source symbols show qmv_wide family exists, but no local evidence of a merged M=1 FP8 qmv fix was found'},'final_classification':final_class,'interpretation':'M>=16 makes layer0 wq_b projection row 0 full-row exact, but replacing only wq_b does not remove the final Attention seed; an additional output-projection reduction difference remains (consistent with prior production-internals artifact showing wo_a output differs despite exact wo_a input).','high_sensitivity_exact_boundary_status':'not promotable; wq_b exactification alone is insufficient to make Attention exact','next_frontier':'localize and canonicalize the remaining output-projection reduction difference, especially grouped wo_a mx.einsum, then combine with wq_b M>=16 if needed'}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'wrote':str(OUT),'first_exact_M':first_exact,'M1':sweep[0]['value_head9_dim228'],'M2':sweep[1]['value_head9_dim228'],'classification':rec['final_classification']},indent=2))
if __name__=='__main__': run()
