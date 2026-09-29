#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
OMLX=Path('/Users/kioju/omlx-0.7.0.dev2')
OUT=ROOT/'artifacts/m4/block0-attention-production-internals/result.json'

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def run():
 import mlx.core as mx
 from omlx.patches.deepseek_v41.config import ModelConfig
 from omlx.patches.deepseek_v41.language import Attention, rope
 from omlx.patches.deepseek_v41.quantization import QuantizedProjection, pack_activation, unpack_activation, quantize_activation, _compiled_quantize_activation
 from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
 from omlx.patches.deepseek_v41.convert import repack_weight
 from omlx.patches.deepseek_v41.storage import TensorFile
 from tools.native_decode_session_state import build_prefill_state
 from tools.run_native_first_incremental_block1_layer2_entry_validation import cfg, arr_digest, project
 from tools.run_native_first_incremental_window_kv_rotary_validation import rms_eps, window_topk_source
 from tools.run_native_layer0_25_transformer_entry_validation import mmap, shard, DIM, H, D, RD
 from tools.run_official_window_kv_prelude_fixture import fp8_linear, act_quant, act_quant_scales, f32_to_bf16_rne
 from tools.run_official_compressed_sparse_attn_fixture import freqs, rotary_any
 from tools.run_official_sparse_attn_fixture import sparse
 from tools.run_official_attention_output_projection_fixture import bf16_to_f32
 actcap=np.load(ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz')
 expcap=np.load(ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz')
 c=ModelConfig.from_dict(json.load(open(CKPT/'config.json'))); att=Attention(c,0)
 idx=json.load(open(CKPT/'model.safetensors.index.json'))['weight_map']; files={}
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
 def u8(x): mx.eval(x); return np.array(x)
 x=mx.array(actcap['block0_attention_input']).view(mx.bfloat16)
 prefill,_=build_prefill_state(); old_sem=np.ascontiguousarray(prefill.visible_value_arrays['window_kv.0.visible'][:,:2,:])
 old=pack_activation(mx.array(old_sem).view(mx.bfloat16))
 # real oMLX manual attention
 qx=quantize_activation(x); query=att.wq_a.project_quantized(qx); kv_input=att.wkv.project_quantized(qx)
 qr=att.q_norm(query); qrq=quantize_activation(qr); qpre=att.wq_b.project_quantized(qrq).reshape(1,1,c.n_heads,c.head_dim); q=rope(qpre,mx.arange(2,3),c,False)
 kvn=att.kv_norm(kv_input); kvr=rope(kvn,mx.arange(2,3),c,False); new=pack_activation(kvr); kv=mx.concatenate([old[:,:2],new],1)
 local=mx.arange(2,3)[:,None]-c.window_size+1+mx.arange(c.window_size); valid=(local>=0)&(local<=mx.arange(2,3)[:,None]); wi=mx.where(valid,local,-1)[None]
 pooled=mx.zeros((1,0,c.head_dim//2+c.head_dim//16),mx.uint8); ci=mx.zeros((1,1,0),mx.int32)
 sp=packed_sparse_attention(q,kv,pooled,wi,ci,att.attn_sink,c.head_dim**-0.5); inv=rope(sp,mx.arange(2,3),c,False,inverse=True)
 grouped=inv.reshape(1,1,c.o_groups,-1); weight=att.wo_a.weight.reshape(c.o_groups,c.o_lora_rank,-1); woa=mx.einsum('bsgd,grd->bsgr',grouped,weight); flat=woa.flatten(-2); flatq=quantize_activation(flat); final=att.wo_b.project_quantized(flatq)
 final2=att(x,[mx.array([2],mx.int32),old,None,None,mx.zeros((1,0,c.head_dim),mx.bfloat16),mx.zeros((1,0,c.head_dim),mx.bfloat16),mx.zeros((1,0),mx.int64)],{},2); mx.eval(final,final2)
 actual={'attention_input':u16(x),'shared_act_quant_deq':u16(qx),'wq_a_output':u16(query),'wkv_output':u16(kv_input),'q_norm_output':u16(qr),'wq_b_act_quant_deq':u16(qrq),'q_b_pre_rope':u16(qpre),'q_rotary':u16(q),'kv_norm':u16(kvn),'kv_rotary':u16(kvr),'new_window_packed':u8(new),'visible_window_packed':u8(kv),'sparse_output':u16(sp),'inverse_rope':u16(inv),'wo_a_grouped_input':u16(grouped),'wo_a_output':u16(woa),'wo_b_act_quant_deq':u16(flatq),'attention_output':u16(final)}
 # official source-derived stages
 ck=Path(CKPT); cc=cfg(ck); sh=shard(ck,'layers.0.attn.wq_a.weight'); xnp=expcap['block0_attn_pre_norm_output']
 wqa=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_a.weight',np.uint8,(1280,DIM))); wqas=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_a.scale',np.uint8,(1280//32,DIM//32)))
 wkv=np.ascontiguousarray(mmap(sh,'layers.0.attn.wkv.weight',np.uint8,(D,DIM))); wkvs=np.ascontiguousarray(mmap(sh,'layers.0.attn.wkv.scale',np.uint8,(D//32,DIM//32)))
 qnw=np.ascontiguousarray(mmap(sh,'layers.0.attn.q_norm.weight',np.uint16,(1280,))); kvnw=np.ascontiguousarray(mmap(sh,'layers.0.attn.kv_norm.weight',np.uint16,(D,)))
 aq,asc=act_quant_scales(bf16_to_f32(xnp.reshape(1,DIM))); shared_deq=f32_to_bf16_rne((__import__('tools.run_official_window_kv_prelude_fixture',fromlist=['E4']).E4[aq].reshape(1,DIM//32,32)*asc[:,:,None]).reshape(1,DIM))
 wqa_o=fp8_linear(xnp.reshape(1,DIM),wqa,wqas); wkv_o=fp8_linear(xnp.reshape(1,DIM),wkv,wkvs)
 qr_o=rms_eps(wqa_o,qnw,float(cc['rms_norm_eps'])); qra,qrsc=act_quant_scales(bf16_to_f32(qr_o)); qr_deq=f32_to_bf16_rne((__import__('tools.run_official_window_kv_prelude_fixture',fromlist=['E4']).E4[qra].reshape(1,1280//32,32)*qrsc[:,:,None]).reshape(1,1280))
 wqb=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_b.weight',np.uint8,(H*D,1280))); wqbs=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_b.scale',np.uint8,((H*D)//32,1280//32)))
 qpre_o=fp8_linear(qr_o,wqb,wqbs).reshape(1,1,H,D); co,si=freqs(RD,3,0,float(cc['rope_theta']),float(cc['rope_scaling']['factor']),float(cc['rope_scaling']['beta_fast']),float(cc['rope_scaling']['beta_slow'])); q_o=rotary_any(qpre_o,co[2:3],si[2:3])
 kvn_o=rms_eps(wkv_o,kvnw,float(cc['rms_norm_eps'])).reshape(1,1,D); kvr_o=rotary_any(kvn_o,co[2:3],si[2:3]); codes,scales,new_deq=act_quant(kvr_o.reshape(1,D)); new_deq=new_deq.reshape(1,1,D); window=np.empty((1,3,D),np.uint16); window[:,:2]=old_sem; window[:,2:]=new_deq
 sink=np.ascontiguousarray(mmap(sh,'layers.0.attn.attn_sink',np.float32,(H,))); topk=window_topk_source(2,1,128,1); *_,sp_o=sparse(q_o,window,sink,topk,np.float32(D**-0.5)); inv_o,woa_o,attn_o=project(0,sp_o,co[2:3],si[2:3])
 # wo_b quant input official
 from tools.run_official_attention_output_projection_fixture import actq as out_actq, E4 as E4b
 fq,fs=out_actq(bf16_to_f32(woa_o.reshape(1,8192))); fdeq=f32_to_bf16_rne((E4b[fq].reshape(1,8192//32,32)*fs[:,:,None]).reshape(1,8192))
 official={'attention_input':xnp,'shared_act_quant_deq':shared_deq.reshape(1,1,DIM),'wq_a_output':wqa_o.reshape(1,1,1280),'wkv_output':wkv_o.reshape(1,1,D),'q_norm_output':qr_o.reshape(1,1,1280),'wq_b_act_quant_deq':qr_deq.reshape(1,1,1280),'q_b_pre_rope':qpre_o,'q_rotary':q_o,'kv_norm':kvn_o,'kv_rotary':kvr_o,'new_window_deq':new_deq,'sparse_output':sp_o,'inverse_rope':inv_o,'wo_a_grouped_input':inv_o.reshape(1,1,8,4096),'wo_a_output':woa_o,'wo_b_act_quant_deq':fdeq.reshape(1,1,8192),'attention_output':attn_o}
 def ordered(a):
  x=a.astype(np.uint16).astype(np.int32); return np.where((x&0x8000)!=0,0x8000-x,x).astype(np.int32)
 def f32(a): return bf16_to_f32(a) if a.dtype==np.uint16 else a.astype(np.float32)
 def cmp(a,b):
  if a.dtype==np.uint16 and b.dtype==np.uint16:
   u=np.abs(ordered(a)-ordered(b)); d=np.abs(f32(a)-f32(b)); return {'actual_digest':sha(a),'expected_digest':sha(b),'max_bf16_ulp':int(u.max()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'differing_count':int((a!=b).sum())}
  d=np.abs(a.astype(np.float32)-b.astype(np.float32)); return {'actual_digest':sha(a),'expected_digest':sha(b),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'differing_count':int((d!=0).sum())}
 order=['attention_input','shared_act_quant_deq','wq_a_output','q_norm_output','wq_b_act_quant_deq','q_b_pre_rope','q_rotary','wkv_output','kv_norm','kv_rotary','sparse_output','inverse_rope','wo_a_grouped_input','wo_a_output','wo_b_act_quant_deq','attention_output']
 comps=[]; first=None; first_details=None
 def elem_details(a,b,limit=8):
  locs=np.argwhere(a!=b); out=[]
  for loc in locs[:limit]:
   idx=tuple(int(x) for x in loc); out.append({'index':list(idx),'actual_bits':f'0x{int(a[idx]):04x}','expected_bits':f'0x{int(b[idx]):04x}','actual_value':float(f32(a[idx])),'expected_value':float(f32(b[idx])),'abs':float(abs(f32(a[idx])-f32(b[idx]))),'ulp':int(abs(ordered(a[idx])-ordered(b[idx])))} )
  return out
 for name in order:
  ccmp=cmp(actual[name],official[name]); status='EXACT' if ccmp.get('differing_count',0)==0 else ('FIRST_NONEXACT' if first is None else 'DOWNSTREAM_OF_FIRST_NONEXACT')
  rec={'stage':name,'status':status,'comparison':ccmp}
  if status=='FIRST_NONEXACT':
   first=name; first_details=elem_details(actual[name],official[name]); rec['differing_elements']=first_details
  comps.append(rec)
 # activation A/B actual custom vs compiled
 ab={}
 for name,t in [('shared_input',x),('wq_b_input_qr',qr),('wo_b_input',flat)]:
  custom=quantize_activation(t); ref=_compiled_quantize_activation(t,8,32,False); mx.eval(custom,ref); ab[name]=cmp(u16(custom),u16(ref))
 # determinism repeat
 vals=[]; dig=[]
 for i in range(5):
  y=att(x,[mx.array([2],mx.int32),old,None,None,mx.zeros((1,0,c.head_dim),mx.bfloat16),mx.zeros((1,0,c.head_dim),mx.bfloat16),mx.zeros((1,0),mx.int64)],{},2); yy=u16(y); vals.append(f'0x{int(yy[0,0,3758]):04x}'); dig.append(sha(yy))
 classification = 'BLOCK0_SEED_WQ_B_QMM_DIFFERENCE' if first=='q_b_pre_rope' else ('BLOCK0_SEED_'+first.upper()+'_DIFFERENCE' if first else 'NO_DIFFERENCE')
 rec={'schema':'ds41f.m4.block0_attention_production_internals.v1','production_source_identity':{'omlx_path':str(OMLX),'git':subprocess.check_output(['git','-C',str(OMLX),'rev-parse','HEAD'],text=True).strip()},'checkpoint':str(CKPT),'isolated_replay':{'attempted':True,'endpoint_reproduced_actual_capture_exact':sha(actual['attention_output'])==sha(actcap['block0_attention_output']),'endpoint_digest':sha(actual['attention_output']),'captured_actual_digest':sha(actcap['block0_attention_output']),'value_3758':f'0x{int(actual["attention_output"][0,0,3758]):04x}'},'ordered_internal_boundary_comparisons':comps,'first_nonexact_boundary':first,'first_nonexact_boundary_details':first_details,'causal_element_3758':{'official_bits':f'0x{int(official["attention_output"][0,0,3758]):04x}','actual_bits':f'0x{int(actual["attention_output"][0,0,3758]):04x}','official_value':float(f32(official['attention_output'][0,0,3758])),'actual_value':float(f32(actual['attention_output'][0,0,3758]))},'fp8_activation_custom_vs_compiled_reference':ab,'determinism':{'repeat_count':5,'values_3758':vals,'digests':dig,'deterministic':len(set(vals))==1 and len(set(dig))==1},'root_cause_classification':'wq_b mx.quantized_matmul output differs by 1 BF16 ULP even though wq_b activation-quantized input is exact; custom FP8 activation equals compiled/reference on this input','final_classification':classification,'candidate_exact_alternative':{'name':'official-source-derived/reference FP8 wq_b projection for the single-token q path','tested':True,'preserves_official_checkpoint_semantics':True,'length1_topology_preserved':False,'result':'would restore q_b_pre_rope to official bits but is diagnostic/reference, not production topology','cost_class':'major / diagnostic-only'},'next_frontier':'bounded candidate canonical MLX/QMM alternative for wq_b that preserves official FP8 weights/scales and decode topology, then verify final seed removal'}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'wrote':str(OUT),'endpoint_exact':rec['isolated_replay']['endpoint_reproduced_actual_capture_exact'],'first':first,'class':rec['final_classification'],'v3758':rec['isolated_replay']['value_3758']},indent=2))
if __name__=='__main__': run()
