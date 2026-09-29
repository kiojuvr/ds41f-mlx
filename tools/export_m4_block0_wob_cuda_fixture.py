#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
OUTDIR=ROOT/'artifacts/m4/block0-wob-official-cuda-oracle'
FIXTURE=OUTDIR/'block0_wob_cuda_fixture.npz'
RESULT=OUTDIR/'result.json'

def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha(a) -> str: return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def bf16_to_f32(x): return (np.asarray(x,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def ordered(a):
    x=np.asarray(a,dtype=np.uint16).astype(np.int32); return np.where((x&0x8000)!=0,0x8000-x,x).astype(np.int32)
def cmp(a,b):
    u=np.abs(ordered(a)-ordered(b)); d=np.abs(bf16_to_f32(a)-bf16_to_f32(b))
    return {'actual_digest':sha(a),'expected_digest':sha(b),'max_bf16_ulp':int(u.max()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'differing_count':int((a!=b).sum()),'count_gt_1_ulp':int((u>1).sum())}
def elem(a,b=None,i=3758):
    rec={'index':[0,0,i],'bits':f'0x{int(a.reshape(1,1,-1)[0,0,i]):04x}','value':float(bf16_to_f32(a.reshape(1,1,-1)[0,0,i]))}
    if b is not None:
        rec.update({'expected_bits':f'0x{int(b.reshape(1,1,-1)[0,0,i]):04x}','expected_value':float(bf16_to_f32(b.reshape(1,1,-1)[0,0,i])),'ulp':int(abs(ordered(a.reshape(1,1,-1)[0,0,i])-ordered(b.reshape(1,1,-1)[0,0,i])))} )
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=str(CKPT)); ap.add_argument('--outdir',default=str(OUTDIR)); args=ap.parse_args()
    ck=Path(args.checkpoint); outdir=Path(args.outdir); fixture=outdir/'block0_wob_cuda_fixture.npz'; result=outdir/'result.json'
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import Attention, rope
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection, pack_activation, quantize_activation
    from omlx.patches.deepseek_v41.convert import repack_weight
    from omlx.patches.deepseek_v41.storage import TensorFile
    from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
    from tools.native_decode_session_state import build_prefill_state
    from tools.run_native_layer0_25_transformer_entry_validation import mmap, shard, H, D
    from tools.run_official_attention_output_projection_fixture import deq_weight_bf16, grouped_woa
    from tools.run_official_window_kv_prelude_fixture import act_quant, e8_arr, E4

    actcap=np.load(ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz')
    expcap=np.load(ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz')
    reference_row=np.ascontiguousarray(expcap['block0_attention_output'],dtype=np.uint16)

    c=ModelConfig.from_dict(json.load(open(ck/'config.json'))); att=Attention(c,0)
    idx=json.load(open(ck/'model.safetensors.index.json'))['weight_map']; files={}
    def read(k):
        f=idx[k]
        if f not in files: files[f]=TensorFile(ck/f)
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

    x=mx.array(actcap['block0_attention_input']).view(mx.bfloat16); qx=quantize_activation(x)
    query=att.wq_a.project_quantized(qx); qr=att.q_norm(query); qrq=quantize_activation(qr)
    prefill,_=build_prefill_state(); old_sem=np.ascontiguousarray(prefill.visible_value_arrays['window_kv.0.visible'][:,:2,:]); old=pack_activation(mx.array(old_sem).view(mx.bfloat16))
    kv_input=att.wkv.project_quantized(qx); qpre=att.wq_b.project_quantized(qrq.reshape(1,1280)).reshape(1,1,c.n_heads,c.head_dim); q=rope(qpre,mx.arange(2,3),c,False)
    kvn=att.kv_norm(kv_input); kvr=rope(kvn,mx.arange(2,3),c,False); new=pack_activation(kvr); kv=mx.concatenate([old[:,:2],new],1)
    local=mx.arange(2,3)[:,None]-c.window_size+1+mx.arange(c.window_size); valid=(local>=0)&(local<=mx.arange(2,3)[:,None]); wi=mx.where(valid,local,-1)[None]
    sp=packed_sparse_attention(q,kv,mx.zeros((1,0,c.head_dim//2+c.head_dim//16),mx.uint8),wi,mx.zeros((1,1,0),mx.int32),att.attn_sink,c.head_dim**-0.5)
    inv=rope(sp,mx.arange(2,3),c,False,inverse=True)
    grouped=inv.reshape(1,1,c.o_groups,-1); weight=att.wo_a.weight.reshape(c.o_groups,c.o_lora_rank,-1); woa=mx.einsum('bsgd,grd->bsgr',grouped,weight)
    prequant_prod=np.ascontiguousarray(u16(woa).reshape(1,8192),dtype=np.uint16)
    flatq=quantize_activation(woa.flatten(-2)); flatq_deq=np.ascontiguousarray(u16(flatq).reshape(1,8192),dtype=np.uint16)
    mlx_row=np.ascontiguousarray(u16(att.wo_b.project_quantized(flatq)).reshape(1,1,5120),dtype=np.uint16)

    sh=shard(ck,'layers.0.attn.wo_a.weight')
    woa_w=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_a.weight',np.uint8,(8192,4096)))
    woa_s=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_a.scale',np.uint8,(8192//32,4096//32)))
    woa_b=deq_weight_bf16(woa_w,woa_s)
    official_woa,_=grouped_woa(u16(inv).reshape(1,1,H,D),woa_b)
    prequant_ref=np.ascontiguousarray(official_woa.reshape(1,8192),dtype=np.uint16)

    act_codes_prod, act_scales_prod, act_deq_prod=act_quant(prequant_prod)
    act_codes_ref, act_scales_ref, act_deq_ref=act_quant(prequant_ref)
    if sha(act_deq_prod)!=sha(flatq_deq): raise SystemExit('production act_quant deq digest mismatch')
    if not np.array_equal(act_codes_prod,act_codes_ref) or not np.array_equal(act_scales_prod,act_scales_ref):
        raise SystemExit('production/reference prequant inputs do not quantize to identical FP8 activation')

    wob=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_b.weight',np.uint8,(5120,8192)))
    wobs=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_b.scale',np.uint8,(5120//32,8192//32)))
    scale_f32=e8_arr(act_scales_prod)
    act_deq_from_codes=(E4[act_codes_prod].reshape(1,8192//32,32)*scale_f32[:,:,None]).reshape(1,8192)
    assert sha(np.asarray(act_deq_prod,dtype=np.uint16))==sha(flatq_deq)

    outdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(fixture,
        prequant_bf16_u16=prequant_prod,
        prequant_reference_bf16_u16=prequant_ref,
        act_fp8_u8=np.ascontiguousarray(act_codes_prod,dtype=np.uint8),
        act_scale_e8m0_u8=np.ascontiguousarray(act_scales_prod,dtype=np.uint8),
        act_deq_bf16_u16=np.ascontiguousarray(act_deq_prod,dtype=np.uint16),
        wob_weight_f8e4m3_u8=wob,
        wob_scale_e8m0_u8=wobs,
        reference_output_bf16_u16=reference_row,
        mlx_m1_output_bf16_u16=mlx_row)

    model_py=ck/'inference/model.py'; kernel_py=ck/'inference/kernel.py'
    tensors={}
    with np.load(fixture) as z:
        for k in z.files:
            tensors[k]={'shape':list(z[k].shape),'dtype':str(z[k].dtype),'sha256':sha(z[k])}
    rec={
      'schema':'ds41f.m4.block0_wob_official_cuda_oracle.v1',
      'status':'PORTABLE_FIXTURE_EXPORTED_CUDA_ORACLE_NOT_RUN_LOCALLY',
      'checkpoint':str(ck),
      'official_source_identities':{'model_py_sha256':sha_bytes(model_py.read_bytes()),'kernel_py_sha256':sha_bytes(kernel_py.read_bytes()),'model_py_path':str(model_py),'kernel_py_path':str(kernel_py)},
      'official_linear_semantics':'x,s = act_quant(x, fp8_block_size, scale_fmt, scale_dtype); return fp8_gemm(x, s, weight, weight.scale, scale_dtype, block_size=fp8_block_size)',
      'official_fp8_gemm_structure':{'block_M':32,'block_N':128,'block_K':32,'accum_dtype':'FP32','out_dtype':'BF16','per_K_block':'T.gemm(A_shared, B_shared, C_local, transpose_B=True); C_local_accum += C_local * activation_scale * weight_scale'},
      'fixture':{'path':str(fixture),'npz_sha256':sha_bytes(fixture.read_bytes()),'tensors':tensors,'portable_size_bytes':fixture.stat().st_size},
      'local_environment':{'cuda_available':False,'reason':'current harness is Apple/MLX host; CUDA/TileLang oracle must be run on the RTX 4090 host using tools/run_m4_block0_wob_cuda_oracle.py'},
      'fixture_verification':{'wo_b_act_quant_deq_digest':sha(flatq_deq),'expected_digest_prefix':'60c026','production_reference_prequant_codes_equal':bool(np.array_equal(act_codes_prod,act_codes_ref)),'production_reference_prequant_scales_equal':bool(np.array_equal(act_scales_prod,act_scales_ref)),'mlx_vs_reference':cmp(mlx_row,reference_row),'mlx_3758':elem(mlx_row),'reference_3758':elem(reference_row)},
      'cuda_oracle':{'executed':False,'official_kernel_available':None,'gpu_environment':None,'oracle_A':None,'oracle_B':None,'comparisons':None},
      'final_classification':'CUDA_ORACLE_PENDING_EXTERNAL_EXECUTION',
      'precision_policy_consequence':'NO_DECISION: do not promote HIGH_SENSITIVITY_EXACT_BOUNDARY and do not implement a custom Metal kernel until official CUDA/TileLang result is recorded.',
      'next_frontier':'Copy the fixture NPZ plus tools/run_m4_block0_wob_cuda_oracle.py to the CUDA host and execute the official TileLang act_quant/fp8_gemm oracle.',
      'ok':False}
    result.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'fixture':str(fixture),'fixture_bytes':fixture.stat().st_size,'result':str(result),'classification':rec['final_classification']},indent=2))
if __name__=='__main__': main()
