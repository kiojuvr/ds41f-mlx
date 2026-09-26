#!/usr/bin/env python3
"""Run layer-0 window-only Attention as one native primitive dataflow and gate it against Boundary 1-4 artifacts."""
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE, DS4_AUTHORITY_SHA, compile_native_prefill_library, load_native_prefill_library  # noqa
from tools.run_official_attention_output_projection_fixture import inv_rot, deq_weight_bf16, f32_to_bf16_rne, digest, BLOCK, WOAOUT, WOAIN, DIM, GROUPS, O_RANK  # noqa
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
B1='artifacts/attention-q-prelude-official-reference-fixture.json'; B2='artifacts/window-kv-prelude-official-reference-fixture.json'; B3='artifacts/sparse-attn-official-reference-fixture.json'; B4='artifacts/attention-output-projection-official-reference-fixture.json'
VOCAB=129280; Q_RANK=1280; H=64; HEAD_DIM=512; RD=64; KV_DIM=512; WINDOW=128

def header(p):
    with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape): h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)
def cmp_digest(name,a,e,tol=0):
    a=np.ascontiguousarray(a); e=np.ascontiguousarray(e); d=np.abs(a.astype(np.int64)-e.astype(np.int64)); m=int(d.max()) if d.size else 0
    return {'name':name,'shape_matches':list(a.shape)==list(e.shape),'bit_exact':digest(a)==digest(e),'max_error':m,'max_error_lte':int(tol),'within_tolerance':m<=int(tol),'native_sha256':digest(a),'reference_sha256':digest(e)}
def topk(seqlen):
    end=np.arange(seqlen,dtype=np.int32)[:,None]; idx=(end-WINDOW+1).clip(0)+np.arange(min(seqlen,WINDOW),dtype=np.int32); return np.where(idx>end,-1,idx).astype(np.int32)[None,:,:]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--b1',default=B1); ap.add_argument('--b2',default=B2); ap.add_argument('--b3',default=B3); ap.add_argument('--b4',default=B4); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-attention-window-only-integration-validation.json'); args=ap.parse_args()
    b1=json.loads(Path(args.b1).read_text()); b2=json.loads(Path(args.b2).read_text()); b3=json.loads(Path(args.b3).read_text()); b4=json.loads(Path(args.b4).read_text())
    tokens=np.array([0,3],dtype=np.int64); ck=Path(args.checkpoint); shard=ck/'model-00003-of-00048.safetensors'; emb_shard=ck/'model-00002-of-00048.safetensors'
    native=load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    x=np.ascontiguousarray(mmap(emb_shard,'embed.weight',np.uint16,(VOCAB,DIM))[tokens])
    # Q prelude.
    wqa=np.ascontiguousarray(mmap(shard,'layers.0.attn.wq_a.weight',np.uint8,(Q_RANK,DIM))); wqa_s=np.ascontiguousarray(mmap(shard,'layers.0.attn.wq_a.scale',np.uint8,(Q_RANK//BLOCK,DIM//BLOCK)))
    qnw=np.ascontiguousarray(mmap(shard,'layers.0.attn.q_norm.weight',np.uint16,(Q_RANK,)))
    wqb=np.ascontiguousarray(mmap(shard,'layers.0.attn.wq_b.weight',np.uint8,(H*HEAD_DIM,Q_RANK))); wqb_s=np.ascontiguousarray(mmap(shard,'layers.0.attn.wq_b.scale',np.uint8,((H*HEAD_DIM)//BLOCK,Q_RANK//BLOCK)))
    wqa_out,r_q1=native.official_fp8_linear_bf16(x,wqa,wqa_s,BLOCK); qn_out,r_q2=native.official_rmsnorm_bf16(wqa_out,qnw,1e-6); wqb_out,r_q3=native.official_fp8_linear_bf16(qn_out,wqb,wqb_s,BLOCK)
    q=np.ascontiguousarray(wqb_out.reshape(1,2,H,HEAD_DIM)); tail=bf16_to_f32(q[...,-RD:]); _,_,rot_tail,_,r_q4=native.official_rotary_f32(tail,{'original_seq_len':0,'base':10000.0,'factor':16.0,'beta_fast':32,'beta_slow':1}); q[...,-RD:]=f32_to_bf16_rne(rot_tail)
    # Window KV prelude.
    wkv=np.ascontiguousarray(mmap(shard,'layers.0.attn.wkv.weight',np.uint8,(KV_DIM,DIM))); wkv_s=np.ascontiguousarray(mmap(shard,'layers.0.attn.wkv.scale',np.uint8,(KV_DIM//BLOCK,DIM//BLOCK))); kvnw=np.ascontiguousarray(mmap(shard,'layers.0.attn.kv_norm.weight',np.uint16,(KV_DIM,)))
    wkv_out,r_k1=native.official_fp8_linear_bf16(x,wkv,wkv_s,BLOCK); kvn,r_k2=native.official_rmsnorm_bf16(wkv_out,kvnw,1e-6); kv=np.ascontiguousarray(kvn.reshape(1,2,KV_DIM)); ktail=bf16_to_f32(kv[...,-RD:]).reshape(1,2,1,RD); _,_,krot_tail,_,r_k3=native.official_rotary_f32(ktail,{'original_seq_len':0,'base':10000.0,'factor':16.0,'beta_fast':32,'beta_slow':1}); kv[...,-RD:]=f32_to_bf16_rne(krot_tail.reshape(1,2,RD)); _qbytes,_qsc,window_flat,r_k4=native.official_act_quant_bf16(kv.reshape(2,KV_DIM),BLOCK); window_kv=window_flat.reshape(1,2,KV_DIM); idx=topk(2)
    # Sparse attention.
    sink=np.ascontiguousarray(mmap(shard,'layers.0.attn.attn_sink',np.float32,(H,))); sparse_out,r_s=native.official_sparse_attn_bf16(q,window_kv,sink,idx,float(HEAD_DIM**-0.5))
    # Output projection.
    inv=inv_rot(sparse_out); grouped=inv.reshape(1,2,GROUPS,WOAIN)
    woa=np.ascontiguousarray(mmap(shard,'layers.0.attn.wo_a.weight',np.uint8,(WOAOUT,WOAIN))); woas=np.ascontiguousarray(mmap(shard,'layers.0.attn.wo_a.scale',np.uint8,(WOAOUT//BLOCK,WOAIN//BLOCK))); woa_bf16=deq_weight_bf16(woa,woas)
    parts=[]; woa_results=[]
    for g in range(GROUPS):
        og,r=native.official_bf16_linear_f32(np.ascontiguousarray(grouped[:,:,g,:].reshape(2,WOAIN)),np.ascontiguousarray(woa_bf16[g*O_RANK:(g+1)*O_RANK,:])); woa_results.append(r); parts.append(og.astype(np.float32))
    woa_f32=np.stack(parts,axis=1).reshape(2,GROUPS,O_RANK)[None,:,:,:]; woa_out=f32_to_bf16_rne(woa_f32); flat=woa_out.reshape(1,2,WOAOUT)
    wob=np.ascontiguousarray(mmap(shard,'layers.0.attn.wo_b.weight',np.uint8,(DIM,WOAOUT))); wobs=np.ascontiguousarray(mmap(shard,'layers.0.attn.wo_b.scale',np.uint8,(DIM//BLOCK,WOAOUT//BLOCK))); final_flat,r_ob=native.official_fp8_linear_bf16(flat.reshape(2,WOAOUT),wob,wobs,BLOCK); final=final_flat.reshape(1,2,DIM)
    comps={
      'q_rotary_output':cmp_digest('q_rotary_output',q,np.asarray(b1['expected']['rotary_applied_q_bf16_uint16'],np.uint16),1),
      'window_kv':cmp_digest('window_kv',window_kv,np.asarray(b2['expected']['window_kv_bf16_uint16'],np.uint16),0),
      'topk_idxs':cmp_digest('topk_idxs',idx,np.asarray(b2['expected']['topk_idxs_int32'],np.int32),0),
      'sparse_attn_output':cmp_digest('sparse_attn_output',sparse_out,np.asarray(b3['expected']['attention_output_bf16_uint16'],np.uint16),2),
      'inverse_rotary':cmp_digest('inverse_rotary',inv,np.asarray(b4['expected']['inverse_rotary_output_bf16_uint16'],np.uint16),1),
      'grouped_wo_a':cmp_digest('grouped_wo_a',woa_out,np.asarray(b4['expected']['grouped_wo_a_output_bf16_uint16'],np.uint16),1),
      'final_attention_output':cmp_digest('final_attention_output',final,np.asarray(b4['expected']['final_wo_b_output_bf16_uint16'],np.uint16),1)}
    all_native=[r_q1,r_q2,r_q3,r_q4,r_k1,r_k2,r_k3,r_k4,r_s]+woa_results+[r_ob]
    rec={'schema':'ds41f.native-attention-window-only-integration-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'layer-0 window-only Attention integrated native dataflow validation; stop before Block/HC','checkpoint':args.checkpoint,'authority':{'boundary_fixtures':[args.b1,args.b2,args.b3,args.b4],'official_checkpoint_raw_bits':args.checkpoint,'native_validation_provider':'single Python-orchestrated native primitive dataflow'},'scope':{'layer':0,'batch':1,'sequence':2,'tokens':[0,3],'start_pos':0,'compress_ratio':0,'world_size':1},'comparison':comps,'digests':{'final_attention_output_bf16_sha256':digest(final),'reference_final_attention_output_bf16_sha256':digest(np.asarray(b4['expected']['final_wo_b_output_bf16_uint16'],np.uint16))},'native_result':{'q_prelude':[r_q1,r_q2,r_q3,r_q4],'window_kv_prelude':[r_k1,r_k2,r_k3,r_k4],'sparse_attn':r_s,'wo_a_group_linears':woa_results,'wo_b':r_ob},'semantic_status':{'native_integrated_path_executed':True,'compressed_kv_entered':False,'block_hc_entered':False,'model_semantics_validated':False,'validated_stage':'layer-0 window-only Attention dataflow only'},'non_claims':['does not execute compressed KV, Compressor, Indexer, candidate selection','does not execute Block or HC residual integration','does not validate MoE, logits, full layer, full prefill, or full model correctness','does not benchmark performance']}
    rec['gates']={'all_boundary_intermediates_match':all(c['within_tolerance'] and c['shape_matches'] for c in comps.values()),'final_attention_output_within_tolerance':comps['final_attention_output']['within_tolerance'],'native_integrated_path_executed':True,'compressed_kv_not_entered':True,'block_hc_not_entered':True,'native_metal_executed':all(r.get('metal_enabled') is True for r in all_native),'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
    rec['ok']=all(rec['gates'].values())
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
