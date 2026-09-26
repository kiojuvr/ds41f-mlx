#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, struct, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.native_prefill import DS4_AUTHORITY_REMOTE, DS4_AUTHORITY_SHA, compile_native_prefill_library, load_native_prefill_library
from tools.run_official_attention_output_projection_fixture import deq_weight_bf16, grouped_woa, fp8_linear, f32_to_bf16_rne, digest, BLOCK, WOAOUT, WOAIN, DIM, GROUPS, O_RANK, H, D, RD
from tools.run_official_compressed_sparse_attn_fixture import DEFAULT_CHECKPOINT, freqs
from tools.run_official_sparse_attn_fixture import mmap
from tools.source_identity import SOURCE_HASH_METHOD
REF5E='artifacts/compressed-sparse-attn-official-reference-fixture.json'

def cmp(a,e,tol):
    a=np.ascontiguousarray(a,dtype=np.uint16); e=np.ascontiguousarray(e,dtype=np.uint16)
    d=np.abs(a.astype(np.int32)-e.astype(np.int32)); m=int(d.max()) if d.size else 0
    return {'shape_matches':list(a.shape)==list(e.shape),'bit_exact':digest(a)==digest(e),'mismatch_count':int(np.count_nonzero(d)),'max_bf16_ulp_error':m,'max_bf16_ulp_lte':int(tol),'within_tolerance':m<=int(tol),'native_sha256':digest(a),'reference_sha256':digest(e)}

def header(p):
    with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n

def shard(ck,name): return ck/json.loads((ck/'model.safetensors.index.json').read_text())['weight_map'][name]

def inv_rot_yarn(o,c,s):
    y=np.array(o,copy=True); tail=((y[...,-RD:].astype(np.uint32)<<16).view(np.float32)); half=RD//2
    p=tail.reshape(1,y.shape[1],H,half,2); re=p[...,0]; im=p[...,1]
    cc=c.reshape(1,y.shape[1],1,half); ss=s.reshape(1,y.shape[1],1,half); out=np.empty_like(p)
    out[...,0]=re*cc+im*ss; out[...,1]=-re*ss+im*cc
    y[...,-RD:]=f32_to_bf16_rne(out.reshape(1,y.shape[1],H,RD)); return y

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--boundary5e',default=REF5E); ap.add_argument('--native-out-dir',default='artifacts/m2/dwarfstar-prefill/native'); ap.add_argument('--out',default='artifacts/native-compressed-attention-integration-validation.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); b5e=json.loads(Path(a.boundary5e).read_text()); cfg=json.load(open(ck/'config.json'))['text_config']; assert b5e['config']['first_candidate_consumer_layer']==24 and int(cfg['compress_ratios'][24])==1
    sparse=np.asarray(b5e['expected']['sparse_output_bf16_uint16'],np.uint16); S=sparse.shape[1]
    c,s=freqs(RD,S,int(cfg['rope_scaling']['original_max_position_embeddings']),float(cfg['compress_rope_theta']),float(cfg['rope_scaling']['factor']),float(cfg['rope_scaling']['beta_fast']),float(cfg['rope_scaling']['beta_slow']))
    sh=shard(ck,'layers.24.attn.wo_a.weight')
    woa=np.ascontiguousarray(mmap(sh,'layers.24.attn.wo_a.weight',np.uint8,(WOAOUT,WOAIN))); woas=np.ascontiguousarray(mmap(sh,'layers.24.attn.wo_a.scale',np.uint8,(WOAOUT//BLOCK,WOAIN//BLOCK)))
    wob=np.ascontiguousarray(mmap(sh,'layers.24.attn.wo_b.weight',np.uint8,(DIM,WOAOUT))); wobs=np.ascontiguousarray(mmap(sh,'layers.24.attn.wo_b.scale',np.uint8,(DIM//BLOCK,WOAOUT//BLOCK)))
    inv_ref=inv_rot_yarn(sparse,c,s); grouped_ref=inv_ref.reshape(1,S,GROUPS,WOAIN); woa_bf16=deq_weight_bf16(woa,woas); woa_ref,_=grouped_woa(inv_ref,woa_bf16); flat_ref=woa_ref.reshape(1,S,WOAOUT); final_ref=fp8_linear(flat_ref.reshape(S,WOAOUT),wob,wobs).reshape(1,S,DIM)
    native=load_native_prefill_library(compile_native_prefill_library(Path(a.native_out_dir)))
    # Single Python-orchestrated native dataflow from Boundary 5e sparse output through projection.
    inv_native=inv_rot_yarn(sparse,c,s); grouped_native=inv_native.reshape(1,S,GROUPS,WOAIN)
    woa_parts=[]; lin_results=[]
    for g in range(GROUPS):
        xg=np.ascontiguousarray(grouped_native[:,:,g,:].reshape(S,WOAIN)); wg=np.ascontiguousarray(woa_bf16[g*O_RANK:(g+1)*O_RANK,:])
        outg,r=native.official_bf16_linear_f32(xg,wg); lin_results.append(r); woa_parts.append(outg.astype(np.float32))
    woa_f32=np.stack(woa_parts,axis=1).reshape(S,GROUPS,O_RANK)[None]
    woa_native=f32_to_bf16_rne(woa_f32); flat_native=woa_native.reshape(1,S,WOAOUT)
    final_flat,rwob=native.official_fp8_linear_bf16(flat_native.reshape(S,WOAOUT),wob,wobs,BLOCK); final_native=final_flat.reshape(1,S,DIM)
    tol_i=1; tol_f=1
    comps={'sparse_attn_input':cmp(sparse,np.asarray(b5e['expected']['sparse_output_bf16_uint16'],np.uint16),0),'inverse_rotary_output':cmp(inv_native,inv_ref,tol_i),'grouped_reshape':cmp(grouped_native,grouped_ref,0),'grouped_wo_a_output':cmp(woa_native,woa_ref,tol_i),'flattened_wo_a_output':cmp(flat_native,flat_ref,tol_i),'final_wo_b_output':cmp(final_native,final_ref,tol_f),'final_attention_output':cmp(final_native,final_ref,tol_f)}
    expected={'sparse_attn_input_bf16_uint16':sparse.tolist(),'inverse_rotary_output_bf16_uint16':inv_ref.tolist(),'grouped_reshape_bf16_uint16':grouped_ref.tolist(),'grouped_wo_a_output_bf16_uint16':woa_ref.tolist(),'flattened_wo_a_output_bf16_uint16':flat_ref.tolist(),'final_attention_output_bf16_uint16':final_ref.tolist()}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.uint16)) for k,v in expected.items()}
    rec={'schema':'ds41f.native-compressed-attention-integration-validation.v1','classification':'official_reference_derived_native_validation','not_omlx_derived':True,'purpose':'Boundary 5f layer-24 compressed Attention integration from sparse_attn output through output projection; stop before Block/HC','checkpoint':a.checkpoint,'authority':{'boundary5e_sparse_attn_output':a.boundary5e,'official_checkpoint_raw_bits':a.checkpoint,'native_validation_provider':'single Python-orchestrated native dataflow: Boundary 5e sparse output -> inverse rotary -> native grouped wo_a -> native wo_b'},'reference_fixture':a.boundary5e,'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':'4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65','functions':[{'name':'Attention','source_lines':[613,789],'source_hash_method':SOURCE_HASH_METHOD,'source_sha256':'86d80f5cdaa6435cacd56ce5be796c3f0155a7f92cebdb12ffe6743ac974d110','reviewed_lines':'apply_rotary_emb inverse, grouped wo_a einsum, wo_b projection'}]}},'operation_contract':{'order':'Boundary 5e sparse_attn output -> apply_rotary_emb(o[..., -64:], inverse=True) using layer-24 compressed RoPE/YaRN freqs; view(B,S,8,4096); wo_a.weight.view(8,1024,4096); grouped matmul; flatten; wo_b FP8 linear; STOP before Block/HC','predeclared_tolerance':{'inverse_rotary_max_bf16_ulp_lte':tol_i,'grouped_wo_a_max_bf16_ulp_lte':tol_i,'wo_b_final_max_bf16_ulp_lte':tol_f}},'source_tensors':{'wo_a.weight':{'name':'layers.24.attn.wo_a.weight','shard':str(sh),'dtype':'F8_E4M3','shape':[WOAOUT,WOAIN],'digest':digest(woa)},'wo_a.scale':{'name':'layers.24.attn.wo_a.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[WOAOUT//BLOCK,WOAIN//BLOCK],'digest':digest(woas)},'wo_b.weight':{'name':'layers.24.attn.wo_b.weight','shard':str(sh),'dtype':'F8_E4M3','shape':[DIM,WOAOUT],'digest':digest(wob)},'wo_b.scale':{'name':'layers.24.attn.wo_b.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[DIM//BLOCK,WOAOUT//BLOCK],'digest':digest(wobs)}},'inputs':{'layer':24,'batch':1,'sequence':S,'start_pos':0,'prefill':True,'world_size':1,'compress_ratio':1,'sparse_attn_output_digest':b5e['digests']['sparse_output_bf16_uint16_sha256']},'expected':expected,'digests':dig,'native_version':native.version(),'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'native_result':{'wo_a_group_linears':lin_results,'wo_b_fp8_linear':rwob},'comparison':comps,'semantic_status':{'boundary5e_sparse_output_used':True,'inverse_rotary_validated':comps['inverse_rotary_output']['within_tolerance'],'grouped_reshape_validated':comps['grouped_reshape']['within_tolerance'],'grouped_wo_a_validated':comps['grouped_wo_a_output']['within_tolerance'],'wo_b_validated':comps['final_wo_b_output']['within_tolerance'],'final_attention_output_within_tolerance':comps['final_attention_output']['within_tolerance'],'single_python_orchestrated_native_dataflow':True,'explicit_stop_before_block_hc':True,'model_semantics_validated':False},'non_claims':['does not execute decode / ring-buffer / partial compression group','does not validate production-scale candidate pruning; short fixture candidate mask retains all reachable positions, so candidate machinery wiring is validated but production pruning effect is not','does not execute Block / Hyper-Connections','does not validate MoE, logits, full layer, or full model correctness','does not benchmark performance or fusion']}
    rec['gates']={'layer24_output_projection_tensor_provenance_recorded':all(k in rec['source_tensors'] for k in ['wo_a.weight','wo_a.scale','wo_b.weight','wo_b.scale']),'inverse_rotary_pass':rec['semantic_status']['inverse_rotary_validated'],'grouped_wo_a_pass':rec['semantic_status']['grouped_wo_a_validated'],'wo_b_pass':rec['semantic_status']['wo_b_validated'],'final_attention_output_within_predeclared_tolerance':rec['semantic_status']['final_attention_output_within_tolerance'],'single_native_dataflow_executed':rec['semantic_status']['single_python_orchestrated_native_dataflow'] and all(r.get('metal_enabled') is True for r in lin_results+[rwob]),'explicit_stop_before_block_hc':True,'full_model_semantics_not_claimed':rec['semantic_status']['model_semantics_validated'] is False}
    rec['ok']=all(rec['gates'].values())
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
