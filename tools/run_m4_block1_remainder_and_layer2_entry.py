#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.official_model_math import OfficialModelMath
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT
from tools.run_official_hyper_connections_fixture import mmap, shard, digest, bf16_to_f32, hc_mixes, hc_pre, hc_post
from tools.run_native_layer0_25_transformer_entry_validation import VOCAB,DIM,HC,attn,rms_model,block as native_block
from tools.run_native_layer24_25_connected_validation import moe_layer


def sha(a:np.ndarray)->str: return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def ordered_bf16(a:np.ndarray)->np.ndarray:
    x=a.astype(np.uint16).astype(np.int32); return np.where((x&0x8000)!=0,0x8000-x,x).astype(np.int32)
def ulp(a:np.ndarray,b:np.ndarray)->np.ndarray: return np.abs(ordered_bf16(a)-ordered_bf16(b))
def f32(a:np.ndarray)->np.ndarray: return bf16_to_f32(a) if a.dtype==np.uint16 else a.astype(np.float32)
def info(a:np.ndarray, dt:str|None=None)->dict[str,Any]: return {'shape':list(a.shape),'dtype':dt or str(a.dtype),'sha256':sha(a)}
def cmp_bf16(a:np.ndarray,b:np.ndarray,tol:int=1)->dict[str,Any]:
    u=ulp(a,b); d=np.abs(f32(a)-f32(b)); return {**info(a,'BF16(uint16)'),'reference_sha256':sha(a),'actual_sha256':sha(b),'sha_equal':sha(a)==sha(b),'shape_pair':[list(a.shape),list(b.shape)],'max_bf16_ulp':int(u.max()) if u.size else 0,'ulp_counts':{'ulp0':int(np.count_nonzero(u==0)),'ulp1':int(np.count_nonzero(u==1)),'ulp_gt1':int(np.count_nonzero(u>1))},'max_abs_diff':float(d.max()) if d.size else 0.0,'mean_abs_diff':float(d.mean()) if d.size else 0.0,'mismatched_elements':int(np.count_nonzero(a!=b)),'max_bf16_ulp_lte':tol,'within_contract':(int(u.max()) if u.size else 0)<=tol}
def cmp_f32(a:np.ndarray,b:np.ndarray,tol:float)->dict[str,Any]:
    d=np.abs(a.astype(np.float32)-b.astype(np.float32)); return {**info(a,'FP32'),'reference_sha256':sha(a),'actual_sha256':sha(b),'sha_equal':sha(a)==sha(b),'shape_pair':[list(a.shape),list(b.shape)],'max_abs_diff':float(d.max()) if d.size else 0.0,'mean_abs_diff':float(d.mean()) if d.size else 0.0,'mismatched_elements':int(np.count_nonzero(a!=b)),'max_abs_lte':tol,'within_contract':(float(d.max()) if d.size else 0.0)<=tol}
def cmp_i(a:np.ndarray,b:np.ndarray)->dict[str,Any]: return {'shape_pair':[list(a.shape),list(b.shape)],'reference_sha256':sha(a),'actual_sha256':sha(b),'exact':bool(np.array_equal(a,b)),'mismatched_elements':int(np.count_nonzero(a!=b))}

def load_manifest(trace:Path)->dict[str,list[dict[str,Any]]]:
    out:dict[str,list[dict[str,Any]]]={}
    for l in open(trace/'manifest.jsonl'):
        e=json.loads(l); out.setdefault(e['boundary'],[]).append(e)
    return out
def arr(ent:dict[str,list[dict[str,Any]]], k:str)->np.ndarray:
    entries=ent[k]; vals=[]
    for e in entries:
        dt={'bfloat16':np.uint16,'float32':np.float32,'uint32':np.uint32,'int32':np.int32,'uint16':np.uint16,'uint8':np.uint8}[e['dtype']]
        vals.append(np.fromfile(e['path'],dt).reshape(e['shape']))
    if len(vals)==1: return vals[0]
    return np.concatenate(vals,axis=0)
def has(ent,k): return k in ent

def run_hist(hist:Path, ck:Path)->tuple[Path,dict[str,Any]]:
    td=tempfile.mkdtemp(prefix='ds41f-hist-block1-'); tok=Path(td)/'tokens.txt'; trace=Path(td)/'trace'; tok.write_text('0 3\n')
    env=os.environ.copy(); env.update({'TOKENS_FILE':str(tok),'CHECKPOINT':str(ck),'DSV41_SEMANTIC_TRACE_DIR':str(trace),'DSV41_RUNTIME_PACKED_EXPERT_BANK':'0','DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS':'0','DSV41_RUNTIME_GROUP_SELECTED_EXPERTS':'0'})
    t=time.perf_counter(); p=subprocess.run(['bash','tools/benchmark/run_text_backbone_reference.sh'],cwd=hist,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
    if p.returncode: raise RuntimeError(p.stdout[-4000:])
    return trace,{'repository':str(hist),'trace_dir_untracked':str(trace),'elapsed_s':time.perf_counter()-t,'stdout_tail':p.stdout[-2000:]}

def layer1_exact_ops(ck:Path,c:dict[str,Any], r:dict[str,np.ndarray])->dict[str,np.ndarray|dict[str,Any]]:
    # exact post-attention HC
    post_attn=hc_post(r['attn_out'].reshape(1,2,DIM),r['engram1_hidden'].reshape(1,2,HC,DIM),r['attn_post'].reshape(1,2,HC),r['attn_comb'].reshape(1,2,HC,HC))
    fsh=shard(ck,'layers.1.hc_ffn_fn'); fn=np.ascontiguousarray(mmap(fsh,'layers.1.hc_ffn_fn',np.float32,(24,20480))); base=np.ascontiguousarray(mmap(fsh,'layers.1.hc_ffn_base',np.float32,(24,))); scale=np.ascontiguousarray(mmap(fsh,'layers.1.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(r['post_attn'].reshape(1,2,HC,DIM),fn,scale,base,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    ffn_hc_pre=hc_pre(r['post_attn'].reshape(1,2,HC,DIM),r['attn_pre'].reshape(1,2,HC))
    nw=np.ascontiguousarray(mmap(shard(ck,'layers.1.ffn_norm.weight'),'layers.1.ffn_norm.weight',np.uint16,(DIM,)))
    ffn_in=rms_model(ffn_hc_pre.reshape(2,DIM),nw,c).reshape(1,2,DIM)
    moe=moe_layer(ck,c,1,r['ffn_in'].reshape(1,2,DIM))
    hidden=hc_post(r['moe_out'].reshape(1,2,DIM),r['post_attn'].reshape(1,2,HC,DIM),r['ffn_post'].reshape(1,2,HC),r['ffn_comb'].reshape(1,2,HC,HC))
    return {'post_attn_from_exact':post_attn,'ffn_pre':ffn_pre,'ffn_post':ffn_post,'ffn_comb':ffn_comb,'ffn_hc_pre':ffn_hc_pre,'ffn_in':ffn_in,'moe':moe,'hidden_from_exact':hidden}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--historical',default='/Volumes/SDXC-512/deepseek-v41-flash-mlx'); ap.add_argument('--trace-dir'); ap.add_argument('--out',default='artifacts/m4/block1-remainder-layer2-entry/result.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); hist=Path(a.historical); trace,hrec=(Path(a.trace_dir),{'trace_dir_untracked':a.trace_dir}) if a.trace_dir else run_hist(hist,ck)
    ent=load_manifest(trace); math=OfficialModelMath(ck); c=math.config
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    tokens=np.array([[0,3]],np.int64); x=np.repeat(emb[tokens].copy()[:,:,None,:],HC,axis=2); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; o0=math.execute_block(0,x,pre,shared); hashes=math.engram_hashes_for_tokens(tokens); post1,_=math.apply_engram(1,o0['x_out'],hashes['layer1_hash']); o1=math.execute_block(1,post1,o0['ffn_pre'],shared)
    r={
      'engram1_hidden':arr(ent,'encoder.engram1.hidden'),'attn_pre':arr(ent,'encoder.layer1.attn_pre'),'attn_post':arr(ent,'encoder.layer1.attn_post'),'attn_comb':arr(ent,'encoder.layer1.attn_comb'),'attn_out':arr(ent,'encoder.layer1.attn_out'),'post_attn':arr(ent,'encoder.layer1.post_attn'),'ffn_pre':arr(ent,'encoder.layer1.ffn_pre'),'ffn_post':arr(ent,'encoder.layer1.ffn_post'),'ffn_comb':arr(ent,'encoder.layer1.ffn_comb'),'ffn_hc_pre':arr(ent,'encoder.layer1.ffn_hc_pre'),'ffn_in':arr(ent,'encoder.layer1.ffn_in'),'moe_out':arr(ent,'encoder.layer1.moe_out'),'hidden':arr(ent,'encoder.layer1.hidden'),'pre_mix':arr(ent,'encoder.layer1.pre_mix')}
    native_map={'post_attn':o1['x_after_attn'].reshape(2,HC,DIM),'ffn_in':o1['moe_input'].reshape(2,DIM),'moe_out':o1['full_moe_output'].reshape(2,DIM),'hidden':o1['x_out'].reshape(2,HC,DIM),'pre_mix':o1['ffn_pre'].reshape(2,HC)}
    native_prop={}
    for k,dv in native_map.items():
        if r[k].dtype==np.uint16: native_prop[k]=cmp_bf16(r[k],dv,1)
        else: native_prop[k]=cmp_f32(r[k],dv,1e-3)
    ex=layer1_exact_ops(ck,c,r)
    exact={
      'post_attention_hc':cmp_bf16(r['post_attn'],ex['post_attn_from_exact'].reshape(2,HC,DIM),1),
      'ffn_pre':cmp_f32(r['ffn_pre'],ex['ffn_pre'].reshape(2,HC),1e-3),
      'ffn_post':cmp_f32(r['ffn_post'],ex['ffn_post'].reshape(2,HC),1e-3),
      'ffn_comb':cmp_f32(r['ffn_comb'],ex['ffn_comb'].reshape(2,HC,HC),1e-3),
      'ffn_hc_pre':cmp_bf16(r['ffn_hc_pre'],ex['ffn_hc_pre'].reshape(2,DIM),1),
      'ffn_in':cmp_bf16(r['ffn_in'],ex['ffn_in'].reshape(2,DIM),1),
      'final_hc_post_hidden':cmp_bf16(r['hidden'],ex['hidden_from_exact'].reshape(2,HC,DIM),1),
      'returned_pre_mix':cmp_f32(r['pre_mix'],ex['ffn_pre'].reshape(2,HC),1e-3),
    }
    moe=ex['moe']; moe_cmp={
      'route_ids_token0':cmp_i(arr(ent,'moe.layer1.token0.route_ids').astype(np.int64),moe['idx'][0].astype(np.int64)),
      'route_ids_token1':cmp_i(arr(ent,'moe.layer1.token1.route_ids').astype(np.int64),moe['idx'][1].astype(np.int64)),
      'route_weights_token0':cmp_f32(arr(ent,'moe.layer1.token0.route_weights'),moe['weights'][0],1e-3),
      'route_weights_token1':cmp_f32(arr(ent,'moe.layer1.token1.route_weights'),moe['weights'][1],1e-3),
      'routed_sum_token0':cmp_f32(arr(ent,'moe.layer1.token0.routed_accumulated_f32'),moe['routed'][0:1],1e-3),
      'routed_sum_token1':cmp_f32(arr(ent,'moe.layer1.token1.routed_accumulated_f32'),moe['routed'][1:2],1e-3),
      'shared_token0':cmp_f32(arr(ent,'moe.layer1.token0.shared_f32'),bf16_to_f32(moe['shared'][0:1]),1e-3),
      'shared_token1':cmp_f32(arr(ent,'moe.layer1.token1.shared_f32'),bf16_to_f32(moe['shared'][1:2]),1e-3),
      'final_moe_out':cmp_bf16(r['moe_out'],moe['final'].reshape(2,DIM),1),
    }
    exact['moe']={'within_contract':all(v.get('within_contract',v.get('exact',False)) for v in moe_cmp.values()),'comparisons':moe_cmp}
    block1_ok=all(v.get('within_contract',False) for k,v in exact.items() if k!='moe') and exact['moe']['within_contract']
    layer2={}
    if block1_ok:
        # Enter layer2: native entry/profile and exact historical entry diagnostic.
        native_entry={'hidden_input':info(o1['x_out'],'BF16(uint16)'),'incoming_pre_mix':info(o1['ffn_pre'],'FP32')}
        r2_hidden=arr(ent,'encoder.layer1.hidden').reshape(1,2,HC,DIM); r2_pre=arr(ent,'encoder.layer1.pre_mix').reshape(1,2,HC)
        shared2={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; d2_on_r=native_block(ck,c,2,r2_hidden,r2_pre,shared2)
        layer2={'reached':True,'native_entry':native_entry,'exact_entry_comparisons':{'attention_input':cmp_bf16(arr(ent,'encoder.layer2.attn_in'),d2_on_r['attention_input'].reshape(2,DIM),1),'attention_output':cmp_bf16(arr(ent,'encoder.layer2.attn_out'),d2_on_r['attention_output'].reshape(2,DIM),1),'compress_kv_publication_digest':digest(shared2['compress_kv']) if shared2['compress_kv'] is not None else None,'index_k_publication_digest':digest(shared2['index_k']) if shared2['index_k'] is not None else None,'topk_publication_digest':digest(shared2['topk_idxs']) if shared2['topk_idxs'] is not None else None},'classification':'INCOMPLETE'}
        e=layer2['exact_entry_comparisons']
        if e['attention_input']['within_contract'] and e['attention_output']['within_contract'] and e['compress_kv_publication_digest'] and e['index_k_publication_digest'] and e['topk_publication_digest']:
            layer2['classification']='QUALIFIED_COARSE_ENTRY_NO_HISTORICAL_PUBLICATION_COMPARISON'
        elif e['attention_input']['within_contract'] and not e['attention_output']['within_contract']:
            layer2['classification']='LOCAL SEMANTIC BUG AT COARSE ATTENTION OUTPUT'
    remaining=None if block1_ok else next((k for k,v in exact.items() if not v.get('within_contract',False)),None)
    if block1_ok and layer2 and layer2.get('classification','').startswith('LOCAL SEMANTIC BUG'):
        remaining='encoder.layer2.attn_out under exact historical Block2 entry'
    rec={'schema':'ds41f.m4.block1-remainder-layer2-entry.v1','checkpoint':str(ck),'historical_export':hrec,'qualification_only':True,'production_path_changed':False,'native_block1_propagation':native_prop,'exact_boundary_substitution':exact,'block1_classification':'COMPLETE' if block1_ok else 'INCOMPLETE','layer2':layer2 or {'reached':False},'first_remaining_genuinely_local_semantic_divergence':remaining,'ok':block1_ok}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print(rec['block1_classification'], rec['layer2'].get('classification')); return 0 if block1_ok else 2
if __name__=='__main__': raise SystemExit(main())
