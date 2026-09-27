#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT
from tools.run_m4_block1_remainder_and_layer2_entry import load_manifest, arr, cmp_bf16, cmp_f32, run_hist
from tools.run_native_layer0_25_transformer_entry_validation import cfg, attention_freqs, DIM, HC, D, H, RD, rms_model
from tools.run_native_layer24_25_connected_validation import moe_layer
from tools.run_official_hyper_connections_fixture import mmap, shard, bf16_to_f32, f32_to_bf16, hc_mixes, hc_pre, hc_post, MIX, HCD
from tools.run_official_attention_output_projection_fixture import deq_weight_bf16, grouped_woa, fp8_linear as fp8_linear2, WOAOUT, WOAIN, BLOCK

def layer2_from_exact_sparse(ck,c,ent):
    S=2; layer=2
    sp=arr(ent,'encoder.layer2.attn_core').reshape(1,S,H,D)
    co,si=attention_freqs(c,layer,S)
    tail=bf16_to_f32(sp[...,-RD:]); half=RD//2; p=tail.reshape(1,S,H,half,2); re=p[...,0]; im=p[...,1]
    cc=co.reshape(1,S,1,half); ss=si.reshape(1,S,1,half); o=np.empty_like(p); o[...,0]=re*cc+im*ss; o[...,1]=-re*ss+im*cc
    inv=np.array(sp,copy=True); inv[...,-RD:]=f32_to_bf16(o.reshape(1,S,H,RD))
    sh=shard(ck,f'layers.{layer}.attn.wq_a.weight')
    woa=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wo_a.weight',np.uint8,(WOAOUT,WOAIN)))
    woas=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wo_a.scale',np.uint8,(WOAOUT//BLOCK,WOAIN//BLOCK)))
    wob=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wo_b.weight',np.uint8,(DIM,WOAOUT)))
    wobs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wo_b.scale',np.uint8,(DIM//BLOCK,WOAOUT//BLOCK)))
    woa_b=deq_weight_bf16(woa,woas); woao,_=grouped_woa(inv,woa_b); final=fp8_linear2(woao.reshape(S,WOAOUT),wob,wobs).reshape(1,S,DIM)
    x=arr(ent,'encoder.layer1.hidden').reshape(1,S,HC,DIM); pre=arr(ent,'encoder.layer1.pre_mix').reshape(1,S,HC)
    hsh=shard(ck,f'layers.{layer}.hc_attn_fn')
    afn=np.ascontiguousarray(mmap(hsh,f'layers.{layer}.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(hsh,f'layers.{layer}.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(hsh,f'layers.{layer}.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    post_attn=hc_post(final,x,attn_post,attn_comb)
    fsh=shard(ck,f'layers.{layer}.hc_ffn_fn')
    ffn=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(post_attn,ffn,fscale,fbase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(post_attn,attn_pre)
    fnw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.ffn_norm.weight'),f'layers.{layer}.ffn_norm.weight',np.uint16,(DIM,)))
    moe_in=rms_model(fh.reshape(S,DIM),fnw,c).reshape(1,S,DIM)
    moe=moe_layer(ck,c,layer,moe_in)
    hidden=hc_post(moe['final'],post_attn,ffn_post,ffn_comb)
    return {'inverse_rope':inv,'grouped':woao.reshape(1,S,WOAOUT),'attn_out':final,'post_attn':post_attn,'ffn_pre':ffn_pre,'ffn_in':moe_in,'moe_out':moe['final'],'hidden':hidden}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--historical',default='/Volumes/SDXC-512/deepseek-v41-flash-mlx'); ap.add_argument('--trace-dir'); ap.add_argument('--out',default='artifacts/m4/layer2-remainder-after-sparse/result.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); trace,hrec=(Path(a.trace_dir),{'trace_dir_untracked':a.trace_dir}) if a.trace_dir else run_hist(Path(a.historical),ck)
    ent=load_manifest(trace); c=cfg(ck); got=layer2_from_exact_sparse(ck,c,ent)
    comps={
      'inverse_rope':cmp_bf16(arr(ent,'encoder.layer2.attn_inverse_rope'),got['inverse_rope'].reshape(2,H,D),1),
      'grouped_wo_a':cmp_bf16(arr(ent,'encoder.layer2.attn_grouped'),got['grouped'].reshape(2,WOAOUT),1),
      'attention_output_wo_b':cmp_bf16(arr(ent,'encoder.layer2.attn_out'),got['attn_out'].reshape(2,DIM),1),
      'post_attention_hc':cmp_bf16(arr(ent,'encoder.layer2.post_attn'),got['post_attn'].reshape(2,HC,DIM),1),
      'ffn_input':cmp_bf16(arr(ent,'encoder.layer2.ffn_in'),got['ffn_in'].reshape(2,DIM),1),
      'moe_output':cmp_bf16(arr(ent,'encoder.layer2.moe_out'),got['moe_out'].reshape(2,DIM),1),
      'final_hidden':cmp_bf16(arr(ent,'encoder.layer2.hidden'),got['hidden'].reshape(2,HC,DIM),1),
      'returned_pre_mix':cmp_f32(arr(ent,'encoder.layer2.pre_mix'),got['ffn_pre'].reshape(2,HC),1e-3),
    }
    ok=all(v.get('within_contract',False) for v in comps.values())
    rec={'schema':'ds41f.m4.layer2-remainder-after-sparse.v1','checkpoint':str(ck),'historical_export':hrec,'qualification_only':True,'production_path_changed':False,'input_boundary':'exact historical layer2 attn_core sparse output; sparse itself separately classified as official compact semantics vs historical topology-specific numerical behavior','comparisons':comps,'classification':'Layer2 remainder from sparse output COMPLETE under exact-boundary substitution' if ok else 'INCOMPLETE','ok':ok}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print(rec['classification']); return 0 if ok else 2
if __name__=='__main__': raise SystemExit(main())
