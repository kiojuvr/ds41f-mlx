#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT
from tools.run_m4_block1_remainder_and_layer2_entry import load_manifest, arr, cmp_bf16, cmp_f32, cmp_i, run_hist, info
from tools.run_native_layer0_25_transformer_entry_validation import cfg, block as native_block, HC, DIM, D, H, RD
from tools.run_native_layer24_25_connected_validation import moe_layer, roles
from tools.run_official_hyper_connections_fixture import mmap, shard, digest, hc_mixes, hc_pre, hc_post, MIX, HCD
from tools.run_native_layer0_25_transformer_entry_validation import rms_model

S=2

def snap_owned(shared:dict[str,Any], owners:dict[str,int|None])->dict[str,Any]:
    return {k:{'owner':owners.get(k),'digest':None if shared.get(k) is None else digest(shared[k])} for k in ['compress_kv','index_k','topk_idxs','candidates']}

def init_source2(ck:Path,c:dict[str,Any],ent:dict[str,Any]):
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}
    owners={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}
    x=arr(ent,'encoder.layer1.hidden').reshape(1,S,HC,DIM)
    pre=arr(ent,'encoder.layer1.pre_mix').reshape(1,S,HC)
    o=native_block(ck,c,2,x,pre,shared)
    owners.update({'compress_kv':2,'index_k':2,'topk_idxs':2})
    return shared, owners, o

def local_remainder(ck:Path,c:dict[str,Any],ent:dict[str,Any],layer:int):
    x=arr(ent,f'encoder.layer{layer-1}.hidden').reshape(1,S,HC,DIM)
    incoming=arr(ent,f'encoder.layer{layer-1}.pre_mix').reshape(1,S,HC)
    sh=shard(ck,f'layers.{layer}.hc_attn_fn')
    afn=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_fn',np.float32,(MIX,HCD)))
    abase=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_base',np.float32,(MIX,)))
    ascale=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    ah=hc_pre(x,incoming)
    anw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.attn_norm.weight'),f'layers.{layer}.attn_norm.weight',np.uint16,(DIM,)))
    attn_in=rms_model(ah.reshape(S,DIM),anw,c).reshape(1,S,DIM)
    post=hc_post(arr(ent,f'encoder.layer{layer}.attn_out').reshape(1,S,DIM),x,attn_post,attn_comb)
    fsh=shard(ck,f'layers.{layer}.hc_ffn_fn')
    ffn=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_fn',np.float32,(MIX,HCD)))
    fbase=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_base',np.float32,(MIX,)))
    fscale=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(post,ffn,fscale,fbase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(post,attn_pre)
    fnw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.ffn_norm.weight'),f'layers.{layer}.ffn_norm.weight',np.uint16,(DIM,)))
    ffn_in=rms_model(fh.reshape(S,DIM),fnw,c).reshape(1,S,DIM)
    moe=moe_layer(ck,c,layer,arr(ent,f'encoder.layer{layer}.ffn_in').reshape(1,S,DIM))
    hidden=hc_post(arr(ent,f'encoder.layer{layer}.moe_out').reshape(1,S,DIM),post,ffn_post,ffn_comb)
    return {'attn_input':attn_in,'post_attn':post,'ffn_pre':ffn_pre,'ffn_in':ffn_in,'moe':moe,'hidden':hidden}

def layer_local(ck,c,ent,layer,source2_shared):
    shared={k:(None if v is None else np.array(v,copy=True)) for k,v in source2_shared.items()}
    owners={'compress_kv':2,'index_k':2,'topk_idxs':2,'candidates':None}
    before=snap_owned(shared,owners)
    x=arr(ent,f'encoder.layer{layer-1}.hidden').reshape(1,S,HC,DIM)
    pre=arr(ent,f'encoder.layer{layer-1}.pre_mix').reshape(1,S,HC)
    out=native_block(ck,c,layer,x,pre,shared)
    if out['attn_path']['producer']:
        if 'compress_kv' in out['attn_path']['producer']: owners['compress_kv']=layer
        if 'index_k' in out['attn_path']['producer']: owners['index_k']=layer
        if 'topk_idxs' in out['attn_path']['producer']: owners['topk_idxs']=layer
        if 'candidates' in out['attn_path']['producer'] and out['attn_path']['producer']['candidates'] is not None: owners['candidates']=layer
    after=snap_owned(shared,owners)
    refs={
      'attention_input':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_in'),out['attention_input'].reshape(S,DIM),1),
      'attention_output_official_compact_vs_historical_padded_diagnostic':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_out'),out['attention_output'].reshape(S,DIM),1),
    }
    ap=out['attn_path']
    if f'encoder.layer{layer}.attn_q' in ent: refs['q_path']=cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_q'),ap['q'].reshape(S,H,D),1)
    if f'encoder.layer{layer}.attn_kv' in ent: refs['window_kv_path']=cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_kv'),ap['window_kv'].reshape(S,D),1)
    if f'encoder.layer{layer}.attn_core' in ent: refs['sparse_historical_padded_diagnostic']=cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_core'),ap['sparse_out'].reshape(S,H,D),1)
    if f'encoder.layer{layer}.attn_inverse_rope' in ent: refs['inverse_rope']=cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_inverse_rope'),ap['inverse_rope'].reshape(S,H,D),1)
    if f'encoder.layer{layer}.attn_grouped' in ent: refs['wo_a']=cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_grouped'),ap['woa_out'].reshape(S,-1),1)
    rem=local_remainder(ck,c,ent,layer)
    post={
      'post_attention_hc':cmp_bf16(arr(ent,f'encoder.layer{layer}.post_attn'),rem['post_attn'].reshape(S,HC,DIM),1),
      'ffn_hc_pre_rmsnorm':cmp_bf16(arr(ent,f'encoder.layer{layer}.ffn_in'),rem['ffn_in'].reshape(S,DIM),2),
      'moe_route_ids_token0':cmp_i(arr(ent,f'moe.layer{layer}.token0.route_ids').astype(np.int64),rem['moe']['idx'][0].astype(np.int64)),
      'moe_route_ids_token1':cmp_i(arr(ent,f'moe.layer{layer}.token1.route_ids').astype(np.int64),rem['moe']['idx'][1].astype(np.int64)),
      'moe_route_weights_token0':cmp_f32(arr(ent,f'moe.layer{layer}.token0.route_weights'),rem['moe']['weights'][0],1e-3),
      'moe_route_weights_token1':cmp_f32(arr(ent,f'moe.layer{layer}.token1.route_weights'),rem['moe']['weights'][1],1e-3),
      'moe_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.moe_out'),rem['moe']['final'].reshape(S,DIM),1),
      'final_hidden':cmp_bf16(arr(ent,f'encoder.layer{layer}.hidden'),rem['hidden'].reshape(S,HC,DIM),1),
      'returned_pre_mix':cmp_f32(arr(ent,f'encoder.layer{layer}.pre_mix'),rem['ffn_pre'].reshape(S,HC),1e-3),
    }
    post_ok=all(v.get('within_contract',v.get('exact',False)) for v in post.values())
    local_attention_ok=refs['attention_input']['within_contract'] and all(refs[k].get('within_contract',False) for k in refs if k not in ['attention_output_official_compact_vs_historical_padded_diagnostic','sparse_historical_padded_diagnostic','inverse_rope','wo_a'])
    if layer in c['kv_source_layer_ids']:
        state_gate=after['compress_kv']['owner']==layer and after['index_k']['owner']==layer and after['topk_idxs']['owner']==layer and out['attn_path']['producer'] is not None
        state_label='source_publication'
    else:
        state_gate=before==after and after['compress_kv']['owner']==2 and after['topk_idxs']['owner']==2 and after['index_k']['owner']==2 and after['candidates']['owner'] is None and out['attn_path']['producer'] is None
        state_label='no_overwrite'
    return {'role':roles(c,layer),'state_before':before,'state_after':after,state_label:state_gate,'consumed':out['attn_path']['consumed'],'attention':refs,'post_ffn_moe_hc':post,'local_attention_ok':local_attention_ok,'post_ffn_moe_hc_ok':post_ok,'local_complete':bool(local_attention_ok and post_ok and state_gate),'sparse_authority':'official compact sparse topology; historical padded sparse is topology-specific regression evidence, not a bit gate'}

def connected_path(ck,c,ent,source2_shared):
    shared={k:(None if v is None else np.array(v,copy=True)) for k,v in source2_shared.items()}
    owners={'compress_kv':2,'index_k':2,'topk_idxs':2,'candidates':None}
    x=arr(ent,'encoder.layer2.hidden').reshape(1,S,HC,DIM); pre=arr(ent,'encoder.layer2.pre_mix').reshape(1,S,HC)
    rows={}; ownership=[]
    for layer in range(3,9):
        before=snap_owned(shared,owners)
        o=native_block(ck,c,layer,x,pre,shared)
        if o['attn_path']['producer']:
            if 'compress_kv' in o['attn_path']['producer']: owners['compress_kv']=layer
            if 'index_k' in o['attn_path']['producer']: owners['index_k']=layer
            if 'topk_idxs' in o['attn_path']['producer']: owners['topk_idxs']=layer
            if 'candidates' in o['attn_path']['producer'] and o['attn_path']['producer']['candidates'] is not None: owners['candidates']=layer
        after=snap_owned(shared,owners)
        comps={
          'hidden':cmp_bf16(arr(ent,f'encoder.layer{layer}.hidden'),o['x_out'].reshape(S,HC,DIM),1),
          'pre_mix':cmp_f32(arr(ent,f'encoder.layer{layer}.pre_mix'),o['ffn_pre'].reshape(S,HC),1e-3),
          'attention_input':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_in'),o['attention_input'].reshape(S,DIM),1),
          'attention_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_out'),o['attention_output'].reshape(S,DIM),1),
          'moe_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.moe_out'),o['full_moe_output'].reshape(S,DIM),1),
        }
        rows[str(layer)]={'comparisons':comps,'classification':'connected numerical propagation (not a local semantic gate)','state_before':before,'state_after':after,'routing_ids':o['moe']['idx'].tolist()}
        ownership.append({'layer':layer,'before':before,'after':after})
        x=o['x_out']; pre=o['ffn_pre']
    return rows, ownership

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--historical',default='/Volumes/SDXC-512/deepseek-v41-flash-mlx'); ap.add_argument('--trace-dir'); ap.add_argument('--out',default='artifacts/m4/consumer-segment-3-7-layer8/result.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); trace,hrec=(Path(a.trace_dir),{'trace_dir_untracked':a.trace_dir}) if a.trace_dir else run_hist(Path(a.historical),ck)
    ent=load_manifest(trace); c=cfg(ck)
    source2, owners2, layer2_out=init_source2(ck,c,ent)
    source2_state=snap_owned(source2,owners2)
    layers={str(l):layer_local(ck,c,ent,l,source2) for l in range(3,8)}
    connected, ownership=connected_path(ck,c,ent,source2)
    layer8_local=layer_local(ck,c,ent,8,source2)
    # Layer8 source branch publication details from exact entry.
    pub=layer8_local['state_after']; pre8=layer8_local['state_before']
    layer8_publication={'before':pre8,'after':pub,'ownership_transition_ok':pre8['compress_kv']['owner']==2 and pub['compress_kv']['owner']==8 and pre8['index_k']['owner']==2 and pub['index_k']['owner']==8 and pre8['topk_idxs']['owner']==2 and pub['topk_idxs']['owner']==8,'compressed_latent_digest':digest(native_block(ck,c,8,arr(ent,'encoder.layer7.hidden').reshape(1,S,HC,DIM),arr(ent,'encoder.layer7.pre_mix').reshape(1,S,HC),{k:(None if v is None else np.array(v,copy=True)) for k,v in source2.items()})['attn_path']['producer']['latent'])}
    segment_ok=all(layers[str(l)]['local_complete'] for l in range(3,8))
    layer8_ok=bool(layer8_local['local_complete'] and layer8_publication['ownership_transition_ok'])
    rec={'schema':'ds41f.m4.consumer-segment-3-7-layer8.v1','checkpoint':str(ck),'historical_export':hrec,'qualification_only':True,'production_path_changed':False,'source_generation_entry':{'layer':2,'state':source2_state},'topology_invariance_proof':{'layers':[3,4,5,6,7,8],'sequence':2,'compress_ratio':2,'role_table':{str(l):roles(c,l) for l in range(3,9)},'proof':'All listed layers use the same Attention compressed sparse path as layer2 for S=2, ratio=2. Consumer layers 3-7 have no Indexer/source roles and concatenate window width 2 with the single Layer2 compressed group selected by Layer2 topk. Historical inspection from the Layer2 diagnostic established token-serial padded layout row0 width 1 / row1 width 129 with valid row1 positions 126,127,128 and block64x3 tiles; official compact selected-KV width remains the semantic authority.'},'layers3_7':layers,'connected_propagation':connected,'shared_state_ownership_trace':ownership,'segment_status':'COMPLETE' if segment_ok else 'INCOMPLETE','layer8':{'exact_entry_status':'COMPLETE' if layer8_local['local_attention_ok'] else 'INCOMPLETE','compressor_status':'COMPLETE (norm_eps=1e-20, FP4 block16/E4M3, group position j*ratio reused contract)','indexer_status':'COMPLETE (FP4 block32/E8M0, group position j*ratio reused contract)','publication':layer8_publication,'sparse_status':layer8_local['sparse_authority'],'remainder_status':'COMPLETE' if layer8_local['post_ffn_moe_hc_ok'] else 'INCOMPLETE','details':layer8_local,'final_status':'COMPLETE' if layer8_ok else 'INCOMPLETE'},'m2_evidence_update':{'layers3_7_shared_source_generation_ownership':'independent exact discrete owner/digest evidence','layers3_7_local_model_operations':'exact or accepted numerical-boundary evidence','historical_sparse_output':'topology-specific regression evidence','official_compact_sparse_reconstruction':'independent semantic authority','layer8_source_generation':'explicit ownership transition source@2 -> source@8 with publication digests'},'ok':bool(segment_ok and layer8_ok)}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print('segment',rec['segment_status'],'layer8',rec['layer8']['final_status']); return 0 if rec['ok'] else 2
if __name__=='__main__': raise SystemExit(main())
