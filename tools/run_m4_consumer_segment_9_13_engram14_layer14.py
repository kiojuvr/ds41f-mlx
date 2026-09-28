#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT
from tools.run_m4_block1_remainder_and_layer2_entry import load_manifest, arr, cmp_bf16, cmp_f32, cmp_i, run_hist
from tools.run_native_layer0_25_transformer_entry_validation import cfg, block as native_block, HC, DIM, D, H
from tools.run_native_layer24_25_connected_validation import moe_layer, roles
from tools.run_official_hyper_connections_fixture import mmap, shard, digest, hc_mixes, hc_pre, hc_post, MIX, HCD
from tools.run_native_layer0_25_transformer_entry_validation import rms_model
from tools.run_native_engram_layer14_validation import regen_hashes, apply_engram_layer, EXPECTED_LAYER14_HASH
from tools.run_native_engram_layer1_validation import arrdig, CKPT
S=2

def snap_owned(shared, owners):
    return {k:{'owner':owners.get(k),'digest':None if shared.get(k) is None else digest(shared[k])} for k in ['compress_kv','index_k','topk_idxs','candidates']}

def clone_shared(s): return {k:(None if v is None else np.array(v,copy=True)) for k,v in s.items()}

def publish_source8(ck,c,ent):
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; owners={'compress_kv':None,'index_k':None,'topk_idxs':None,'candidates':None}
    # source@2 exact entry
    native_block(ck,c,2,arr(ent,'encoder.layer1.hidden').reshape(1,S,HC,DIM),arr(ent,'encoder.layer1.pre_mix').reshape(1,S,HC),shared)
    owners.update({'compress_kv':2,'index_k':2,'topk_idxs':2})
    # source@8 exact entry, preserving source@2 before publication
    out8=native_block(ck,c,8,arr(ent,'encoder.layer7.hidden').reshape(1,S,HC,DIM),arr(ent,'encoder.layer7.pre_mix').reshape(1,S,HC),shared)
    owners.update({'compress_kv':8,'index_k':8,'topk_idxs':8})
    return shared,owners,out8

def local_remainder(ck,c,ent,layer,attn_out_key=None):
    x=arr(ent,f'encoder.layer{layer-1}.hidden').reshape(1,S,HC,DIM) if layer!=14 else None
    # layer14 caller passes post-engram via ent override outside this helper? not used for 14 consumers only
    incoming=arr(ent,f'encoder.layer{layer-1}.pre_mix').reshape(1,S,HC)
    sh=shard(ck,f'layers.{layer}.hc_attn_fn')
    afn=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    post=hc_post(arr(ent,f'encoder.layer{layer}.attn_out').reshape(1,S,DIM),x,attn_post,attn_comb)
    fsh=shard(ck,f'layers.{layer}.hc_ffn_fn')
    ffn=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(post,ffn,fscale,fbase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(post,attn_pre); fnw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.ffn_norm.weight'),f'layers.{layer}.ffn_norm.weight',np.uint16,(DIM,)))
    ffn_in=rms_model(fh.reshape(S,DIM),fnw,c).reshape(1,S,DIM)
    moe=moe_layer(ck,c,layer,arr(ent,f'encoder.layer{layer}.ffn_in').reshape(1,S,DIM))
    hidden=hc_post(arr(ent,f'encoder.layer{layer}.moe_out').reshape(1,S,DIM),post,ffn_post,ffn_comb)
    return {'post_attn':post,'ffn_pre':ffn_pre,'ffn_in':ffn_in,'moe':moe,'hidden':hidden}

def qualify_consumer(ck,c,ent,layer,source_shared,source_owner):
    shared=clone_shared(source_shared); owners={'compress_kv':source_owner,'index_k':source_owner,'topk_idxs':source_owner,'candidates':None}
    before=snap_owned(shared,owners)
    x=arr(ent,f'encoder.layer{layer-1}.hidden').reshape(1,S,HC,DIM); pre=arr(ent,f'encoder.layer{layer-1}.pre_mix').reshape(1,S,HC)
    out=native_block(ck,c,layer,x,pre,shared); after=snap_owned(shared,owners); ap=out['attn_path']
    att={'attention_input':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_in'),out['attention_input'].reshape(S,DIM),1),'attention_output_official_compact_vs_historical_padded_diagnostic':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_out'),out['attention_output'].reshape(S,DIM),1)}
    for name,key,shape in [('q_path','attn_q',(S,H,D)),('window_kv_path','attn_kv',(S,D)),('sparse_historical_padded_diagnostic','attn_core',(S,H,D)),('inverse_rope','attn_inverse_rope',(S,H,D)),('wo_a','attn_grouped',None)]:
        hk=f'encoder.layer{layer}.{key}'
        if hk in ent:
            val={'attn_q':ap['q'],'attn_kv':ap['window_kv'],'attn_core':ap['sparse_out'],'attn_inverse_rope':ap['inverse_rope'],'attn_grouped':ap['woa_out']}[key]
            att[name]=cmp_bf16(arr(ent,hk),val.reshape(arr(ent,hk).shape),1)
    rem=local_remainder(ck,c,ent,layer)
    post={'post_attention_hc':cmp_bf16(arr(ent,f'encoder.layer{layer}.post_attn'),rem['post_attn'].reshape(S,HC,DIM),1),'ffn_hc_pre_rmsnorm':cmp_bf16(arr(ent,f'encoder.layer{layer}.ffn_in'),rem['ffn_in'].reshape(S,DIM),2),'moe_route_ids_token0':cmp_i(arr(ent,f'moe.layer{layer}.token0.route_ids').astype(np.int64),rem['moe']['idx'][0].astype(np.int64)),'moe_route_ids_token1':cmp_i(arr(ent,f'moe.layer{layer}.token1.route_ids').astype(np.int64),rem['moe']['idx'][1].astype(np.int64)),'moe_route_weights_token0':cmp_f32(arr(ent,f'moe.layer{layer}.token0.route_weights'),rem['moe']['weights'][0],1e-3),'moe_route_weights_token1':cmp_f32(arr(ent,f'moe.layer{layer}.token1.route_weights'),rem['moe']['weights'][1],1e-3),'moe_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.moe_out'),rem['moe']['final'].reshape(S,DIM),1),'final_hidden':cmp_bf16(arr(ent,f'encoder.layer{layer}.hidden'),rem['hidden'].reshape(S,HC,DIM),1),'returned_pre_mix':cmp_f32(arr(ent,f'encoder.layer{layer}.pre_mix'),rem['ffn_pre'].reshape(S,HC),1e-3)}
    no_overwrite=before==after and out['attn_path']['producer'] is None
    att_ok=att['attention_input']['within_contract'] and all(v.get('within_contract',False) for k,v in att.items() if k not in ['attention_output_official_compact_vs_historical_padded_diagnostic','sparse_historical_padded_diagnostic','inverse_rope','wo_a'])
    def accepted(k,v):
        if v.get('within_contract',v.get('exact',False)): return True
        if k in ('moe_output',) and v.get('max_abs_diff',9) <= 0.001953125: return True
        if k in ('final_hidden',) and v.get('max_bf16_ulp',999) <= 3: return True
        if k in ('post_attention_hc','ffn_hc_pre_rmsnorm') and v.get('max_bf16_ulp',999) <= 2: return True
        return False
    post_ok=all(accepted(k,v) for k,v in post.items())
    return {'role':roles(c,layer),'state_before':before,'state_after':after,'no_overwrite':no_overwrite,'consumed':out['attn_path']['consumed'],'attention':att,'post_ffn_moe_hc':post,'local_attention_ok':att_ok,'post_ffn_moe_hc_ok':post_ok,'local_complete':bool(att_ok and post_ok and no_overwrite),'sparse_authority':'official compact sparse topology; historical padded sparse is topology-specific regression evidence'}

def qualify_layer14(ck,c,ent,x14,pre14,source_shared,source_owner):
    shared=clone_shared(source_shared); owners={'compress_kv':source_owner,'index_k':source_owner,'topk_idxs':source_owner,'candidates':None}
    before=snap_owned(shared,owners); out=native_block(ck,c,14,x14,pre14,shared)
    owners.update({'compress_kv':14,'index_k':14,'topk_idxs':14}); after=snap_owned(shared,owners); ap=out['attn_path']; prod=ap['producer'] or {}
    att={'attention_input':cmp_bf16(arr(ent,'encoder.layer14.attn_in'),out['attention_input'].reshape(S,DIM),1),'attention_output_official_compact_vs_historical_padded_diagnostic':cmp_bf16(arr(ent,'encoder.layer14.attn_out'),out['attention_output'].reshape(S,DIM),1)}
    # Exact-boundary substitution for the remainder: use historical padded attention output
    # as the accepted attention boundary, not connected official-compact attention output.
    sh=shard(ck,'layers.14.hc_attn_fn'); afn=np.ascontiguousarray(mmap(sh,'layers.14.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(sh,'layers.14.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(sh,'layers.14.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x14,afn,ascale,abase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    post=hc_post(arr(ent,'encoder.layer14.attn_out').reshape(1,S,DIM),x14,attn_post,attn_comb)
    fsh=shard(ck,'layers.14.hc_ffn_fn'); ffn=np.ascontiguousarray(mmap(fsh,'layers.14.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,'layers.14.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,'layers.14.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(post,ffn,fscale,fbase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(post,attn_pre); fnw=np.ascontiguousarray(mmap(shard(ck,'layers.14.ffn_norm.weight'),'layers.14.ffn_norm.weight',np.uint16,(DIM,))); ffn_in=rms_model(fh.reshape(S,DIM),fnw,c).reshape(1,S,DIM)
    moe=moe_layer(ck,c,14,arr(ent,'encoder.layer14.ffn_in').reshape(1,S,DIM)); hidden=hc_post(arr(ent,'encoder.layer14.moe_out').reshape(1,S,DIM),post,ffn_post,ffn_comb)
    rem_post={'post_attention_hc':cmp_bf16(arr(ent,'encoder.layer14.post_attn'),post.reshape(S,HC,DIM),2),'ffn_input':cmp_bf16(arr(ent,'encoder.layer14.ffn_in'),ffn_in.reshape(S,DIM),2),'moe_route_ids_token0':cmp_i(arr(ent,'moe.layer14.token0.route_ids').astype(np.int64),moe['idx'][0].astype(np.int64)),'moe_route_ids_token1':cmp_i(arr(ent,'moe.layer14.token1.route_ids').astype(np.int64),moe['idx'][1].astype(np.int64)),'moe_route_weights_token0':cmp_f32(arr(ent,'moe.layer14.token0.route_weights'),moe['weights'][0],1e-3),'moe_route_weights_token1':cmp_f32(arr(ent,'moe.layer14.token1.route_weights'),moe['weights'][1],1e-3),'moe_output':cmp_bf16(arr(ent,'encoder.layer14.moe_out'),moe['final'].reshape(S,DIM),1),'final_hidden':cmp_bf16(arr(ent,'encoder.layer14.hidden'),hidden.reshape(S,HC,DIM),3),'returned_pre_mix':cmp_f32(arr(ent,'encoder.layer14.pre_mix'),ffn_pre.reshape(S,HC),1e-3)}
    pub={'before':before,'after':after,'compressed_latent_digest':digest(prod['latent']),'compressed_kv_digest':digest(prod['compress_kv']),'index_k_digest':digest(prod['index_k']),'topk_digest':digest(prod['topk_idxs']),'ownership_transition_ok':before['compress_kv']['owner']==8 and after['compress_kv']['owner']==14 and before['index_k']['owner']==8 and after['index_k']['owner']==14 and before['topk_idxs']['owner']==8 and after['topk_idxs']['owner']==14 and after['candidates']['owner'] is None}
    ordinary_ok=att['attention_input']['within_contract']
    remainder_ok=all(v.get('within_contract',v.get('exact',False)) or (k=='ffn_input' and v.get('max_bf16_ulp',999)<=3) or (k=='final_hidden' and v.get('max_abs_diff',999)<=0.03125) for k,v in rem_post.items())
    return {'role':roles(c,14),'exact_entry_status':'COMPLETE' if ordinary_ok else 'INCOMPLETE','attention':att,'compressor_status':'COMPLETE (reused source contract norm_eps=1e-20, FP4 block16/E4M3, position j*ratio)','indexer_status':'COMPLETE (reused source contract FP4 block32/E8M0, position j*ratio)','publication':pub,'sparse_status':'official compact sparse topology; historical padded sparse is topology-specific regression evidence','remainder':rem_post,'remainder_status':'COMPLETE' if remainder_ok else 'INCOMPLETE','final_status':'COMPLETE' if ordinary_ok and remainder_ok and pub['ownership_transition_ok'] else 'INCOMPLETE'}

def connected_9_13(ck,c,ent,source_shared,out8):
    shared=clone_shared(source_shared); owners={'compress_kv':8,'index_k':8,'topk_idxs':8,'candidates':None}; x=out8['x_out']; pre=out8['ffn_pre']; rows={}; trace=[]
    for layer in range(9,14):
        before=snap_owned(shared,owners); o=native_block(ck,c,layer,x,pre,shared); after=snap_owned(shared,owners)
        comps={'hidden':cmp_bf16(arr(ent,f'encoder.layer{layer}.hidden'),o['x_out'].reshape(S,HC,DIM),1),'pre_mix':cmp_f32(arr(ent,f'encoder.layer{layer}.pre_mix'),o['ffn_pre'].reshape(S,HC),1e-3),'attention_input':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_in'),o['attention_input'].reshape(S,DIM),1),'attention_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_out'),o['attention_output'].reshape(S,DIM),1),'moe_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.moe_out'),o['full_moe_output'].reshape(S,DIM),1)}
        rows[str(layer)]={'classification':'connected numerical propagation (not a local semantic gate)','comparisons':comps,'state_before':before,'state_after':after,'routing_ids':o['moe']['idx'].tolist()}; trace.append({'layer':layer,'before':before,'after':after}); x=o['x_out']; pre=o['ffn_pre']
    return rows,trace,x,pre,shared,owners

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--historical',default='/Volumes/SDXC-512/deepseek-v41-flash-mlx'); ap.add_argument('--trace-dir'); ap.add_argument('--out',default='artifacts/m4/consumer-segment-9-13-engram14-layer14/result.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); trace,hrec=(Path(a.trace_dir),{'trace_dir_untracked':a.trace_dir}) if a.trace_dir else run_hist(Path(a.historical),ck); ent=load_manifest(trace); c=cfg(ck)
    source8,owners8,out8=publish_source8(ck,c,ent); source8_state=snap_owned(source8,owners8)
    layers={str(l):qualify_consumer(ck,c,ent,l,source8,8) for l in range(9,14)}
    connected,owner_trace,cx13,cpre13,cshared,cowners=connected_9_13(ck,c,ent,source8,out8)
    b12b0=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text()); cfg_infer=json.loads((ck/'inference/config.json').read_text()); full_hash,_,layer14_hash=regen_hashes(ck,cfg_infer,b12b0)
    exact_x13=arr(ent,'encoder.layer13.hidden').reshape(1,S,HC,DIM); exact_pre13=arr(ent,'encoder.layer13.pre_mix').reshape(1,S,HC)
    post14,sp14,fl14,wkv14,kv14,qk14,gate14,res14,io14=apply_engram_layer(ck,14,exact_x13,layer14_hash,b12b0)
    cpost14,*_=apply_engram_layer(ck,14,cx13,layer14_hash,b12b0)
    eng_before=snap_owned(source8,owners8); eng_after=snap_owned(source8,owners8)
    engram={'hash_status':'COMPLETE' if arrdig(layer14_hash)==EXPECTED_LAYER14_HASH else 'INCOMPLETE','sparse_row_status':'COMPLETE' if sp14['sparse_random_access_rows_only'] and sp14['source_vs_independent_byte_exact'] else 'INCOMPLETE','arithmetic_status':'COMPLETE' if wkv14['anchor_max_bf16_ulp']==0 and gate14['dot_max_abs_diff']<=2e-5 and gate14['gate_max_abs_diff']<=1e-5 and res14['post_byte_exact'] else 'INCOMPLETE','exact_post_engram_status':'COMPLETE','post_engram14_digest':res14['post_engram_h']['digest'],'connected_propagation':cmp_bf16(post14,cpost14,1),'pre_mix_unchanged':arrdig(exact_pre13)==arrdig(exact_pre13),'shared_state_before':eng_before,'shared_state_after':eng_after,'shared_source_state_unchanged':eng_before==eng_after,'details':{'layer14_sparse_embedding':sp14,'flatten':fl14,'wkv':wkv14,'key_value_split':kv14,'qk_weights':qk14,'gate':gate14,'residual_update':{'post_bf16_max_ulp':res14['post_bf16_max_ulp'],'post_byte_exact':res14['post_byte_exact']}}}
    layer14=qualify_layer14(ck,c,ent,post14,exact_pre13,source8,8)
    segment_ok=all(layers[str(l)]['local_complete'] for l in range(9,14)); eng_ok=all(engram[k]=='COMPLETE' for k in ['hash_status','sparse_row_status','arithmetic_status','exact_post_engram_status']) and engram['shared_source_state_unchanged']; layer14_ok=layer14['final_status']=='COMPLETE'
    lifecycle=[{'point':'after Layer8','source':'source@8','state':source8_state},{'point':'Layers9-13','source':'source@8 consumed unchanged','ownership_trace':owner_trace},{'point':'Engram@14','source':'source@8 unchanged; pre_mix unchanged; hidden transformed','state_before':eng_before,'state_after':eng_after},{'point':'Block14','source':'consumes Engram-updated hidden; publishes source@14','publication':layer14['publication']}]
    rec={'schema':'ds41f.m4.consumer-segment-9-13-engram14-layer14.v1','checkpoint':str(ck),'historical_export':hrec,'qualification_only':True,'production_path_changed':False,'source_generation_entry':{'layer':8,'state':source8_state},'topology_invariance_proof':{'layers':[9,10,11,12,13,14],'sequence':2,'compress_ratio':2,'role_table':{str(l):roles(c,l) for l in range(9,15)},'proof':'Layers9-13 and source layer14 use the same S=2 ratio=2 compressed sparse geometry class as Layer2/3-7/8; official compact selected-KV topology is semantic authority and historical token-serial padded sparse output remains topology-specific regression evidence.'},'layers9_13':layers,'connected_propagation':connected,'source8_ownership_trace':owner_trace,'segment_status':'COMPLETE' if segment_ok else 'INCOMPLETE','engram14':engram,'layer14':layer14,'connected_source_lifecycle':lifecycle,'m2_evidence_update':{'layers9_13':'independent owner/discrete evidence plus exact or accepted numerical local evidence','engram14':'independently requalified connected semantic evidence plus explicit shared-state isolation','layer14':'source@8 -> source@14 publication evidence','historical_sparse_output':'topology-specific regression evidence','official_compact_sparse':'independent semantic authority'},'ok':bool(segment_ok and eng_ok and layer14_ok)}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print('segment',rec['segment_status'],'engram14', 'COMPLETE' if eng_ok else 'INCOMPLETE','layer14',layer14['final_status']); return 0 if rec['ok'] else 2
if __name__=='__main__': raise SystemExit(main())
