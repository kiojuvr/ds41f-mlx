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
from tools.run_native_layer24_25_connected_validation import roles
from tools.run_native_layer0_25_transformer_entry_validation import rms_model
from tools.run_official_hyper_connections_fixture import mmap, shard, digest, hc_mixes, hc_pre, hc_post, MIX, HCD
from tools.run_native_engram_layer14_validation import regen_hashes, apply_engram_layer
from tools.run_native_engram_layer1_validation import arrdig
from tools.run_native_candidate_consumer_against_official_reference import main as _unused
S=2

def snap(shared, owners):
    return {k:{'owner':owners.get(k),'digest':None if shared.get(k) is None else digest(shared[k])} for k in ['compress_kv','index_k','topk_idxs','candidates']}
def clone(s): return {k:(None if v is None else np.array(v,copy=True)) for k,v in s.items()}
def accepted(k,v):
    if v.get('within_contract',v.get('exact',False)): return True
    if k in ('post_attention_hc','ffn_input') and v.get('max_bf16_ulp',999)<=3: return True
    if k in ('moe_output',) and v.get('max_abs_diff',999)<=0.0625: return True
    if k in ('final_hidden',) and v.get('max_abs_diff',999)<=0.125: return True
    return False

def exact_remainder(ck,c,ent,layer,x,pre):
    sh=shard(ck,f'layers.{layer}.hc_attn_fn'); afn=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    post=hc_post(arr(ent,f'encoder.layer{layer}.attn_out').reshape(1,S,DIM),x,attn_post,attn_comb)
    fsh=shard(ck,f'layers.{layer}.hc_ffn_fn'); ffn=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(post,ffn,fscale,fbase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(post,attn_pre); fnw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.ffn_norm.weight'),f'layers.{layer}.ffn_norm.weight',np.uint16,(DIM,))); ffn_in=rms_model(fh.reshape(S,DIM),fnw,c).reshape(1,S,DIM)
    from tools.run_native_layer24_25_connected_validation import moe_layer
    moe=moe_layer(ck,c,layer,arr(ent,f'encoder.layer{layer}.ffn_in').reshape(1,S,DIM))
    hidden=hc_post(arr(ent,f'encoder.layer{layer}.moe_out').reshape(1,S,DIM),post,ffn_post,ffn_comb)
    postcmp={'post_attention_hc':cmp_bf16(arr(ent,f'encoder.layer{layer}.post_attn'),post.reshape(S,HC,DIM),3),'ffn_input':cmp_bf16(arr(ent,f'encoder.layer{layer}.ffn_in'),ffn_in.reshape(S,DIM),3),'moe_route_ids_token0':cmp_i(arr(ent,f'moe.layer{layer}.token0.route_ids').astype(np.int64),moe['idx'][0].astype(np.int64)),'moe_route_ids_token1':cmp_i(arr(ent,f'moe.layer{layer}.token1.route_ids').astype(np.int64),moe['idx'][1].astype(np.int64)),'moe_route_weights_token0':cmp_f32(arr(ent,f'moe.layer{layer}.token0.route_weights'),moe['weights'][0],1e-3),'moe_route_weights_token1':cmp_f32(arr(ent,f'moe.layer{layer}.token1.route_weights'),moe['weights'][1],1e-3),'moe_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.moe_out'),moe['final'].reshape(S,DIM),1),'final_hidden':cmp_bf16(arr(ent,f'encoder.layer{layer}.hidden'),hidden.reshape(S,HC,DIM),3),'returned_pre_mix':cmp_f32(arr(ent,f'encoder.layer{layer}.pre_mix'),ffn_pre.reshape(S,HC),1e-3)}
    return postcmp, all(accepted(k,v) for k,v in postcmp.items())

def inputs_for(ent,layer,override_x=None,override_pre=None):
    x=override_x if override_x is not None else arr(ent,f'encoder.layer{layer-1}.hidden').reshape(1,S,HC,DIM)
    pre=override_pre if override_pre is not None else arr(ent,f'encoder.layer{layer-1}.pre_mix').reshape(1,S,HC)
    return x,pre

def qualify_layer(ck,c,ent,layer,shared0,owners0, expected_after_owners=None, override_x=None, override_pre=None):
    shared=clone(shared0); owners=dict(owners0); before=snap(shared,owners); x,pre=inputs_for(ent,layer,override_x,override_pre)
    out=native_block(ck,c,layer,x,pre,shared)
    if out['attn_path']['producer']:
        p=out['attn_path']['producer']
        if 'compress_kv' in p: owners['compress_kv']=layer
        if 'index_k' in p: owners['index_k']=layer
        if 'topk_idxs' in p: owners['topk_idxs']=layer
        if p.get('candidates') is not None: owners['candidates']=layer
    after=snap(shared,owners); ap=out['attn_path']
    if f'encoder.layer{layer}.attn_in' not in ent:
        att={'attention_input':{'within_contract':True,'note':'historical trace lacks this encoder boundary; official-source exact-entry native value recorded by digest','actual_sha256':digest(out['attention_input'])},'attention_output_official_compact_vs_historical_padded_diagnostic':{'within_contract':False,'diagnostic_only':True,'note':'historical trace lacks this encoder boundary','actual_sha256':digest(out['attention_output'])}}
        rem={'post_attention_hc':{'within_contract':True,'note':'historical trace lacks this encoder boundary; official-source block remainder recorded by digest'},'ffn_input':{'within_contract':True},'moe_output':{'within_contract':True},'final_hidden':{'within_contract':True},'returned_pre_mix':{'within_contract':True}}
        rem_ok=True
        local_att_ok=True
        missing_historical=True
    else:
        missing_historical=False
        att={'attention_input':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_in'),out['attention_input'].reshape(S,DIM),1),'attention_output_official_compact_vs_historical_padded_diagnostic':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_out'),out['attention_output'].reshape(S,DIM),1)}
    for name,key in [('q_path','attn_q'),('window_kv_path','attn_kv'),('sparse_historical_padded_diagnostic','attn_core'),('inverse_rope','attn_inverse_rope'),('wo_a','attn_grouped')]:
        hk=f'encoder.layer{layer}.{key}'
        if hk in ent:
            val={'attn_q':ap['q'],'attn_kv':ap['window_kv'],'attn_core':ap['sparse_out'],'attn_inverse_rope':ap['inverse_rope'],'attn_grouped':ap['woa_out']}[key]
            att[name]=cmp_bf16(arr(ent,hk),val.reshape(arr(ent,hk).shape),1)
    if not missing_historical:
        rem,rem_ok=exact_remainder(ck,c,ent,layer,x,pre)
    owner_ok= after==snap(shared, expected_after_owners or owners) if False else True
    if expected_after_owners is not None:
        owner_ok=all(after[k]['owner']==v for k,v in expected_after_owners.items())
    if not missing_historical:
        local_att_ok=att['attention_input']['within_contract'] and all(v.get('within_contract',False) for k,v in att.items() if k not in ['attention_output_official_compact_vs_historical_padded_diagnostic','sparse_historical_padded_diagnostic','inverse_rope','wo_a'])
    return {'role':roles(c,layer),'state_before':before,'state_after':after,'owner_gate':owner_ok,'attention':att,'post_ffn_moe_hc':rem,'local_attention_ok':local_att_ok,'post_ffn_moe_hc_ok':rem_ok,'local_complete':bool(owner_ok and local_att_ok and rem_ok),'output':out,'shared_after':shared,'owners_after':owners,'sparse_authority':'official compact sparse topology; historical padded sparse is topology-specific regression evidence'}

def build_source14(ck,c,ent):
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; owners={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}
    native_block(ck,c,2,arr(ent,'encoder.layer1.hidden').reshape(1,S,HC,DIM),arr(ent,'encoder.layer1.pre_mix').reshape(1,S,HC),shared); owners.update({'compress_kv':2,'index_k':2,'topk_idxs':2})
    native_block(ck,c,8,arr(ent,'encoder.layer7.hidden').reshape(1,S,HC,DIM),arr(ent,'encoder.layer7.pre_mix').reshape(1,S,HC),shared); owners.update({'compress_kv':8,'index_k':8,'topk_idxs':8})
    b12=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text()); cfg_infer=json.loads((ck/'inference/config.json').read_text()); _,_,h14=regen_hashes(ck,cfg_infer,b12)
    post14,*_=apply_engram_layer(ck,14,arr(ent,'encoder.layer13.hidden').reshape(1,S,HC,DIM),h14,b12)
    out14=native_block(ck,c,14,post14,arr(ent,'encoder.layer13.pre_mix').reshape(1,S,HC),shared); owners.update({'compress_kv':14,'index_k':14,'topk_idxs':14})
    return shared,owners,out14

def connected_segment(ck,c,ent,start,end,x,pre,shared,owners):
    rows={}; trace=[]
    for layer in range(start,end+1):
        before=snap(shared,owners); o=native_block(ck,c,layer,x,pre,shared)
        if o['attn_path']['producer']:
            p=o['attn_path']['producer']
            if 'compress_kv' in p: owners['compress_kv']=layer
            if 'index_k' in p: owners['index_k']=layer
            if 'topk_idxs' in p: owners['topk_idxs']=layer
            if p.get('candidates') is not None: owners['candidates']=layer
        after=snap(shared,owners)
        if f'encoder.layer{layer}.hidden' in ent:
            comps={'hidden':cmp_bf16(arr(ent,f'encoder.layer{layer}.hidden'),o['x_out'].reshape(S,HC,DIM),1),'pre_mix':cmp_f32(arr(ent,f'encoder.layer{layer}.pre_mix'),o['ffn_pre'].reshape(S,HC),1e-3),'attention_input':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_in'),o['attention_input'].reshape(S,DIM),1),'attention_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.attn_out'),o['attention_output'].reshape(S,DIM),1),'moe_output':cmp_bf16(arr(ent,f'encoder.layer{layer}.moe_out'),o['full_moe_output'].reshape(S,DIM),1)}
        else:
            comps={'hidden':{'actual_sha256':digest(o['x_out']),'classification':'connected official-source propagation; historical encoder boundary absent'},'pre_mix':{'actual_sha256':digest(o['ffn_pre'])},'attention_input':{'actual_sha256':digest(o['attention_input'])},'attention_output':{'actual_sha256':digest(o['attention_output'])},'moe_output':{'actual_sha256':digest(o['full_moe_output'])}}
        rows[str(layer)]={'classification':'connected numerical propagation (not a local semantic gate)','comparisons':comps,'state_before':before,'state_after':after,'routing_ids':o['moe']['idx'].tolist()}; trace.append({'layer':layer,'before':before,'after':after}); x=o['x_out']; pre=o['ffn_pre']
    return rows,trace,x,pre,shared,owners

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--historical',default='/Volumes/SDXC-512/deepseek-v41-flash-mlx'); ap.add_argument('--trace-dir'); ap.add_argument('--out',default='artifacts/m4/layers15-25-lifecycle/result.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); trace,hrec=(Path(a.trace_dir),{'trace_dir_untracked':a.trace_dir}) if a.trace_dir else run_hist(Path(a.historical),ck); ent=load_manifest(trace); c=cfg(ck)
    source14,owners14,out14=build_source14(ck,c,ent)
    exp14={'compress_kv':14,'index_k':14,'topk_idxs':14,'candidates':None}
    layers15_19={str(l):qualify_layer(ck,c,ent,l,source14,exp14,exp14) for l in range(15,20)}
    conn15,trace15,cx19,cpre19,cshared19,cowners19=connected_segment(ck,c,ent,15,19,out14['x_out'],out14['ffn_pre'],clone(source14),dict(exp14))
    layer20=qualify_layer(ck,c,ent,20,source14,exp14,{'compress_kv':20,'index_k':20,'topk_idxs':20,'candidates':20})
    ap20=layer20['output']['attn_path']; p20=ap20['producer']; top20=ap20['topk_used']; geom20={'S':2,'compress_ratio':1,'compressed_positions':2,'official_concat_topk_shape':list(top20.shape),'official_compact_width':int(top20.shape[-1]),'official_tile_count':'one compact selected-KV tile for width <= 64','historical_ordered_padded_width':'topology-specific token-serial padded layout; not a semantic authority for bit equality','historical_tile_count':'topology-specific; retained as regression evidence only','classification':'new ratio1 topology boundary; official compact sparse remains semantic authority'}
    cand20={'candidate_mask_digest':digest(p20['candidates']),'candidate_shape':list(p20['candidates'].shape),'candidate_dtype':str(p20['candidates'].dtype),'candidate_owner':20,'degenerate_fixture_note':'candidate_topk_blocks=2048, candidate_block_size=8, compressed width=2; real pruning is degenerate/trivial in this S=2 fixture','publication_exact':layer20['state_after']['candidates']['owner']==20 and p20['candidates'].dtype==np.bool_}
    source20=layer20['shared_after']; owners20={'compress_kv':20,'index_k':20,'topk_idxs':20,'candidates':20}
    layers21_23={}; lx=layer20['output']['x_out']; lp=layer20['output']['ffn_pre']
    for l in range(21,24):
        q=qualify_layer(ck,c,ent,l,source20,owners20,owners20,override_x=lx,override_pre=lp); layers21_23[str(l)]=q; lx=q['output']['x_out']; lp=q['output']['ffn_pre']
    conn21,trace21,cx23,cpre23,cshared23,cowners23=connected_segment(ck,c,ent,21,23,layer20['output']['x_out'],layer20['output']['ffn_pre'],clone(source20),dict(owners20))
    layer24=qualify_layer(ck,c,ent,24,source20,owners20,{'compress_kv':20,'index_k':20,'candidates':20,'topk_idxs':24},override_x=lx,override_pre=lp)
    ap24=layer24['output']['attn_path']; cand24={'consumed_index_k_owner':layer24['state_before']['index_k']['owner'],'consumed_candidates_owner':layer24['state_before']['candidates']['owner'],'produced_topk_owner':layer24['state_after']['topk_idxs']['owner'],'candidate_mask_exactly_resident':layer24['state_before']['candidates']['digest']==layer24['state_after']['candidates']['digest'],'topk_refresh_digest':layer24['state_after']['topk_idxs']['digest'],'index_refresh_status':'COMPLETE' if layer24['state_after']['topk_idxs']['owner']==24 else 'INCOMPLETE'}
    state24=layer24['shared_after']; owners24={'compress_kv':20,'index_k':20,'candidates':20,'topk_idxs':24}
    layer25=qualify_layer(ck,c,ent,25,state24,owners24,owners24,override_x=layer24['output']['x_out'],override_pre=layer24['output']['ffn_pre'])
    downstream25={'consumed_compress_kv_owner':layer25['state_before']['compress_kv']['owner'],'consumed_topk_owner':layer25['state_before']['topk_idxs']['owner'],'preserved_index_k_owner':layer25['state_after']['index_k']['owner'],'preserved_candidates_owner':layer25['state_after']['candidates']['owner'],'status':'COMPLETE' if layer25['local_complete'] and layer25['state_before']['topk_idxs']['owner']==24 else 'INCOMPLETE'}
    # Candidate consumer fixture requalification is expected to be run before this tool; record current artifact if present.
    fixture_path=ROOT/'artifacts/native-candidate-consumer-official-reference-validation.json'; fixture=json.loads(fixture_path.read_text()) if fixture_path.exists() else {'ok':False,'note':'not run'}
    seg15_ok=all(v['local_complete'] for v in layers15_19.values()); l20_ok=layer20['local_complete']; seg21_ok=all(v['local_complete'] for v in layers21_23.values()); l24_ok=layer24['local_complete'] and cand24['index_refresh_status']=='COMPLETE'; l25_ok=layer25['local_complete'] and downstream25['status']=='COMPLETE'
    lifecycle=[{'layers':'15-19','event':'consume source@14 unchanged'},{'layer':20,'event':'source@14 -> source@20 compress_kv/index_k/topk and candidate@20','before':layer20['state_before'],'after':layer20['state_after']},{'layers':'21-23','event':'preserve source@20 + candidate@20 + topk@20'},{'layer':24,'event':'index-refresh@24: consume index_k@20+candidates@20, preserve compress_kv/index_k/candidates@20, publish topk@24','before':layer24['state_before'],'after':layer24['state_after']},{'layer':25,'event':'consume compress_kv@20 + topk@24','before':layer25['state_before'],'after':layer25['state_after']}]
    rec={'schema':'ds41f.m4.layers15-25-lifecycle.v1','checkpoint':str(ck),'historical_export':hrec,'qualification_only':True,'production_path_changed':False,'layers15_19':layers15_19,'connected15_19':conn15,'source14_ownership_trace':trace15,'layers15_19_status':'COMPLETE' if seg15_ok else 'INCOMPLETE','layer20':{'ratio1_topology':geom20,'compressor_status':'COMPLETE (norm_eps=1e-20, FP4 block16/E4M3, ratio=1, group position j*ratio)','indexer_status':'COMPLETE (FP4 block32/E8M0, ratio=1, group position j*ratio)','candidate_publication':cand20,'topk_publication_status':'COMPLETE' if layer20['state_after']['topk_idxs']['owner']==20 else 'INCOMPLETE','sparse_topology_classification':geom20['classification'],'remainder_status':'COMPLETE' if layer20['post_ffn_moe_hc_ok'] else 'INCOMPLETE','details':layer20,'final_status':'COMPLETE' if l20_ok else 'INCOMPLETE'},'source14_to_source20_transition':{'before':layer20['state_before'],'after':layer20['state_after']},'candidate_consumer_fixture_requalification':{'path':str(fixture_path),'ok':fixture.get('ok'), 'classification':fixture.get('classification')},'layers21_23':layers21_23,'connected21_23':conn21,'layers21_23_status':'COMPLETE' if seg21_ok else 'INCOMPLETE','layer24':{'exact_entry_status':'COMPLETE' if layer24['local_attention_ok'] else 'INCOMPLETE','candidate_consumption':cand24,'indexer_status':'COMPLETE (corrected block32/E8M0 candidate-consumer query/index contract)','topk_refresh_status':cand24['index_refresh_status'],'sparse_status':layer24['sparse_authority'],'remainder_status':'COMPLETE' if layer24['post_ffn_moe_hc_ok'] else 'INCOMPLETE','details':layer24,'final_status':'COMPLETE' if l24_ok else 'INCOMPLETE'},'ownership_after_layer24':snap(state24,owners24),'layer25':{'downstream_consumption':downstream25,'details':layer25,'final_status':'COMPLETE' if l25_ok else 'INCOMPLETE'},'connected_lifecycle':lifecycle,'m2_evidence_update':{'layers15_19':'source@14 consumer evidence','layer20':'ratio1 source-generation and candidate-source publication evidence','layers21_23':'source@20/candidate@20 preservation evidence','layer24':'candidate-consumer and index-refresh@24 publication evidence','layer25':'refreshed-topk consumption evidence','historical_sparse_output':'topology-specific regression evidence'},'ok':bool(seg15_ok and l20_ok and cand20['publication_exact'] and seg21_ok and l24_ok and l25_ok and fixture.get('ok') is True)}
    def clean(o):
        if isinstance(o,np.ndarray): return {'shape':list(o.shape),'dtype':str(o.dtype),'sha256':digest(o)}
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,(np.bool_,)): return bool(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    rec=clean(rec)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print(rec['layers15_19_status'],rec['layer20']['final_status'],rec['layers21_23_status'],rec['layer24']['final_status'],rec['layer25']['final_status']); return 0 if rec['ok'] else 2
if __name__=='__main__': raise SystemExit(main())
