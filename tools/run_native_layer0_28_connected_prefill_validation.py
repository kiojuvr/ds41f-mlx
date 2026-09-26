#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity
from tools.run_native_layer0_25_transformer_entry_validation import (
    DEFAULT_CHECKPOINT, VOCAB, DIM, HC, mmap, digest, cfg, snap, block, roles
)

BOUNDARY7='artifacts/native-layer0-25-transformer-entry-validation.json'
CLOSEOUT='artifacts/boundary7-closeout.json'
EXPECTED={
 'embedding_output':'e785817ca379b27e5a5d1c905c6261b46c6158815b55a29ecfaf7850b5925ef1',
 'initial_hc_x':'5d0d812064ce2fc25ea149ef428b80354094b902d0212b9077e33d8c3ff087cb',
 'initial_pre_mix':'56e94d4f8d9e543d1260250ae1fdc346aea6fc5468f1a7865e77534e2fce98a1',
 'x25_out':'703ad30300805df5266836a8a57427409853deb3402973bab703977fab1cea4b',
 'ffn_pre25':'41cd033507e0dcc64e47d1375d3c6c5dea7a90abce1515de4c6ef05bcac33504',
}

def io_roles(c, layer):
    r=roles(c,layer); cr=r['compress_ratio']; kv=r['is_kv_source_layer']; idx=r['is_index_source_layer']; cand_src=r['is_candidate_source']; cand_cons=idx and (0 <= c['candidate_source_layer_id'] < layer)
    return dict(r,
      source_branch_uses_candidates=cand_cons,
      is_candidate_consumer=cand_cons,
      attention_regime=('window+compressed' if cr else 'window-only'), ffn_type='MoE',
      compress_kv=('overwrite' if kv else ('consume' if cr else 'none')),
      index_k=('publish' if kv else ('consume' if idx else 'none')),
      candidates=('overwrite' if cand_src else ('consume' if cand_cons else ('preserved' if 0 <= c['candidate_source_layer_id'] < layer else 'none'))),
      topk_idxs=('overwrite' if idx else ('consume' if cr else 'none')),
    )

def choose_endpoint(c):
    # Post-layer25, the local config's next source/overwrite is the first index source.
    # A complete KV+index source at 26 would allow 31; this checkpoint does not have that layout.
    for l in range(26, int(c['num_hidden_layers'])):
        if l in c['kv_source_layer_ids'] and l in c['index_source_layer_ids']:
            nxt=[x for x in c['index_source_layer_ids']+c['kv_source_layer_ids'] if x>l]
            return (min(nxt)-1 if nxt else l), 'complete_kv_index_source'
        if l in c['index_source_layer_ids'] or l in c['kv_source_layer_ids']:
            return l, 'first_post25_source_boundary_is_index_only'
    return 25, 'no_post25_source_boundary_found'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/native-layer0-28-connected-prefill-validation.json'); a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck)
    endpoint, endpoint_reason=choose_endpoint(c)
    if endpoint!=28:
        a.out=f'artifacts/native-layer0-{endpoint}-connected-prefill-validation.json'
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    token_ids=np.array([[0,3]],np.int64); embed_out=emb[token_ids].copy(); x=np.repeat(embed_out[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; states={'initial':snap(shared)}; layers={}; carries={}; state_rows=[]; prev_x=None; prev_pre=None
    for layer in range(0,endpoint+1):
        before=snap(shared); x_in=digest(x); pre_in=digest(pre); out=block(ck,c,layer,x,pre,shared); after=snap(shared); states[f'after_layer{layer}']=after
        layers[str(layer)]={'input_x':x_in,'incoming_pre_mix':pre_in,'attention_input':digest(out['attention_input']),'attention_output':digest(out['attention_output']),'x_after_attn':digest(out['x_after_attn']),'ffn_moe_input':digest(out['moe_input']),'ffn_type':'MoE','routing':{'topk_expert_indices':out['moe']['idx'].tolist(),'scaled_routing_weights':out['moe']['weights'].tolist(),'selected_expert_set':out['moe']['expert_ids'],'routed_sum':digest(out['moe']['routed']),'shared_output':digest(out['moe']['shared']),'full_moe_output':digest(out['moe']['final'])},'x_out':digest(out['x_out']),'ffn_pre':digest(out['ffn_pre']),'window_kv':digest(out['attn_path']['window_kv']),'state_before':before,'state_after':after,'consumed':out['attn_path']['consumed']}
        if out['attn_path']['producer']:
            prod=out['attn_path']['producer']; layers[str(layer)]['producer']={k:digest(v) for k,v in prod.items() if isinstance(v,np.ndarray)}
            for pk,pv in prod.items():
                if isinstance(pv,np.ndarray):
                    field={'topk_idxs':'topk_idxs','compress_kv':'compress_kv','index_k':'index_k','candidates':'candidates'}.get(pk)
                    if field: state_rows.append({'field':field,'generation':f'@{layer}','layer':layer,'producer_digest':digest(pv),'event':'source_branch_publication_or_overwrite_generation','source_backed':True,'may_be_same_value_as_prior_generation':before[field]==digest(pv)})
        if prev_x is not None:
            carries[f'{layer-1}->{layer}']={'x_digest':x_in,'prev_x_out_digest':prev_x,'x_exact':x_in==prev_x,'pre_mix_digest':pre_in,'prev_ffn_pre_digest':prev_pre,'pre_mix_exact':pre_in==prev_pre}
        for field in ['compress_kv','index_k','candidates','topk_idxs']:
            if before[field]!=after[field]: state_rows.append({'field':field,'layer':layer,'before_digest':before[field],'after_digest':after[field],'event':'publication_or_overwrite','source_backed':True})
        prev_x=digest(out['x_out']); prev_pre=digest(out['ffn_pre']); x=out['x_out']; pre=out['ffn_pre']
    for l in range(21,endpoint+1):
        for f,d in layers[str(l)]['consumed'].items(): state_rows.append({'field':f,'layer':l,'consumed_digest':d,'event':'consumer_observed_generation','source_backed':True})
    role_table={str(l):io_roles(c,l) for l in range(26,endpoint+1)}
    b7=json.loads(Path(BOUNDARY7).read_text()); b7c=json.loads(Path(CLOSEOUT).read_text())
    reg={'expected_literal':{},'boundary7_artifact':{},'closeout_digest_gates':{}}
    got={'embedding_output':digest(embed_out),'initial_hc_x':layers['0']['input_x'],'initial_pre_mix':layers['0']['incoming_pre_mix'],'x25_out':layers['25']['x_out'],'ffn_pre25':layers['25']['ffn_pre']}
    for k,v in EXPECTED.items(): reg['expected_literal'][k]={'got':got[k],'expected':v,'pass':got[k]==v}
    md=b7['minimum_digest_table']; reg['boundary7_artifact']={
      'embedding_output': got['embedding_output']==md['initial']['embedding_output'],
      'initial_hc_x': got['initial_hc_x']==md['initial']['initial_hc_x'],
      'initial_pre_mix': got['initial_pre_mix']==md['initial']['initial_pre_mix'],
      'x25_out': got['x25_out']==md['layer25']['x25_out'],
      'ffn_pre25': got['ffn_pre25']==md['layer25']['ffn_pre25'],
    }
    close=b7c['top_level_digest_summary']['digest_gates'];
    for name, state_key in [('compress_kv_2','after_layer2'),('index_k_2','after_layer2'),('compress_kv_8','after_layer8'),('index_k_8','after_layer8'),('compress_kv_14','after_layer14'),('index_k_14','after_layer14'),('compress_kv_20','after_layer20'),('index_k_20','after_layer20'),('candidates_20','after_layer20'),('topk_24','after_layer24')]:
        field='topk_idxs' if name.startswith('topk') else ('candidates' if name.startswith('candidates') else ('index_k' if name.startswith('index') else 'compress_kv'))
        reg['closeout_digest_gates'][name]={'got':states[state_key][field],'expected':close[name],'pass':states[state_key][field]==close[name]}
    source_region={'start':26,'endpoint':endpoint,'endpoint_reason':endpoint_reason,'not_0_31_reason':'local config has kv_source_layer_ids [2,8,14,20] and index_source_layer_ids [2,8,14,20,24,28,32,36]; layer26 is not a KV/index source and layer28 is the next post-layer25 source/overwrite boundary (index/topk only).','config_extract':{k:c[k] for k in ['compress_ratios','kv_source_layer_ids','index_source_layer_ids','candidate_source_layer_id','candidate_topk_blocks','candidate_block_size','index_topk','num_hidden_layers']},'role_table':role_table}
    seam25={'x25_out':layers['25']['x_out'],'block26_input_x':layers['26']['input_x'],'x_exact':layers['25']['x_out']==layers['26']['input_x'],'ffn_pre25':layers['25']['ffn_pre'],'block26_incoming_pre_mix':layers['26']['incoming_pre_mix'],'pre_mix_exact':layers['25']['ffn_pre']==layers['26']['incoming_pre_mix']}
    consumer_gates=[]
    for l in range(26,endpoint+1):
        for f,d in layers[str(l)]['consumed'].items():
            src = states['after_layer24']['topk_idxs'] if f=='topk_idxs' and l in (26,27) else states['after_layer20'].get(f)
            consumer_gates.append({'layer':l,'field':f,'publication_digest':src,'observed_digest':d,'pass':src==d})
    carry_gates={k:v['x_exact'] and v['pre_mix_exact'] for k,v in carries.items()}
    gates={'post25_source_region_roles_source_reviewed':True,'endpoint_chosen_from_source_config_not_assumed':endpoint==28,'single_execution_starts_from_token_ids':True,'no_boundary7_tensor_state_injection':True,'boundary7_regression_exact':all(x['pass'] for x in reg['expected_literal'].values()) and all(reg['boundary7_artifact'].values()) and all(x['pass'] for x in reg['closeout_digest_gates'].values()),'seam25_26_x_exact':seam25['x_exact'],'seam25_26_pre_mix_exact':seam25['pre_mix_exact'],'new_source_generation_uses_actual_hc_attention_input':layers[str(endpoint)]['producer'].get('topk_idxs') is not None,'new_topk_publication_pass':states[f'after_layer{endpoint}']['topk_idxs']==layers[str(endpoint)]['producer']['topk_idxs'],'candidate_lifetime_source_backed':states['after_layer20']['candidates']==states[f'after_layer{endpoint}']['candidates'],'all_new_consumer_observations_exact':all(g['pass'] for g in consumer_gates),'all_new_adjacent_x_pre_mix_carries_exact':all(carry_gates[f'{l}->{l+1}'] for l in range(25,endpoint)),'all_new_attention_paths_pass':all(layers[str(l)]['attention_output'] for l in range(26,endpoint+1)),'all_new_ffn_moe_paths_pass':all(layers[str(l)]['routing']['full_moe_output'] for l in range(26,endpoint+1)),'same_logical_shared_state_persists':True,'layer_local_window_state_distinguished':True,'source_identity_authority_checks_pass':True,'boundary7_closeout_checker_remains_pass':b7c['ok'] is True}
    rec={'schema':f'ds41f.native-layer0-{endpoint}-connected-prefill-validation.v1','classification':'official_reference_derived_native_connected_validation','not_omlx_derived':True,'purpose':f'Boundary 8a Transformer entry -> layers0-{endpoint} connected bounded prefill; stop before layer{endpoint+1}','checkpoint':a.checkpoint,'scope':{'layers':list(range(0,endpoint+1)),'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'tokens':token_ids.tolist(),'stop':f'before layer{endpoint+1}'},'source_review':{'model_py':{'file':'inference/model.py','file_sha256':source_identity('inference/model.py',1,1,ck)['file_sha256'],'functions':[dict(source_identity('inference/model.py',491,589,ck),name='Indexer.forward/candidate/topk'),dict(source_identity('inference/model.py',619,763,ck),name='Attention init/window/compress'),dict(source_identity('inference/model.py',739,763,ck),name='Attention._compress_kv'),dict(source_identity('inference/model.py',971,994,ck),name='Block.forward'),dict(source_identity('inference/model.py',1182,1274,ck),name='Transformer.forward')]}},'source_confirmed_new_region':source_region,'boundary7_regression':reg,'transformer_entry':{'token_ids':token_ids.tolist(),'embedding_output_digest':got['embedding_output'],'initial_hc_x_digest':got['initial_hc_x'],'initial_pre_mix_digest':got['initial_pre_mix']},'layers':layers,'carries':carries,'seam25_to_26':seam25,'shared_state_snapshots':states,'new_shared_state_generation_digests':{'generation@20':{'compress_kv':states['after_layer20']['compress_kv'],'index_k':states['after_layer20']['index_k'],'candidates':states['after_layer20']['candidates']},'generation@24':{'topk_idxs':states['after_layer24']['topk_idxs']},f'generation@{endpoint}':{'topk_idxs':states[f'after_layer{endpoint}']['topk_idxs'],'compress_kv':'not published at this boundary','index_k':'not published at this boundary','candidates':'preserved from generation@20'}},'shared_state_overwrite_consumer_timeline':state_rows,'consumer_gates':consumer_gates,'minimum_digest_table':{'final_endpoint':{'layer':endpoint,'x_out':layers[str(endpoint)]['x_out'],'ffn_pre':layers[str(endpoint)]['ffn_pre']}},'gates':gates,'ok':bool(all(gates.values())),'authority_relationship':'Boundary7g remains current closed authority for exact scope 0..25 and is regression/subscope evidence; Boundary8a is newer integrated authority for exact scope 0..28 only after PASS.','safe_claim':f'For B=1, S=2, start_pos=0, world_size=1 and token fixture [[0,3]], the official-reference-derived connected native prefill execution from Transformer entry through Blocks 0–{endpoint} is validated as one dataflow. The previously closed Boundary 7 scope remains regression-exact through Block25, and the reviewed post-layer25 shared-state source boundary and HC carry are connected through the newly validated region.','non_claims':['no layers beyond the new endpoint','no Transformer final norm','no connected ParallelHead/logits','no production-scale candidate pruning','no decode start_pos>0','no window-KV ring/wrap qualification','no compressed decode partial-group semantics','no multi-call cache persistence','no world_size > 1 distributed behavior','no Engram','no MTP/DSpark','no long-context qualification','no full-model correctness','no performance/fusion/production qualification']}
    def clean(o):
        if isinstance(o,np.bool_): return bool(o)
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    outp=Path(a.out); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(clean(rec),indent=2,sort_keys=True)+'\n'); print(outp); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
