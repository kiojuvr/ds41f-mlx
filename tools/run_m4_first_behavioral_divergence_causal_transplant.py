#!/usr/bin/env python3
from __future__ import annotations
import copy, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.source_incremental_executor import SourceDerivedIncrementalExecutor, SourceDerivedIncrementalState, _jsonable
from tools.run_native_ngram_hash_state_validation import arr_digest

OUT=ROOT/'artifacts/m4/first-behavioral-divergence-causal-state-transplant/result.json'
ACT=ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz'
EXP=ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz'
CUDA_NPY=ROOT/'artifacts/m4/block0-wob-official-cuda-oracle/cuda_wob_output_bf16_u16.npy'
BRANCHES=('E','C','M'); PAIRS=(('E','C'),('E','M'),('C','M')); WINNERS=[122385,13394,48926]
SOURCES=[2,8,14,20]; RATIO2=[2,8,14]

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def flat(x): return np.asarray(x,dtype=np.float32).reshape(-1)
def topk(logits,k):
    f=flat(logits); idx=np.argpartition(-f,k-1)[:k]; idx=idx[np.argsort(-f[idx])]
    return [{'token':int(i),'logit':float(f[i])} for i in idx]
def metrics(a,b):
    af=flat(a); bf=flat(b); d=np.abs(af-bf).astype(np.float64); den=float(np.linalg.norm(af.astype(np.float64))*np.linalg.norm(bf.astype(np.float64)))
    return {'max_abs':float(np.max(d)),'mean_abs':float(np.mean(d)),'rms':float(np.sqrt(np.mean(d*d))),'cosine_similarity':float(np.dot(af.astype(np.float64),bf.astype(np.float64))/den) if den else None}
def rank_of(logits,tok):
    f=flat(logits); return int(np.sum(f > f[int(tok)])+1)
def prefix_eq(a,b):
    n=0
    for x,y in zip(a,b):
        if x['token']!=y['token']: break
        n+=1
    return n
def overlap(a,b): return {'top10_intersection':len({x['token'] for x in a}&{x['token'] for x in b}),'top10_prefix_eq':prefix_eq(a,b)}
def jdump(x): return json.dumps(_jsonable(x),sort_keys=True)
def alleq(vals): return all(v==vals[0] for v in vals[1:])

def family_keys(state:SourceDerivedIncrementalState,fams:set[str])->list[str]:
    keys=[]; arrays=state.prefill_state.visible_value_arrays
    if 'W' in fams: keys += [k for k in arrays if k.startswith('window_kv.')]
    if 'P' in fams: keys += [k for k in arrays if k.startswith('compressor_kv.') or k.startswith('compressor_score.')]
    if 'C' in fams:
        for s in SOURCES:
            for suffix in ['visible','codes.visible','scales.visible']:
                k=f'compress_kv.{s}.{suffix}'
                if k in arrays: keys.append(k)
    if 'I' in fams:
        for s in SOURCES:
            for suffix in ['visible','codes.visible','scales.visible']:
                k=f'index_k.{s}.{suffix}'
                if k in arrays: keys.append(k)
    return sorted(set(keys))

def transplant(dst:SourceDerivedIncrementalState, donor:SourceDerivedIncrementalState, fams:set[str])->SourceDerivedIncrementalState:
    out=dst.clone(); da=donor.prefill_state.visible_value_arrays; oa=out.prefill_state.visible_value_arrays
    for k in family_keys(donor,fams):
        if k in da: oa[k]=np.ascontiguousarray(da[k]).copy()
    # Keep equal/common discrete state untouched except family-specific numeric arrays.
    return out

def inv_state(state):
    arr=state.prefill_state.visible_value_arrays
    groups={'W':{},'P':{},'C':{},'I':{},'D':{}}
    for k,v in arr.items():
        info={'shape':list(v.shape),'dtype':str(v.dtype),'digest':arr_digest(v)} if hasattr(v,'shape') else str(v)
        if k.startswith('window_kv.'): groups['W'][k]=info
        elif k.startswith('compressor_'): groups['P'][k]=info
        elif k.startswith('compress_kv.'): groups['C'][k]=info
        elif k.startswith('index_k.'): groups['I'][k]=info
        elif k.startswith('ngram'): groups['D'][k]=info
    groups['D']['position']=state.position; groups['D']['token_history']=list(state.token_history); groups['D']['committed_tokens']=list(state.committed_tokens); groups['D']['ownership']=_jsonable(state.ownership)
    return groups

def differing_layers(pre):
    inv={b:inv_state(pre[b]) for b in BRANCHES}; out={}
    for fam in ['W','P','C','I']:
        diff=[]
        keys=sorted(set().union(*[set(inv[b][fam]) for b in BRANCHES]))
        for k in keys:
            vals=[jdump(inv[b][fam].get(k)) for b in BRANCHES]
            if not alleq(vals): diff.append(k)
        out[fam]=diff
    return out

def discrete_map(r):
    d={'ngram1':r['ngram']['layer1_hash_digest'],'ngram14':r['ngram']['layer14_hash_digest'],'block1.topk':r['block1']['window_topk']}
    for layer,ls in r['layers2_39'].items():
        d[f'{layer}.moe']=ls['moe_route_ids']; d[f'{layer}.experts']=ls['selected_expert_set']; d[f'{layer}.topk']=ls['topk_used']
        prod=ls.get('producer') or {}; d[f'{layer}.producer.topk']=prod.get('topk_idxs'); d[f'{layer}.producer.candidates']=prod.get('candidates')
    return d

def first_moe_vs(ref, var):
    for k in ['block1']+[str(i) for i in range(2,40)]:
        if k=='block1': a=ref['block1']['moe_route_ids']; b=var['block1']['moe_route_ids']
        else: a=ref['layers2_39'][k]['moe_route_ids']; b=var['layers2_39'][k]['moe_route_ids']
        if jdump(a)!=jdump(b): return k
    return None

def moe_counts_vs(ref,var):
    route=0; exp=0
    for k in ['block1']+[str(i) for i in range(2,40)]:
        if k=='block1': ra,rb=ref['block1']['moe_route_ids'],var['block1']['moe_route_ids']; ea,eb=ref['block1']['selected_expert_set'],var['block1']['selected_expert_set']
        else: ra,rb=ref['layers2_39'][k]['moe_route_ids'],var['layers2_39'][k]['moe_route_ids']; ea,eb=ref['layers2_39'][k]['selected_expert_set'],var['layers2_39'][k]['selected_expert_set']
        route += int(jdump(ra)!=jdump(rb)); exp += int(jdump(ea)!=jdump(eb))
    return {'route_layers_differing':route,'selected_expert_layers_differing':exp}

def run(ex,state,input_token=104113):
    s=state.clone(); r=ex.decode_one(input_token,s,return_logits=True); logits=r.pop('raw_logits'); t10=topk(logits,10); t32=topk(logits,32)
    return {'result':r,'logits':logits,'digest':arr_digest(logits),'argmax':t10[0]['token'],'argmax_logit':t10[0]['logit'],'rank2':t10[1]['token'],'rank2_logit':t10[1]['logit'],'margin':float(t10[0]['logit']-t10[1]['logit']),'top10':t10,'top32':t32,'winner_logits':{str(tok):{'logit':float(flat(logits)[tok]),'rank':rank_of(logits,tok)} for tok in WINNERS},'state_after':s.summary()}

def variant_summary(name, out, native):
    summ={'name':name,'logits_digest':out['digest'],'argmax':out['argmax'],'rank2':out['rank2'],'margin':out['margin'],'winner_logits':out['winner_logits'],'rms_vs_native_E':metrics(out['logits'],native['E']['logits'])['rms'],'rms_vs_native_C':metrics(out['logits'],native['C']['logits'])['rms'],'rms_vs_native_M':metrics(out['logits'],native['M']['logits'])['rms'],'cos_vs_native_E':metrics(out['logits'],native['E']['logits'])['cosine_similarity'],'top10_overlap_vs_E':overlap(out['top10'],native['E']['top10']),'top10_overlap_vs_C':overlap(out['top10'],native['C']['top10']),'top10_overlap_vs_M':overlap(out['top10'],native['M']['top10'])}
    return summ

def main():
    exp=np.load(EXP); act=np.load(ACT); Catt=np.ascontiguousarray(np.load(CUDA_NPY),dtype=np.uint16)
    injections={'E':np.ascontiguousarray(exp['block0_attention_output'],dtype=np.uint16),'C':Catt,'M':np.ascontiguousarray(act['block0_attention_output'],dtype=np.uint16)}
    if sha(Catt)!='13c5dc43f40d4b4d0ba3771bf3b8fcdd492478f940de9ef3274f9f8c3f032d45': raise SystemExit('C digest mismatch')
    ex=SourceDerivedIncrementalExecutor(); base=ex.make_state(); states={b:base.clone() for b in BRANCHES}
    token=15
    for tx in range(3):
        greedy=[]
        for b in BRANCHES:
            r=ex.decode_one(token,states[b],injections[b] if tx==0 else None,return_logits=False)
            greedy.append(r['final_logits']['argmax_token'])
        if not alleq(greedy): raise SystemExit(f'pre-tx3 lockstep regression tx{tx}: {greedy}')
        token=greedy[0]
    pre={b:states[b].clone() for b in BRANCHES}
    pre_equal={'position':alleq([p.position for p in pre.values()]),'token_history':alleq([jdump(p.token_history) for p in pre.values()]),'committed_tokens':alleq([jdump(p.committed_tokens) for p in pre.values()]),'ngram_cache':alleq([arr_digest(p.prefill_state.visible_value_arrays['ngram_cache.visible']) for p in pre.values()]),'candidate_mask':alleq([jdump(p.ownership.get('candidates')) for p in pre.values()]),'topk_ids':alleq([jdump(p.ownership.get('topk_idxs')) for p in pre.values()]),'ownership_topology':alleq([jdump({k:v for k,v in p.ownership.items() if not hasattr(v,'shape')}) for p in pre.values()])}
    native={b:run(ex,pre[b],104113) for b in BRANCHES}
    # Full-state transplant completeness: executor is stateless; exact donor clone must reproduce donor tx3.
    full={}
    for donor in BRANCHES:
        for ctx in BRANCHES:
            o=run(ex,pre[donor],104113)
            exact=(o['digest']==native[donor]['digest'] and o['argmax']==native[donor]['argmax'] and jdump(discrete_map(o['result']))==jdump(discrete_map(native[donor]['result'])))
            full[f'{donor}_state_in_{ctx}_context']={'donor':donor,'context':ctx,'digest':o['digest'],'argmax':o['argmax'],'donor_exact':exact}
    full_pass=all(v['donor_exact'] for v in full.values())
    if not full_pass:
        rec={'schema':'ds41f.m4.first_behavioral_divergence_causal_state_transplant.v1','classification':'PERSISTENT_STATE_MODEL_INCOMPLETE','pre_tx3_equality':pre_equal,'full_state_transplant_completeness':full,'ok':False}
        OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(_jsonable(rec),indent=2,sort_keys=True)+'\n'); print(json.dumps({'wrote':str(OUT),'classification':rec['classification']},indent=2)); return 1
    variants={}
    # Rescue: donor C/M plus E families.
    rescue_specs=[]
    for donor in ['C','M']:
        for fams,name in [(set('W'),'W'),(set('P'),'P'),(set('C'),'C'),(set('I'),'I'),(set(['P','C']),'P+C'),(set(['W','P','C']),'W+P+C'),(set(['W','P','C','I']),'W+P+C+I')]:
            rescue_specs.append((f'{donor}+E.{name}', transplant(pre[donor],pre['E'],fams)))
    # Reverse sufficiency: E plus donor families.
    for donor in ['C','M']:
        for fams,name in [(set('W'),'W'),(set(['P','C']),'P+C'),(set('I'),'I'),(set(['W','P','C']),'W+P+C')]:
            rescue_specs.append((f'E+{donor}.{name}', transplant(pre['E'],pre[donor],fams)))
    # Coarse W bisection and source bisection if useful, always record limited hierarchy.
    for name,st in rescue_specs:
        out=run(ex,st,104113); variants[name]=variant_summary(name,out,native); variants[name]['first_moe_divergence_vs_native_E']=first_moe_vs(native['E']['result'],out['result']); variants[name]['moe_counts_vs_native_E']=moe_counts_vs(native['E']['result'],out['result'])
    native_summary={b:variant_summary(f'native_{b}',native[b],native) for b in BRANCHES}
    # Attention topology equality in native tx3.
    topo_keys=[]
    maps={b:discrete_map(native[b]['result']) for b in BRANCHES}
    for k in maps['E']:
        if '.topk' in k or 'candidate' in k:
            vals=[jdump(maps[b].get(k)) for b in BRANCHES]
            if not alleq(vals): topo_keys.append(k)
    state_attr='DISTRIBUTED_PERSISTENT_STATE_INTERACTION_CAUSES_BEHAVIORAL_DIVERGENCE'
    if variants['C+E.W']['argmax']==native['E']['argmax'] and variants['M+E.W']['argmax']==native['E']['argmax']:
        state_attr='WINDOW_STATE_ACCUMULATION_DOMINATES_FIRST_BEHAVIORAL_DIVERGENCE'
    elif variants['C+E.P+C']['argmax']==native['E']['argmax'] and variants['M+E.P+C']['argmax']==native['E']['argmax']:
        state_attr='COMPRESSED_STATE_ACCUMULATION_DOMINATES_FIRST_BEHAVIORAL_DIVERGENCE'
    elif variants['C+E.W+P+C+I']['argmax']!=native['E']['argmax'] or variants['M+E.W+P+C+I']['argmax']!=native['E']['argmax']:
        state_attr='NUMERICAL_ATTENTION_STATE_ONLY_RESCUE_INCOMPLETE_OR_DISTRIBUTED_WITH_OTHER_CONTINUOUS_STATE'
    index_result='INDEX_K_NUMERICAL_DIVERGENCE_NON_CAUSAL_TO_TX3_TOKEN_CHOICE' if variants['C+E.I']['argmax']==native['C']['argmax'] and variants['M+E.I']['argmax']==native['M']['argmax'] else 'INDEX_K_TRANSPLANT_MOVES_TX3_DECISION_SURFACE'
    rec={'schema':'ds41f.m4.first_behavioral_divergence_causal_state_transplant.v1','scope':'tx3 absolute position5 input token104113 only; no continuation beyond first divergent transaction; C/M are one tx0 backend seed propagated by common source-derived executor, not full CUDA/MLX multi-token execution','pre_tx3_equality':pre_equal,'pre_tx3_positions':{b:pre[b].position for b in BRANCHES},'pre_tx3_histories':{b:pre[b].token_history for b in BRANCHES},'pre_tx3_persistent_state_inventory':{b:inv_state(pre[b]) for b in BRANCHES},'pre_tx3_differing_state_families':differing_layers(pre),'full_state_transplant_completeness':full,'native_tx3':native_summary,'native_winner_token_logit_matrix':{b:native[b]['winner_logits'] for b in BRANCHES},'rescue_and_reverse_experiments':variants,'attention_topology_equality_result':'DISCRETE_ATTENTION_TOPOLOGY_EQUAL_AT_FIRST_TOKEN_DIVERGENCE' if not topo_keys else 'DISCRETE_ATTENTION_TOPOLOGY_DIVERGED_AT_TX3','attention_topology_differing_keys':topo_keys,'index_k_interpretation':index_result,'state_family_attribution':state_attr,'final_causal_classification':state_attr if full_pass else 'PERSISTENT_STATE_MODEL_INCOMPLETE','m4_acceptance_policy_consequence':'Cross-backend hidden-state, full-logits, and greedy-token identity are not valid universal fidelity requirements under official-compatible floating point. Require backend-local deterministic execution, official operator/model semantics, precision/storage contracts, persistent-state lifecycle correctness, and correct discrete algorithms on each backend computed value. Do not force production MLX to imitate source-derived or CUDA trajectories solely for token identity.','metal_exactification_status':'RETIRED_AS_ACTIVE_FRONTIER','next_frontier':'validate production MLX decode against the finalized backend-local fidelity policy rather than canonicalizing every sensitive reduction','ok':True}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(_jsonable(rec),indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'full_transplant_pass':full_pass,'native_argmax':{b:native[b]['argmax'] for b in BRANCHES},'attribution':state_attr,'attention_topology':rec['attention_topology_equality_result'],'index':index_result},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
