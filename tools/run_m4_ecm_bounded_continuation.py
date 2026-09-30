#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.source_incremental_executor import SourceDerivedIncrementalExecutor, _jsonable
from tools.run_native_ngram_hash_state_validation import arr_digest

OUT=ROOT/'artifacts/m4/reduction-trajectory-behavioral-stability/result.json'
ACT=ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz'
EXP=ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz'
CUDA_NPY=ROOT/'artifacts/m4/block0-wob-official-cuda-oracle/cuda_wob_output_bf16_u16.npy'
TARGET_LEN=8
BRANCHES=('E','C','M')
PAIRS=(('E','C'),('E','M'),('C','M'))

def sha(a:np.ndarray)->str: return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def flat(x): return np.asarray(x,dtype=np.float32).reshape(-1)
def topk(logits,k):
    f=flat(logits); idx=np.argpartition(-f,k-1)[:k]; idx=idx[np.argsort(-f[idx])]
    return [{'token':int(i),'logit':float(f[i])} for i in idx]
def logits_metrics(a,b):
    af=flat(a); bf=flat(b); d=np.abs(af-bf).astype(np.float64); denom=float(np.linalg.norm(af.astype(np.float64))*np.linalg.norm(bf.astype(np.float64)))
    return {'max_abs':float(np.max(d)),'mean_abs':float(np.mean(d)),'rms':float(np.sqrt(np.mean(d*d))),'cosine_similarity':float(np.dot(af.astype(np.float64),bf.astype(np.float64))/denom) if denom else None}
def prefix_eq(a,b):
    n=0
    for x,y in zip(a,b):
        if x['token']!=y['token']: break
        n+=1
    return n
def rank_overlap(ta,tb):
    return {'intersection_count':len({x['token'] for x in ta}&{x['token'] for x in tb}),'ordered_prefix_equality_length':prefix_eq(ta,tb)}
def jdump(x): return json.dumps(_jsonable(x),sort_keys=True)
def all_equal(vals): return all(v==vals[0] for v in vals[1:])

def discrete_map(r:dict[str,Any])->dict[str,Any]:
    d={'ngram.layer1_hash_digest':r['ngram']['layer1_hash_digest'],'ngram.layer14_hash_digest':r['ngram']['layer14_hash_digest'],'block0.moe_route_ids':r['block0']['moe_route_ids'],'block1.window_topk':r['block1']['window_topk'],'block1.moe_route_ids':r['block1']['moe_route_ids'],'block1.selected_expert_set':r['block1']['selected_expert_set']}
    for layer,ls in r['layers2_39'].items():
        d[f'layer{layer}.moe_route_ids']=ls['moe_route_ids']
        d[f'layer{layer}.selected_expert_set']=ls['selected_expert_set']
        d[f'layer{layer}.topk_used']=ls['topk_used']
        prod=ls.get('producer')
        if prod:
            # Keep discrete/publication shape, drop continuous query/score digests.
            d[f'layer{layer}.producer.topk_idxs']=prod.get('topk_idxs')
            d[f'layer{layer}.producer.candidates']=prod.get('candidates')
        pub=ls.get('publication') or {}
        pp=pub.get('pending_partial') or {}
        nl=pub.get('new_latent') or {}
        d[f'layer{layer}.publication.pending']={k:pp.get(k) for k in ('owner','written_slot','group_complete')}
        d[f'layer{layer}.publication.new_latent']={k:nl.get(k) for k in ('owner','group_index','group_position','compress_kv_before_len','compress_kv_after_len','index_k_before_len','index_k_after_len')}
    return d

def first_discrete(results):
    maps={b:discrete_map(results[b]) for b in BRANCHES}; keys=list(maps['E'].keys())
    for k in keys:
        vals=[jdump(maps[b].get(k)) for b in BRANCHES]
        if not all_equal(vals): return k
    return None

def moe_counts(results):
    route=0; expert=0; route_layers=[]; expert_layers=[]
    keys=['block0','block1']+[str(i) for i in range(2,40)]
    for k in keys:
        if k=='block0': vals=[jdump(results[b]['block0']['moe_route_ids']) for b in BRANCHES]; evals=['']*3
        elif k=='block1': vals=[jdump(results[b]['block1']['moe_route_ids']) for b in BRANCHES]; evals=[jdump(results[b]['block1']['selected_expert_set']) for b in BRANCHES]
        else: vals=[jdump(results[b]['layers2_39'][k]['moe_route_ids']) for b in BRANCHES]; evals=[jdump(results[b]['layers2_39'][k]['selected_expert_set']) for b in BRANCHES]
        if not all_equal(vals): route+=1; route_layers.append(k)
        if k!='block0' and not all_equal(evals): expert+=1; expert_layers.append(k)
    return {'route_id_differing_layer_count':route,'selected_expert_set_differing_layer_count':expert,'route_id_layers':route_layers,'selected_expert_set_layers':expert_layers}

def state_array_digests(state): return state.summary()['array_digests']
def first_array_diff(states,prefixes):
    digs={b:state_array_digests(states[b]) for b in BRANCHES}; keys=sorted(k for k in digs['E'] if any(k.startswith(p) for p in prefixes))
    for k in keys:
        vals=[digs[b].get(k) for b in BRANCHES]
        if not all_equal(vals): return k
    return None

def ownership_digest(state,key):
    v=state.ownership.get(key)
    if isinstance(v,np.ndarray): return arr_digest(v)
    return jdump(v)

def main():
    exp=np.load(EXP); act=np.load(ACT); C=np.ascontiguousarray(np.load(CUDA_NPY),dtype=np.uint16)
    E=np.ascontiguousarray(exp['block0_attention_output'],dtype=np.uint16); M=np.ascontiguousarray(act['block0_attention_output'],dtype=np.uint16)
    if sha(C)!='13c5dc43f40d4b4d0ba3771bf3b8fcdd492478f940de9ef3274f9f8c3f032d45': raise SystemExit('C digest mismatch')
    ex=SourceDerivedIncrementalExecutor(); clone_probe=ex.assert_clone_independence(); base=ex.make_state(); states={b:base.clone() for b in BRANCHES}
    initial_independence={'clone_probe_pass':bool(clone_probe['pass']),'initial_summaries_equal':states['E'].summary()==states['C'].summary()==states['M'].summary(),'distinct_state_objects':len({id(states[b]) for b in BRANCHES})==3}
    injections={'E':E,'C':C,'M':M}
    input_token=15; per=[]; token_sequences={b:[] for b in BRANCHES}; first_token_div=None
    first_discrete_overall=None; first_window=None; first_compress=None; first_index=None; first_candidate=None; first_indexer=None; first_semantic_persistent=None; first_cont_persistent=None
    min_margin={'branch':None,'transaction':None,'absolute_position':None,'margin':None}
    no_replay_pass=True; state_history_integrity=True; eng_hash_equal_all=True
    for tx in range(TARGET_LEN):
        before_hist={b:list(states[b].token_history) for b in BRANCHES}; before_pos={b:int(states[b].position) for b in BRANCHES}
        results={}; logits={}; summaries_before={b:states[b].summary() for b in BRANCHES}
        for b in BRANCHES:
            results[b]=ex.decode_one(input_token,states[b],injections[b] if tx==0 else None,return_logits=True)
            logits[b]=results[b].pop('raw_logits')
        t10={b:topk(logits[b],10) for b in BRANCHES}; t32={b:topk(logits[b],32) for b in BRANCHES}
        greedy={b:t10[b][0]['token'] for b in BRANCHES}; rank2={b:t10[b][1]['token'] for b in BRANCHES}; margins={b:float(t10[b][0]['logit']-t10[b][1]['logit']) for b in BRANCHES}
        for b,m in margins.items():
            if min_margin['margin'] is None or m < min_margin['margin']: min_margin={'branch':b,'transaction':tx,'absolute_position':results[b]['absolute_position'],'margin':m}
        for b in BRANCHES: token_sequences[b].append(greedy[b])
        pair_metrics={f'{a}_vs_{b}':logits_metrics(logits[a],logits[b]) for a,b in PAIRS}
        overlaps={f'{a}_vs_{b}':{'top10':rank_overlap(t10[a],t10[b]),'top32':rank_overlap(t32[a],t32[b])} for a,b in PAIRS}
        fd=first_discrete(results)
        if fd and first_discrete_overall is None: first_discrete_overall={'transaction':tx,'absolute_position':results['E']['absolute_position'],'path':fd}
        mc=moe_counts(results)
        # Persistent divergence after commit.
        wdiff=first_array_diff(states,['window_kv.'])
        cdiff=first_array_diff(states,['compress_kv.'])
        idiff=first_array_diff(states,['index_k.'])
        pdiff=first_array_diff(states,['compressor_kv.','compressor_score.'])
        if wdiff and first_window is None: first_window={'transaction':tx,'absolute_position':results['E']['absolute_position'],'array':wdiff}
        if cdiff and first_compress is None: first_compress={'transaction':tx,'absolute_position':results['E']['absolute_position'],'array':cdiff}
        if idiff and first_index is None: first_index={'transaction':tx,'absolute_position':results['E']['absolute_position'],'array':idiff}
        if (wdiff or cdiff or idiff or pdiff) and first_cont_persistent is None: first_cont_persistent={'transaction':tx,'absolute_position':results['E']['absolute_position'],'array':wdiff or cdiff or idiff or pdiff}
        cand_vals=[ownership_digest(states[b],'candidates') for b in BRANCHES]
        if not all_equal(cand_vals) and first_candidate is None: first_candidate={'transaction':tx,'absolute_position':results['E']['absolute_position']}
        topk_vals=[ownership_digest(states[b],'topk_idxs') for b in BRANCHES]
        if not all_equal(topk_vals) and first_indexer is None: first_indexer={'transaction':tx,'absolute_position':results['E']['absolute_position'],'owner':states['E'].ownership.get('topk_owner')}
        sem_vals=[jdump({'ownership_keys':sorted(states[b].ownership.keys()),'topk_owner':states[b].ownership.get('topk_owner'),'candidates_owner':states[b].ownership.get('candidates_owner')}) for b in BRANCHES]
        if not all_equal(sem_vals) and first_semantic_persistent is None: first_semantic_persistent={'transaction':tx,'absolute_position':results['E']['absolute_position'],'field':'ownership_metadata'}
        eng_equal=(results['E']['ngram']['layer1_hash_digest']==results['C']['ngram']['layer1_hash_digest']==results['M']['ngram']['layer1_hash_digest'] and results['E']['ngram']['layer14_hash_digest']==results['C']['ngram']['layer14_hash_digest']==results['M']['ngram']['layer14_hash_digest'])
        eng_hash_equal_all=eng_hash_equal_all and eng_equal
        hist_equal=(states['E'].token_history==states['C'].token_history==states['M'].token_history); pos_equal=(states['E'].position==states['C'].position==states['M'].position)
        state_history_integrity=state_history_integrity and hist_equal and pos_equal
        no_replay_pass=no_replay_pass and all(results[b]['history_before']==before_hist[b] and results[b]['absolute_position']==before_pos[b] for b in BRANCHES)
        rec={'transaction':tx,'input_token':input_token,'absolute_position':results['E']['absolute_position'],'per_branch':{b:{'argmax_token':greedy[b],'argmax_logit':t10[b][0]['logit'],'rank2_token':rank2[b],'rank2_logit':t10[b][1]['logit'],'top1_top2_margin':margins[b],'top10_tokens':[x['token'] for x in t10[b]],'top32_tokens':[x['token'] for x in t32[b]],'logits_digest':arr_digest(logits[b])} for b in BRANCHES},'pairwise_logits_metrics':pair_metrics,'top_overlap':overlaps,'first_discrete_divergence_in_step':fd,'moe_divergence_counts':mc,'persistent_divergence_after_commit':{'window_kv':wdiff,'compress_kv':cdiff,'index_k':idiff,'pending':pdiff,'candidate_mask':None if all_equal(cand_vals) else True,'topk_idxs':None if all_equal(topk_vals) else True},'engram_hash_equal':eng_equal,'state_history_integrity':hist_equal and pos_equal,'no_replay_step':all(results[b]['history_before']==before_hist[b] and results[b]['absolute_position']==before_pos[b] for b in BRANCHES)}
        per.append(rec)
        if not all_equal([greedy[b] for b in BRANCHES]):
            first_token_div={'transaction':tx,'absolute_position':results['E']['absolute_position'],'greedy_tokens':greedy,'margins':margins,'top10':{b:t10[b] for b in BRANCHES}}
            break
        input_token=greedy['E']
    completed=len(per)
    stable=first_token_div is None and completed==TARGET_LEN
    bounded_class='BEHAVIORALLY_STABLE_ACROSS_VALID_FP8_REDUCTION_TRAJECTORIES_BOUNDED_8' if stable else 'BEHAVIOR_SENSITIVE_TO_FP8_REDUCTION_TRAJECTORY'
    internal_discrete=first_discrete_overall is not None
    final_class='THREE_TRAJECTORY_BEHAVIORAL_STABILITY_BOUNDED_COMPLETE' if stable else 'BEHAVIOR_SENSITIVE_TO_FP8_REDUCTION_TRAJECTORY'
    existing=json.loads(OUT.read_text()) if OUT.exists() else {}
    existing['bounded_continuation']={'executed':True,'scope_note':'Only transaction0 injects E/C/M Block0 Attention outputs. Later steps are source-derived execution on each branch committed state with no repeated CUDA/MLX tensor injection; this measures propagation of one known valid backend reduction perturbation, not full CUDA-vs-MLX multi-token backend equivalence.','target_length':TARGET_LEN,'completed_length':completed,'per_transaction':per,'token_sequences':token_sequences,'first_token_divergence':first_token_div,'first_continuous_persistent_divergence':first_cont_persistent,'first_discrete_persistent_divergence':first_semantic_persistent,'first_indexer_divergence':first_indexer,'first_compressed_state_divergence':first_compress,'first_index_k_divergence':first_index,'first_candidate_mask_divergence':first_candidate,'first_topk_indexer_discrete_divergence':first_indexer,'first_window_kv_divergence':first_window,'moe_divergence_counts':[x['moe_divergence_counts'] for x in per],'minimum_margin':min_margin,'final_branch_state_summaries':{b:states[b].summary() for b in BRANCHES},'state_history_integrity_pass':state_history_integrity,'no_replay_pass':no_replay_pass,'engram_hash_equality_all_steps':eng_hash_equal_all,'first_discrete_divergence_overall':first_discrete_overall,'classification':bounded_class,'internal_discrete_classification':'INTERNAL_DISCRETE_TRAJECTORY_DIVERGENCE_WITH_BOUNDED_OUTPUT_STABILITY' if stable and internal_discrete else None}
    existing['first_token_behavioral_stability_classification']='FIRST_TOKEN_BEHAVIOR_STABLE_ACROSS_REDUCTION_TRAJECTORIES'
    existing['final_classification']=final_class
    existing['precision_policy_decision']='operator-level semantic correctness and persistent state lifecycle are authoritative; backend-valid internal discrete trajectories may diverge; connected hidden-state and full-logits bit identity are non-authoritative; greedy-token behavior has bounded-8 stability evidence, not a universal identity guarantee.' if stable else 'bounded continuation found greedy-token sensitivity to valid FP8 reduction trajectory.'
    existing['metal_exactification_status']='RETIRED; canonical Metal wo_b exactification remains retired as active frontier.'
    existing['next_frontier']='broader Milestone 4 completion decision using bounded behavioral stability evidence' if stable else 'analyze first token divergence and preceding persistent/discrete divergence'
    existing['ok']=bool(stable)
    OUT.write_text(json.dumps(_jsonable(existing),indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'completed':completed,'stable':stable,'classification':bounded_class,'final':final_class,'tokens':token_sequences,'first_token_divergence':first_token_div,'first_discrete':first_discrete_overall,'min_margin':min_margin},indent=2))
    return 0 if state_history_integrity and no_replay_pass else 1
if __name__=='__main__': raise SystemExit(main())
