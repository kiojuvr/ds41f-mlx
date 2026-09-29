#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.source_incremental_executor import SourceDerivedFirstIncrementalExecutor, _jsonable
from tools.run_native_ngram_hash_state_validation import arr_digest

OUT=ROOT/'artifacts/m4/reduction-trajectory-behavioral-stability/result.json'
ACT=ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz'
EXP=ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz'
CUDA_JSON=ROOT/'artifacts/m4/block0-wob-official-cuda-oracle/cuda-oracle-result.json'
CUDA_NPY=ROOT/'artifacts/m4/block0-wob-official-cuda-oracle/cuda_wob_output_bf16_u16.npy'
EXEC=ROOT/'ds41f_mlx/source_incremental_executor.py'


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x,dtype=np.float32).reshape(-1)

def logits_metrics(a: np.ndarray,b: np.ndarray) -> dict[str,Any]:
    af=f32(a); bf=f32(b); d=np.abs(af-bf).astype(np.float64)
    denom=float(np.linalg.norm(af.astype(np.float64))*np.linalg.norm(bf.astype(np.float64)))
    return {'max_abs':float(np.max(d)),'mean_abs':float(np.mean(d)),'rms':float(np.sqrt(np.mean(d*d))),'cosine_similarity':float(np.dot(af.astype(np.float64),bf.astype(np.float64))/denom) if denom else None}

def topk(logits: np.ndarray,k:int) -> list[dict[str,Any]]:
    flat=f32(logits); idx=np.argpartition(-flat,k-1)[:k]; idx=idx[np.argsort(-flat[idx])]
    return [{'token':int(i),'logit':float(flat[i])} for i in idx]

def margin(top:list[dict[str,Any]]) -> float:
    return float(top[0]['logit']-top[1]['logit'])

def prefix_eq(a:list[dict[str,Any]],b:list[dict[str,Any]]) -> int:
    n=0
    for x,y in zip(a,b):
        if x['token']!=y['token']: break
        n+=1
    return n

def ranking_pair(a,b,k:int) -> dict[str,Any]:
    ta=topk(a,k); tb=topk(b,k); sa={x['token'] for x in ta}; sb={x['token'] for x in tb}
    return {'intersection_count':len(sa&sb),'ordered_prefix_equality_length':prefix_eq(ta,tb)}

def discrete_pub(x: Any) -> Any:
    if not isinstance(x,dict): return x
    y={k:v for k,v in x.items() if k not in ('index_query_digest','index_score_digest','query_digest','score_digest')}
    return _jsonable(y)

def discrete_branch(r:dict[str,Any]) -> dict[str,Any]:
    d={
      'ngram.layer1_hash_digest':r['ngram']['layer1_hash_digest'],
      'ngram.layer14_hash_digest':r['ngram']['layer14_hash_digest'],
      'block0.moe_route_ids':r['block0']['moe_route_ids'],
      'block1.window_topk':r['block1']['window_topk'],
      'block1.moe_route_ids':r['block1']['moe_route_ids'],
      'block1.selected_expert_set':r['block1']['selected_expert_set'],
      'layer2_entry.window_topk':r['layer2_entry']['window_topk'],
      'layer2_entry.indexer_topk':r['layer2_entry']['indexer_topk'],
      'layer2_entry.compressor_partial':{k:r['layer2_entry']['compressor_partial'][k] for k in ['written_slot','group_complete','new_latent_produced']},
      'layer2_entry.compress_kv_cache':r['layer2_entry']['compress_kv_cache'],
      'layer2_entry.new_key_publication':r['layer2_entry']['new_key_publication'],
      'engram14.hash_digest':r['engram14']['hash_digest'] if r.get('engram14') else None,
    }
    for layer,ls in r['layers2_39'].items():
        d[f'layer{layer}.moe_route_ids']=ls['moe_route_ids']
        d[f'layer{layer}.selected_expert_set']=ls['selected_expert_set']
        d[f'layer{layer}.topk_used']=ls['topk_used']
        d[f'layer{layer}.producer']=discrete_pub(ls['producer'])
        d[f'layer{layer}.publication']=discrete_pub(ls['publication'])
        d[f'layer{layer}.consumed']=discrete_pub(ls['consumed'])
    return d

def first_diff(branches:dict[str,dict[str,Any]],keys:list[str]) -> str|None:
    for k in keys:
        vals=[json.dumps(branches[b].get(k),sort_keys=True) for b in ['E','C','M']]
        if not (vals[0]==vals[1]==vals[2]): return k
    return None

def persistent_semantic(s:dict[str,Any]) -> dict[str,Any]:
    return {k:s[k] for k in ['position','token_history','committed_tokens','ownership']}

def scan_hardcoded() -> list[dict[str,Any]]:
    out=[]
    text=EXEC.read_text().splitlines()
    pats=[r'token_id != 15',r'state\.position != 2',r'_ngram_for_token15',r'tmap\[15\]',r'token_history.*\[0, 3\]',r'start_pos=2',r'freqs\(64,3',r'co_full\[2:3\]',r'si_full\[2:3\]',r'\+3',r'absolute_position": 2',r'position 2',r'token15']
    rg=re.compile('|'.join(pats))
    for i,line in enumerate(text,1):
        if rg.search(line): out.append({'line':i,'text':line.strip()})
    return out

def main():
    act=np.load(ACT); exp=np.load(EXP); cuda=json.loads(CUDA_JSON.read_text()); C=np.ascontiguousarray(np.load(CUDA_NPY),dtype=np.uint16)
    E=np.ascontiguousarray(exp['block0_attention_output'],dtype=np.uint16); M=np.ascontiguousarray(act['block0_attention_output'],dtype=np.uint16)
    assert sha(C)=='13c5dc43f40d4b4d0ba3771bf3b8fcdd492478f940de9ef3274f9f8c3f032d45'
    executor=SourceDerivedFirstIncrementalExecutor()
    clone=executor.assert_clone_independence()
    base=executor.make_state(); state_E=base.clone(); state_C=base.clone(); state_M=base.clone()
    independence={'executor_clone_probe_pass':bool(clone['pass']),'base_vs_E_before_equal':base.summary()==state_E.summary(),'E_C_M_before_equal':state_E.summary()==state_C.summary()==state_M.summary(),'distinct_state_objects':len({id(state_E),id(state_C),id(state_M)})==3,'distinct_prefill_objects':len({id(state_E.prefill_state),id(state_C.prefill_state),id(state_M.prefill_state)})==3}
    normal_E_state=executor.make_state(); injected_E_state=executor.make_state()
    normal_E=executor.decode_one(15, normal_E_state, None, return_logits=True)
    injected_E=executor.decode_one(15, injected_E_state, E, return_logits=True)
    raw_normal_E=normal_E.pop('raw_logits'); raw_injected_E=injected_E.pop('raw_logits')
    guard_keys=['block0','engram1','block1','layers2_39','engram14','selected_continuous_digests','final_logits','state_summary_after']
    e_guard={k:json.dumps(_jsonable(normal_E[k]),sort_keys=True)==json.dumps(_jsonable(injected_E[k]),sort_keys=True) for k in guard_keys}
    e_guard['full_logits_raw_equal']=bool(np.array_equal(raw_normal_E,raw_injected_E))
    branches={
      'E':executor.decode_one(15,state_E,E,return_logits=True),
      'C':executor.decode_one(15,state_C,C,return_logits=True),
      'M':executor.decode_one(15,state_M,M,return_logits=True),
    }
    logits={k:v.pop('raw_logits') for k,v in branches.items()}
    top10={k:topk(v,10) for k,v in logits.items()}; top32={k:topk(v,32) for k,v in logits.items()}
    pairs=[('E','C'),('E','M'),('C','M')]
    pair_metrics={f'{a}_vs_{b}':logits_metrics(logits[a],logits[b]) for a,b in pairs}
    top_overlap={f'{a}_vs_{b}':{'top10':ranking_pair(logits[a],logits[b],10),'top32':ranking_pair(logits[a],logits[b],32),'rank1_identity_equal':top10[a][0]['token']==top10[b][0]['token'],'rank2_identity_equal':top10[a][1]['token']==top10[b][1]['token'],'rank1_rank2_ordering_equal':[top10[a][0]['token'],top10[a][1]['token']]==[top10[b][0]['token'],top10[b][1]['token']]} for a,b in pairs}
    discrete={k:discrete_branch(v) for k,v in branches.items()}
    dkeys=list(discrete['E'].keys()); first_discrete=first_diff(discrete,dkeys)
    sem={k:persistent_semantic(v['state_summary_after']) for k,v in branches.items()}
    first_sem=None
    for k in ['position','token_history','committed_tokens','ownership']:
        vals=[json.dumps(_jsonable(sem[b][k]),sort_keys=True) for b in ['E','C','M']]
        if not (vals[0]==vals[1]==vals[2]): first_sem=k; break
    arrdig={k:v['state_summary_after']['array_digests'] for k,v in branches.items()}
    first_cont=None
    for key in sorted(arrdig['E']):
        if not (arrdig['E'].get(key)==arrdig['C'].get(key)==arrdig['M'].get(key)):
            first_cont=key; break
    greedy={k:int(top10[k][0]['token']) for k in top10}
    stable=len(set(greedy.values()))==1
    cont_hard=scan_hardcoded()
    continuation_readiness={
      'generic_token_supported':False,
      'generic_position_supported':False,
      'token_history_commit_correct':False,
      'ngram_state_commit_correct':False,
      'window_kv_commit_correct':False,
      'pending_compressor_commit_correct':False,
      'compressed/index_publication_commit_correct':False,
      'engram_history_commit_correct':False,
      'hardcoded_first_token_assumptions':cont_hard,
      'audit_notes':[
        'decode_one rejects any token other than 15 and any state.position other than 2.',
        '_ngram_for_token15 maps token id 15 explicitly; generic continuation must derive compressed_token = source_token_map(tokenizer)[token_id].',
        'commit appends committed_tokens and increments position but does not append token_id to token_history, and returned ngram arrays are not installed into branch state.',
        'attention helpers use start_pos=2, freqs length 3, co_full/si_full[2:3], +3 compressed concat offset, and absolute_position=2 metadata.',
        'computed window KV, pending ratio2 compressor rows, compressed/index publications, top-k/candidates, and Engram/Ngram history are summarized but not committed into the branch NativeDecodeSessionState for token16 consumption.'
      ],
      'classification':'FIRST_INCREMENTAL_ONLY_CONTINUATION_GENERALIZATION_REQUIRED'
    }
    final_class='THREE_TRAJECTORY_BEHAVIORAL_STABILITY_INCOMPLETE_CONTINUATION_PENDING' if stable else 'BEHAVIOR_SENSITIVE_TO_FP8_REDUCTION_TRAJECTORY'
    rec={
      'schema':'ds41f.m4.reduction_trajectory_behavioral_stability.v2',
      'status':'FIRST_TOKEN_LOGITS_COMPLETE_CONTINUATION_PENDING',
      'cuda_oracle_import':{'json':str(CUDA_JSON),'output_npy':str(CUDA_NPY),'fixture_sha256':cuda['fixture']['npz_sha256'],'cuda_output_digest':sha(C),'verified':True},
      'trajectory_identities':{'E':{'name':'source-derived Block0 Attention output','digest':sha(E)},'C':{'name':'official CUDA/TileLang output','digest':sha(C)},'M':{'name':'production MLX M=1 output','digest':sha(M)}},
      'branch_state_independence':independence,
      'normal_E_vs_injected_E_identity':e_guard,
      'full_logits':{k:{'digest':arr_digest(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in logits.items()},
      'logits_pairwise_metrics':pair_metrics,
      'greedy_tokens':greedy,
      'rank2_tokens':{k:int(top10[k][1]['token']) for k in top10},
      'top1_top2_margins':{k:margin(top10[k]) for k in top10},
      'top10_lists':top10,
      'top32_lists':top32,
      'top32_overlap':top_overlap,
      'first_discrete_divergence':first_discrete,
      'discrete_decisions_compared':discrete,
      'first_persistent_state_divergence':{'continuous_numeric':first_cont,'semantic_discrete':first_sem},
      'persistent_state_after':{k:v['state_summary_after'] for k,v in branches.items()},
      'bounded_continuation':{'executed':False,'length':0,'token_sequences':{'E':[],'C':[],'M':[]},'reason':'decode_one is explicitly token15/position2 first-incremental-only and does not commit all state required for token16 consumption.'},
      'first_token_behavioral_stability_classification':'FIRST_TOKEN_BEHAVIOR_STABLE_ACROSS_REDUCTION_TRAJECTORIES' if stable else 'FIRST_TOKEN_BEHAVIOR_SENSITIVE_TO_VALID_REDUCTION_TRAJECTORY',
      'continuation_readiness':continuation_readiness,
      'final_classification':final_class,
      'precision_policy_decision':'First-token greedy behavior is stable across E/C/M, but THREE_TRAJECTORY_BEHAVIORAL_STABILITY remains incomplete until generic multi-token continuation is implemented and qualified.' if stable else 'First-token greedy behavior differs; behavior is sensitive to valid FP8 reduction trajectory.',
      'next_frontier':'generalize and qualify decode_one lifecycle for arbitrary next token/position, including real branch-state commits, before bounded continuation.',
      'ok':stable}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(_jsonable(rec),indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'classification':rec['first_token_behavioral_stability_classification'],'final':final_class,'digests':rec['full_logits'],'greedy_tokens':greedy,'first_discrete_divergence':first_discrete,'persistent':rec['first_persistent_state_divergence']},indent=2))
if __name__=='__main__': main()
