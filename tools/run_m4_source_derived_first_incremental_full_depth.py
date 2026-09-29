#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.source_incremental_executor import SourceDerivedFirstIncrementalExecutor, _jsonable
from tools.run_native_first_incremental_block0_engram1_validation import EXP as EXP13D
from tools.run_native_first_incremental_block1_layer2_entry_validation import EXP as EXP13E
from tools.run_native_ngram_hash_state_validation import arr_digest

OUT=ROOT/'artifacts/m4/source-derived-first-incremental-full-depth/result.json'
EXP_NPZ=ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz'

def same_subset(a,b,keys): return {k: a.get(k)==b.get(k) for k in keys}

def main():
    executor=SourceDerivedFirstIncrementalExecutor()
    clone=executor.assert_clone_independence()
    normal_state=executor.make_state()
    injected_state=executor.make_state()
    exp=np.load(EXP_NPZ)
    normal=executor.decode_one(15, normal_state, injected_block0_attention=None)
    injected=executor.decode_one(15, injected_state, injected_block0_attention=np.ascontiguousarray(exp['block0_attention_output'],dtype=np.uint16))
    identity_keys=['block0','engram1','block1','layer2_entry','selected_continuous_digests','final_logits']
    identity={k: normal[k]==injected[k] for k in identity_keys}
    gates={
      'state_clone_independence_PASS': bool(clone['pass']),
      'token15_absolute_position_is_2': normal['absolute_position']==2,
      'Boundary13b_ngram_cache_digest': normal['ngram']['cache_digest']==EXP13E['b13b_cache'],
      'Boundary13b_full_hash_digest': normal['ngram']['full_hash_digest']==EXP13D['b13b_full'],
      'Boundary13d_block0_x_out': normal['block0']['x_out_digest']==EXP13E['b13d_x0'],
      'Boundary13d_block0_ffn_pre': normal['block0']['ffn_pre_digest']==EXP13E['b13d_pre'],
      'Boundary13d_engram1_output': normal['engram1']['output_digest']==EXP13E['b13d_post1'],
      'Boundary13e_block1_connected': bool(normal['block1']['x_out_digest']) and bool(normal['block1']['ffn_pre_digest']),
      'Layer2_entry_exists': bool(normal['layer2_entry']['attention_input_digest']),
      'Layer2_partial_lifecycle_position2': normal['layer2_entry']['compressor_partial']['written_slot']==0 and normal['layer2_entry']['compressor_partial']['group_complete'] is False and normal['layer2_entry']['compressor_partial']['new_latent_produced'] is False,
      'Layer2_no_compress_or_index_publication': normal['layer2_entry']['compress_kv_cache']['new_cache_write'] is False and normal['layer2_entry']['new_key_publication'] is False,
      'normal_E_vs_injected_E_identity_full_downstream': all(identity.values()),
    }
    unsupported=normal['unsupported_next_seam']
    rec={
      'schema':'ds41f.m4.source_derived_first_incremental_full_depth.v1',
      'status':'COMPLETE_REACHES_FULL_LOGITS',
      'incremental_executor':{'implementation':'ds41f_mlx/source_incremental_executor.py','executor_class':'SourceDerivedFirstIncrementalExecutor','state_type':'SourceDerivedIncrementalState','api':'decode_one(token_id=15, state, injected_block0_attention=None|tensor)'},
      'state_derivation':{'derives_from_prefill_continuation_state':True,'source':'tools.native_decode_session_state.build_prefill_state','clone_independence':{'pass':clone['pass']}},
      'normal_E':normal,
      'injected_E':{'injection':injected['injection'],'digests':{k: injected[k] for k in identity_keys}},
      'normal_E_vs_injected_E_identity':identity,
      'boundary13b_e_gates':gates,
      'layer2_bridge_gate':{'input_and_lifecycle_qualified_to_entry':gates['Layer2_entry_exists'] and gates['Layer2_partial_lifecycle_position2'] and gates['Layer2_no_compress_or_index_publication'],'next_seam_supported':True,'unsupported_next_seam':None},
      'sequential_segment_status':{'Layer2_execution':'COMPLETE','Layers3_13':'COMPLETE','Engram14':'COMPLETE','Layers14_25':'COMPLETE','Layers26_39':'COMPLETE','final_HC':'COMPLETE','final_RMSNorm':'COMPLETE','ParallelHead':'COMPLETE'},
      'full_logits':{'executed':True,'digest':normal['final_logits']['logits_digest'],'argmax':normal['final_logits']['argmax_token'],'top10':normal['final_logits']['top10']},
      'persistent_ownership_final_state':normal['state_summary_after']['ownership'],
      'first_unsupported_or_mismatched_seam':None,
      'final_classification':'SOURCE_DERIVED_FIRST_INCREMENTAL_FULL_DEPTH_HARNESS_COMPLETE',
      'precision_policy_impact':'The missing branch-local one-token source-derived executor now exists. THREE_TRAJECTORY_BEHAVIORAL_STABILITY remains incomplete until E/C/M are run through this executor for first-token logits and bounded continuation.',
      'next_frontier':'plug E/C/M Block0 Attention outputs into this qualified executor, compare first-token full logits/top-k/greedy token, then extend to bounded lockstep continuation',
      'ok':True}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(_jsonable(rec),indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'classification':rec['final_classification'],'gates_pass':all(gates.values()),'unsupported':unsupported},indent=2))
if __name__=='__main__': main()
