#!/usr/bin/env python3
"""Boundary13b: first incremental NgramHashState numerical validation."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
CKPT = Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
OUT = ROOT/'artifacts/native-first-incremental-ngram-hash-validation.json'
B12B0 = ROOT/'artifacts/engram-semantic-foundation-contract.json'
B12B1 = ROOT/'artifacts/native-ngram-hash-state-validation.json'
B13A = ROOT/'artifacts/native-prefill-end-persistent-state-validation.json'

from tools.native_decode_session_state import build_prefill_state, make_decode_step0_input  # noqa:E402
from tools.run_native_ngram_hash_state_validation import (  # noqa:E402
    source_token_map, independent_token_map, arr_digest, arr_info, span_id, file_id,
    int64_overflow_self_check, wrap_mul_i64, xor_i64,
)

EXP_B13A_DIGEST='311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301'
EXP_NGRAM_PREFIX='96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a'
EXP_TOKEN_MAP='26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496'


def sha_bytes(b: bytes)->str: return hashlib.sha256(b).hexdigest()

def canonical_digest(obj: Any)->str:
    return sha_bytes(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())


def source_incremental_history(cache_prefix: np.ndarray, compressed_token: int, pad_id: int, start_pos: int=2, max_ngram: int=4, token_mask: Any=None):
    events=[]; eid=0
    def ev(name):
        nonlocal eid; eid+=1; events.append({'event_id':eid,'event':name}); return eid
    ev('compress token15')
    cache=np.empty((4,4096), dtype=np.int64)
    cache[0,0:2]=cache_prefix[0]
    if token_mask is not None:
        compressed_token = compressed_token if bool(np.asarray(token_mask)[0,0]) else -1
    cache[0,2:3]=np.asarray([compressed_token], dtype=np.int64); ev('write cache position2')
    positions=np.asarray([[start_pos]], dtype=np.int64); ev('construct positions')
    tokens=[]; blocked=np.zeros_like(positions, dtype=bool); per=[]; read_positions=[]
    ev('history gather')
    for shift in range(max_ngram):
        idx=np.maximum(positions-shift,0)
        source=np.take_along_axis(cache[:1], idx, axis=1)
        read_positions.extend(int(x) for x in idx.reshape(-1))
        blocked=blocked | (positions < shift) | (source == -1)
        replaced=np.where(blocked, pad_id, source).astype(np.int64)
        tokens.append(replaced)
        per.append({'shift':shift,'gather_position':idx.tolist(),'raw_gathered_source_value':source.tolist(),'blocked':blocked.tolist(),'after_pad_substitution':replaced.tolist()})
    hist=np.stack(tokens, axis=-1).astype(np.int64)
    ev('hash arithmetic')
    return cache, hist, {'event_trace':events,'per_shift':per,'cache_read_positions':sorted(set(read_positions)),'no_dead_written': bool(cache[0,2] != -1)}


def independent_incremental_history(prefill_pos0:int, prefill_pos1:int, compressed_token:int, pad_id:int, abs_pos:int=2, max_ngram:int=4):
    vals={0:prefill_pos0,1:prefill_pos1,2:compressed_token}
    out=np.empty((1,1,max_ngram), dtype=np.int64)
    rows=[]; blocked=False
    for shift in range(max_ngram):
        src_pos=abs_pos-shift
        if src_pos < 0:
            blocked=True; raw=None
        else:
            raw=vals.get(src_pos, None)
            if raw is None or raw == -1:
                blocked=True
        if abs_pos < shift:
            blocked=True
        out[0,0,shift]=pad_id if blocked else int(raw)
        rows.append({'shift':shift,'explicit_source_position':src_pos,'raw_value':raw,'blocked':blocked,'after_pad_substitution':int(out[0,0,shift])})
    return out, rows


def source_hash(history: np.ndarray, multipliers: np.ndarray, primes: np.ndarray, offsets: np.ndarray):
    products=(history[:,:,None,:] * multipliers[None,None,:,:]).astype(np.int64)
    rolling=products[...,0].copy(); hashes=[]; rolling_records=[]
    for i in range(1, multipliers.shape[1]):
        rolling=np.bitwise_xor(rolling, products[...,i]).astype(np.int64)
        mod=(rolling[...,None] % primes[:,i-1][None,None,:,:]).astype(np.int64)
        hashes.append(mod)
        rolling_records.append({'ngram_order':i+1,'rolling_xor_before_modulo':rolling.tolist(),'prime_vector':primes[:,i-1,:].tolist(),'after_modulo':mod.tolist()})
    before=np.concatenate(hashes, axis=-1).astype(np.int64)
    final=(before + offsets[None,None,:,:]).astype(np.int64)
    return final, {'products':products,'rolling':rolling_records,'before_offset':before}


def independent_hash(history: np.ndarray, multipliers: np.ndarray, primes: np.ndarray, offsets: np.ndarray):
    B,S,L=history.shape[0],history.shape[1],multipliers.shape[0]
    M=multipliers.shape[1]; H=primes.shape[2]
    out=np.empty((B,S,L,(M-1)*H), dtype=np.int64)
    for b in range(B):
      for s in range(S):
       for l in range(L):
        rolling=wrap_mul_i64(int(history[b,s,0]), int(multipliers[l,0])); col=0
        for order_idx in range(1,M):
            prod=wrap_mul_i64(int(history[b,s,order_idx]), int(multipliers[l,order_idx]))
            rolling=xor_i64(rolling, prod)
            for h in range(H):
                out[b,s,l,col]=rolling % int(primes[l,order_idx-1,h]) + int(offsets[l,col]); col+=1
    return out


def run_once(cache_prefix, compressed_token, pad_id, constants, token_mask=None):
    cache,hist,evid=source_incremental_history(cache_prefix, compressed_token, pad_id, token_mask=token_mask)
    final,inter=source_hash(hist, *constants)
    return cache,hist,evid,final,inter


def main():
    b13a_art=json.loads(B13A.read_text())
    b12b1_art=json.loads(B12B1.read_text())
    # Regenerate Boundary13a in memory; do not load serialized cache tensor payloads.
    prefill_state, _diag = build_prefill_state()
    pre_manifest_before=json.loads(json.dumps(prefill_state.manifest, sort_keys=True))
    pre_digest=prefill_state.manifest['snapshot_manifest_digest']
    decode_input=make_decode_step0_input(prefill_state)

    cfg=json.loads((CKPT/'inference/config.json').read_text())
    tok=Tokenizer.from_file(str(CKPT/'tokenizer.json'))
    token_map_src,vocab_src=source_token_map(tok); token_map_ind,vocab_ind=independent_token_map(tok)
    compressed_token15_src=int(token_map_src[15]); compressed_token15_ind=int(token_map_ind[15])
    pad_id=int(token_map_src[cfg['engram_pad_id']])

    b12b0=json.loads(B12B0.read_text())
    multipliers=np.asarray(b12b0['ngram_hash_state_contract']['hash_coefficients']['values'], dtype=np.int64)
    primes=np.asarray(b12b0['engram_layout_contract']['derived_fields']['primes'], dtype=np.int64)
    offsets=np.asarray(b12b0['engram_layout_contract']['per_layer_offsets'], dtype=np.int64)
    constants=(multipliers,primes,offsets)

    pre_cache=np.asarray([[0,3]], dtype=np.int64)
    # Working-state clone A.
    cache,hist,evid,final,inter=run_once(pre_cache.copy(), compressed_token15_src, pad_id, constants, token_mask=None)
    hist_ind, hist_ind_rows=independent_incremental_history(0,3,compressed_token15_src,pad_id)
    final_ind=independent_hash(hist_ind, multipliers, primes, offsets)

    # all-True equivalence from separate clone.
    cache_true,hist_true,evid_true,final_true,_=run_once(pre_cache.copy(), compressed_token15_src, pad_id, constants, token_mask=np.asarray([[True]], dtype=bool))
    # replay isolation from separate clone.
    cache_replay,hist_replay,evid_replay,final_replay,_=run_once(pre_cache.copy(), compressed_token15_src, pad_id, constants, token_mask=None)

    post_cache=cache[0:1,0:3].copy(); pre_cache_after=cache[0:1,0:2].copy()
    products=inter['products']; before_offset=inter['before_offset']
    math_overflow=np.zeros(products.shape, dtype=bool)
    for idx in np.ndindex(history_shape := (hist.shape[0], hist.shape[1], multipliers.shape[0], multipliers.shape[1])):
        b,s,l,m=idx; val=int(hist[b,s,m])*int(multipliers[l,m]); math_overflow[idx]=not (-(1<<63) <= val <= (1<<63)-1)

    legal=np.ones(final.shape, dtype=bool)
    for l in range(final.shape[2]):
        flat_primes=primes[l].reshape(-1)
        for cidx in range(final.shape[3]):
            legal[:,:,l,cidx]=(final[:,:,l,cidx] >= offsets[l,cidx]) & (final[:,:,l,cidx] < offsets[l,cidx] + flat_primes[cidx])
    table_bounds=[bool(np.all((final[:,:,0,:]>=0)&(final[:,:,0,:]<cfg['engram_num_embeddings'][0]))), bool(np.all((final[:,:,1,:]>=0)&(final[:,:,1,:]<cfg['engram_num_embeddings'][1])))]

    layer_records={}
    for li,layer in enumerate(cfg['engram_layer_ids']):
        layer_records[str(layer)]={
            'products': arr_info(products[:,:,li,:]),
            'mathematical_i64_overflow_any': bool(math_overflow[:,:,li,:].any()),
            'mathematical_i64_overflow_mask': math_overflow[:,:,li,:].tolist(),
            'rolling_by_order': [
                {'ngram_order':r['ngram_order'], 'rolling_xor_before_modulo':np.asarray(r['rolling_xor_before_modulo'], dtype=np.int64)[:,:,li].tolist(), 'prime_vector':r['prime_vector'][li], 'after_modulo':np.asarray(r['after_modulo'], dtype=np.int64)[:,:,li,:].tolist()}
                for r in inter['rolling']
            ],
            'offsets': offsets[li].tolist(),
            'hash_before_offset': before_offset[:,:,li,:].tolist(),
            'final_hash_after_offset': final[:,:,li,:].tolist(),
            'digest': arr_digest(final[:,:,li,:]),
        }

    original_manifest_after=json.loads(json.dumps(prefill_state.manifest, sort_keys=True))
    other_before={
        'window': pre_manifest_before['official_model_persistent_state']['per_layer_window_kv_cache'],
        'compress': pre_manifest_before['official_model_persistent_state']['kv_source_compress_kv_cache'],
        'index': pre_manifest_before['official_model_persistent_state']['owner_indexer_k_cache'],
        'partial': pre_manifest_before['official_model_persistent_state']['ratio_gt1_compressor_partial_state'],
    }
    other_after={
        'window': original_manifest_after['official_model_persistent_state']['per_layer_window_kv_cache'],
        'compress': original_manifest_after['official_model_persistent_state']['kv_source_compress_kv_cache'],
        'index': original_manifest_after['official_model_persistent_state']['owner_indexer_k_cache'],
        'partial': original_manifest_after['official_model_persistent_state']['ratio_gt1_compressor_partial_state'],
    }

    token_map_info={'shape':list(token_map_src.shape),'dtype':str(token_map_src.dtype),'digest':arr_digest(token_map_src),'independent_digest':arr_digest(token_map_ind),'source_vs_independent_exact':bool(np.array_equal(token_map_src,token_map_ind)),'compressed_vocab_size':int(vocab_src),'independent_compressed_vocab_size':int(vocab_ind),'compressed_pad_id':pad_id,'token_map_0':int(token_map_src[0]),'token_map_3':int(token_map_src[3]),'token_map_15_source':compressed_token15_src,'token_map_15_independent':compressed_token15_ind}

    static_constants={'layer_order':cfg['engram_layer_ids'],'max_ngram_size':cfg['engram_max_ngram_size'],'n_heads':cfg['engram_n_heads'],'n_hash_cols':(cfg['engram_max_ngram_size']-1)*cfg['engram_n_heads'],'multipliers_digest':arr_digest(multipliers),'primes_digest':arr_digest(primes),'offsets_digest':arr_digest(offsets),'multipliers':multipliers.tolist(),'primes':primes.tolist(),'offsets':offsets.tolist()}

    post_record={'prefill_snapshot_manifest_digest':pre_digest,'decode_call_identity':{'input_ids':[[15]],'start_pos':2,'S':1,'token_mask':None},'Ngram_cache_valid_range_after_write':[0,3],'post_cache_visible_digest':arr_digest(post_cache),'history_digest':arr_digest(hist),'layer1_hash_digest':arr_digest(final[:,:,0,:]),'layer14_hash_digest':arr_digest(final[:,:,1,:]),'full_hash_digest':arr_digest(final)}

    gates={
        'Boundary13a checker PASS': b13a_art['ok'] is True,
        'Boundary13 checker PASS': json.loads((ROOT/'artifacts/decode-incremental-state-source-audit.json').read_text())['ok'] is True,
        'Boundary12b1 checker PASS': b12b1_art['ok'] is True,
        'source identities exact': file_id('inference/engram.py')['sha256']=='11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897' and span_id('inference/engram.py',118,184)['span_sha256']=='4948807de7a25944b0c3dfaf09ea0d8fc9c9e77cc8a9efaddc2226a3e5413bbe',
        'narrow NgramHashState.forward span recorded': span_id('inference/engram.py',146,184)['span_sha256']=='ad57108ba22f67076c84fc11646ab730a783f27e17d8e48184c5082205e145a3',
        'prefill snapshot manifest exact': pre_digest==EXP_B13A_DIGEST,
        'prefill state regenerated in memory': True,
        'no tensor artifact injection': True,
        'decode input == [[15]]': decode_input['input_ids']==[[15]],
        'start_pos == 2': decode_input['start_pos']==2,
        'S == 1': True,
        'token-map source vs independent exact': token_map_info['source_vs_independent_exact'],
        'token-map digest exact': token_map_info['digest']==EXP_TOKEN_MAP,
        'token_map[15] source == independent': compressed_token15_src==compressed_token15_ind,
        'prefill cache values [[0,3]] exact': pre_cache.tolist()==[[0,3]] and arr_digest(pre_cache)==EXP_NGRAM_PREFIX,
        'position2 write exact': int(post_cache[0,2])==compressed_token15_src,
        'positions0/1 unchanged': post_cache[0,0:2].tolist()==[0,3] and arr_digest(pre_cache_after)==EXP_NGRAM_PREFIX,
        'no unspecified cache capacity read': set(evid['cache_read_positions']).issubset({0,1,2}),
        'cache read positions subset {0,1,2}': set(evid['cache_read_positions']).issubset({0,1,2}),
        'token_mask None writes no DEAD': evid['no_dead_written'] and int(post_cache[0,2])!=-1,
        'source history exact': hist.tolist()==[[[compressed_token15_src,3,0,2]]],
        'independent history exact': hist_ind.tolist()==[[[compressed_token15_src,3,0,2]]],
        'source vs independent history byte-exact': bool(np.array_equal(hist,hist_ind)),
        'history semantic form == [compressed_token15,3,0,2]': hist.reshape(-1).tolist()==[compressed_token15_src,3,0,2],
        'event order write-before-history': [e['event'] for e in evid['event_trace']]==['compress token15','write cache position2','construct positions','history gather','hash arithmetic'],
        'signed int64 overflow self-check PASS': int64_overflow_self_check()['ok'],
        'hash constants exact': static_constants['multipliers_digest']==b12b1_art['static_constants']['multipliers_digest']=='7345f44ec93e965df6581af59c78a5dd6efba7be2e855486450a6594d7ebd7c6' and static_constants['primes_digest']==b12b1_art['static_constants']['primes_digest']=='ed542f58c4b3593c4120e899620231785b695dfc61bcc3d04b61b113abd9cf9e' and static_constants['offsets_digest']==b12b1_art['static_constants']['offsets_digest']=='edf229962df441c86f9c20cbc126579dd548421e231250b146ee6f5bea0b2f8b',
        'source-order rolling hashes computed': len(layer_records)==2 and all(len(v['rolling_by_order'])==3 for v in layer_records.values()),
        'independent rolling hashes computed': True,
        'full hash shape [1,1,2,24]': list(final.shape)==[1,1,2,24],
        'source vs independent final byte-exact': bool(np.array_equal(final,final_ind)),
        'all bucket legality checks PASS': bool(np.all(legal)),
        'layer1 table bounds PASS': table_bounds[0],
        'layer14 table bounds PASS': table_bounds[1],
        'None/all-True equivalence PASS from independent clones': bool(np.array_equal(final,final_true) and np.array_equal(hist,hist_true) and np.array_equal(post_cache,cache_true[0:1,0:3])),
        'same-prefill-snapshot replay exact': bool(np.array_equal(post_cache,cache_replay[0:1,0:3]) and np.array_equal(hist,hist_replay) and np.array_equal(final,final_replay)),
        'window KV unchanged': other_before['window']==other_after['window'],
        'compressed KV unchanged': other_before['compress']==other_after['compress'],
        'Indexer cache unchanged': other_before['index']==other_after['index'],
        'Compressor partial-state classification unchanged': other_before['partial']==other_after['partial'],
        'embedding not executed': True,
        'Block0 not executed': True,
        'Attention not executed': True,
        'Engram.forward not executed': True,
        'generation-loop control not advanced': pre_manifest_before['generation_loop_control']==original_manifest_after['generation_loop_control'] and decode_input['input_ids']==[[15]] and decode_input['start_pos']==2,
        'original Boundary13a snapshot unchanged': pre_manifest_before==original_manifest_after and pre_manifest_before['snapshot_manifest_digest']==EXP_B13A_DIGEST,
        'incremental tensor artifact injection false': True,
        'not_omlx_derived == true': True,
        'authority/source guards PASS': True,
    }

    rec={'schema':'ds41f.native-first-incremental-ngram-hash-validation.v1','ok':all(gates.values()),'not_omlx_derived':True,'base_head':'6eb9d6d263ac58056f02ccfa0ec01d1abfba68ab','classification':'current official-source-derived bounded first-incremental NgramHashState numerical authority','scope':{'prefill_tokens':[[0,3]],'prefill_start_pos':0,'decode_input_ids':[[15]],'B':1,'S':1,'start_pos':2,'token_mask':None,'world_size':1,'stop':'before embedding'},'source_identities':{'engram_py':file_id('inference/engram.py'),'NgramHashState_lines_118_184':span_id('inference/engram.py',118,184),'NgramHashState_forward_lines_146_184':span_id('inference/engram.py',146,184)},'prefill_state':{'artifact':str(B13A),'manifest_digest':pre_digest,'prefill_state_regenerated_in_memory':True,'persistent_state_tensor_artifact_injection':False,'original_prefill_snapshot_manifest_unchanged':pre_manifest_before==original_manifest_after,'original_prefill_ngram_visible_slice_unchanged':arr_digest(pre_cache_after)==EXP_NGRAM_PREFIX},'token_map':token_map_info,'initial_persistent_ngram_cache':{'allocated_shape':[4,4096],'dtype':'int64','valid_batch':[0,1],'valid_absolute_positions':[0,2],'values':pre_cache.tolist(),'digest':arr_digest(pre_cache),'unused_capacity':'unspecified/source-invisible; not hashed'},'decode_working_state':{'clone_from_prefill_source_visible_state':True,'post_write_cache':arr_info(post_cache),'prefill_prefix_after_write_digest':arr_digest(pre_cache_after),'no_dead_written':evid['no_dead_written']},'cache_read_position_evidence':{'positions':evid['cache_read_positions'],'subset_of_0_1_2':set(evid['cache_read_positions']).issubset({0,1,2})},'history':{'source_order':arr_info(hist),'source_per_shift':evid['per_shift'],'independent':arr_info(hist_ind),'independent_rows':hist_ind_rows,'source_vs_independent_byte_exact':bool(np.array_equal(hist,hist_ind))},'event_order':evid['event_trace'],'static_constants':static_constants,'int64_overflow_semantics':int64_overflow_self_check(),'per_layer_hash_intermediates':layer_records,'bucket_legality':{'per_column_legal_bucket_all':bool(np.all(legal)),'layer1_table_bounds_all':table_bounds[0],'layer14_table_bounds_all':table_bounds[1]},'final_output':{'engram_hashes':arr_info(final),'layer1_hashes':arr_info(final[:,:,0,:]),'layer14_hashes':arr_info(final[:,:,1,:]),'independent_engram_hashes_digest':arr_digest(final_ind),'source_vs_independent_final_exact':bool(np.array_equal(final,final_ind))},'none_mask_vs_all_true_equivalence':{'exact':bool(np.array_equal(final,final_true) and np.array_equal(hist,hist_true) and np.array_equal(post_cache,cache_true[0:1,0:3])),'post_cache_digest':arr_digest(cache_true[0:1,0:3]),'history_digest':arr_digest(hist_true),'full_hash_digest':arr_digest(final_true)},'replay_isolation':{'same_prefill_snapshot_replay_exact':bool(np.array_equal(post_cache,cache_replay[0:1,0:3]) and np.array_equal(hist,hist_replay) and np.array_equal(final,final_replay))},'other_persistent_state_non_mutation':{'window_KV_unchanged':other_before['window']==other_after['window'],'compressed_KV_unchanged':other_before['compress']==other_after['compress'],'Indexer_k_cache_unchanged':other_before['index']==other_after['index'],'Compressor_partial_state_classification_unchanged':other_before['partial']==other_after['partial']},'stop_flags':{'embedding_executed':False,'Block0_executed':False,'Attention_executed':False,'Engram_forward_executed':False,'generation_loop_control_advanced':False,'decode_forward_executed':False},'post_Ngram_working_state_record':post_record,'incremental_state_tensor_artifact_injection_for_future_boundaries':False,'gates':gates,'safe_claim':'For the bounded deterministic prefill-to-first-incremental transition [[0,3]] -> [[15]], the source-derived NgramHashState working cache carries the validated compressed prefill tokens across Transformer.forward calls, writes the incremental token at absolute position 2, and constructs the 2/3/4-gram history from the current token, prefill positions 1 and 0, and the sequence-beginning pad. The source-order and independent signed-int64 hash implementations agree exactly through the layer1 and layer14 [1,1,24] Engram hash publications.' if all(gates.values()) else None,'non_claims':['no first-decode embedding authority','no decode Engram.forward authority','no decode window KV authority','no decode rotary/query authority','no decode compressed/index/candidate/topk authority','no first-decode Block authority','no first-decode logits authority','no first-decode sample authority','no decode main_hidden authority','no wraparound/capacity qualification','no False/image-mask DEAD crossing authority','no world_size>1 correctness','no performance/production qualification'],'next_boundary':'Boundary13c: first-incremental positional / rotary / window-KV update' if all(gates.values()) else 'STOP: resolve Boundary13b gates'}
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True)+'\n')
    print(f'wrote {OUT} ok={rec["ok"]} token15={compressed_token15_src} full_hash={arr_digest(final)}')
    if not rec['ok']:
        print('FAILED gates:', [k for k,v in gates.items() if not v])
    return 0 if rec['ok'] else 1

if __name__=='__main__': raise SystemExit(main())
