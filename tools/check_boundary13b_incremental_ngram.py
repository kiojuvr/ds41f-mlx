#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-first-incremental-ngram-hash-validation.json'
DOC=ROOT/'docs/first-incremental-ngram-hash-validation.md'
ALLOWED_DIRTY={
 'tools/run_native_first_incremental_ngram_hash_validation.py',
 'tools/check_boundary13b_incremental_ngram.py',
 'docs/first-incremental-ngram-hash-validation.md',
 'artifacts/native-first-incremental-ngram-hash-validation.json',
}

def dirty_paths():
    return {line[3:] for line in subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()}

def main():
    assert ART.exists(), ART
    assert DOC.exists(), DOC
    dirty=dirty_paths()
    if not dirty:
        assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary13a_prefill_state.py')],cwd=ROOT).returncode==0
        assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary13_decode_state_audit.py')],cwd=ROOT).returncode==0
        assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b1_ngram_hash_state.py')],cwd=ROOT).returncode==0
    else:
        assert dirty <= ALLOWED_DIRTY, dirty
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-first-incremental-ngram-hash-validation.v1'
    assert r['ok'] is True and r['not_omlx_derived'] is True
    assert r['base_head']=='6eb9d6d263ac58056f02ccfa0ec01d1abfba68ab'
    assert r['classification']=='current official-source-derived bounded first-incremental NgramHashState numerical authority'
    sc=r['scope']
    assert sc['prefill_tokens']==[[0,3]] and sc['decode_input_ids']==[[15]]
    assert sc['start_pos']==2 and sc['S']==1 and sc['token_mask'] is None and sc['stop']=='before embedding'
    sid=r['source_identities']
    assert sid['engram_py']['sha256']=='11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897'
    assert sid['NgramHashState_lines_118_184']['span_sha256']=='4948807de7a25944b0c3dfaf09ea0d8fc9c9e77cc8a9efaddc2226a3e5413bbe'
    assert sid['NgramHashState_forward_lines_146_184']['span_sha256']=='ad57108ba22f67076c84fc11646ab730a783f27e17d8e48184c5082205e145a3'
    pf=r['prefill_state']
    assert pf['manifest_digest']=='311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301'
    assert pf['prefill_state_regenerated_in_memory'] is True
    assert pf['persistent_state_tensor_artifact_injection'] is False
    assert pf['original_prefill_snapshot_manifest_unchanged'] is True
    assert pf['original_prefill_ngram_visible_slice_unchanged'] is True
    tm=r['token_map']
    assert tm['shape']==[129280] and tm['dtype']=='int64'
    assert tm['digest']=='26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496'
    assert tm['source_vs_independent_exact'] is True
    assert tm['compressed_vocab_size']==99092 and tm['independent_compressed_vocab_size']==99092
    assert tm['compressed_pad_id']==2 and tm['token_map_0']==0 and tm['token_map_3']==3
    assert tm['token_map_15_source']==15 and tm['token_map_15_independent']==15
    init=r['initial_persistent_ngram_cache']
    assert init['values']==[[0,3]] and init['digest']=='96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a'
    post=r['decode_working_state']['post_write_cache']
    assert post['values']==[[0,3,15]] and post['digest']=='04a3a0772a3b03dd471d3ab889112d78bd7661e05aa2073c16017fa43198983c'
    assert r['decode_working_state']['prefill_prefix_after_write_digest']=='96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a'
    assert r['decode_working_state']['no_dead_written'] is True
    assert set(r['cache_read_position_evidence']['positions']) <= {0,1,2}
    assert r['cache_read_position_evidence']['subset_of_0_1_2'] is True
    hist=r['history']
    assert hist['source_order']['values']==[[[15,3,0,2]]]
    assert hist['source_order']['digest']=='cf5abf85fe2bdf166063c0b32e83f38c962d3b19bed561606d97f1271b8f378f'
    assert hist['independent']['values']==[[[15,3,0,2]]]
    assert hist['source_vs_independent_byte_exact'] is True
    per=hist['source_per_shift']
    assert [p['gather_position'][0][0] for p in per]==[2,1,0,0]
    assert [p['after_pad_substitution'][0][0] for p in per]==[15,3,0,2]
    assert per[3]['blocked']==[[True]]
    assert [e['event'] for e in r['event_order']]==['compress token15','write cache position2','construct positions','history gather','hash arithmetic']
    st=r['static_constants']
    assert st['layer_order']==[1,14] and st['max_ngram_size']==4 and st['n_heads']==8 and st['n_hash_cols']==24
    assert st['multipliers_digest']=='7345f44ec93e965df6581af59c78a5dd6efba7be2e855486450a6594d7ebd7c6'
    assert st['primes_digest']=='ed542f58c4b3593c4120e899620231785b695dfc61bcc3d04b61b113abd9cf9e'
    assert st['offsets_digest']=='edf229962df441c86f9c20cbc126579dd548421e231250b146ee6f5bea0b2f8b'
    assert r['int64_overflow_semantics']['ok'] is True
    out=r['final_output']
    assert out['engram_hashes']['shape']==[1,1,2,24] and out['engram_hashes']['dtype']=='int64'
    assert out['engram_hashes']['digest']=='09c32d336e7a23d61ff9ac94674cb30039857eeeb3df82cf475157685d76c530'
    assert out['layer1_hashes']['digest']=='4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373'
    assert out['layer14_hashes']['digest']=='5d93f09bfecb5b8a3a722603bb5e1cd44710849cc4729223df92953386bf3ad7'
    assert out['source_vs_independent_final_exact'] is True
    assert r['bucket_legality']['per_column_legal_bucket_all'] is True
    assert r['bucket_legality']['layer1_table_bounds_all'] is True and r['bucket_legality']['layer14_table_bounds_all'] is True
    assert r['none_mask_vs_all_true_equivalence']['exact'] is True
    assert r['replay_isolation']['same_prefill_snapshot_replay_exact'] is True
    om=r['other_persistent_state_non_mutation']
    assert all(om.values())
    sf=r['stop_flags']
    assert sf=={'Attention_executed':False,'Block0_executed':False,'Engram_forward_executed':False,'decode_forward_executed':False,'embedding_executed':False,'generation_loop_control_advanced':False}
    assert r['incremental_state_tensor_artifact_injection_for_future_boundaries'] is False
    assert all(r['gates'].values()), [k for k,v in r['gates'].items() if not v]
    doc=DOC.read_text()
    for phrase in ['Boundary13b', 'token_map[15]: 15', 'cache[0,0:3] = [[0,3,15]]', 'Full output', 'STOP flags', 'No first-decode']:
        assert phrase in doc, phrase
    print(f'Boundary13b first incremental NgramHashState check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
