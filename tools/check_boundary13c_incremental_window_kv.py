#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-first-incremental-window-kv-rotary-validation.json'
DOC=ROOT/'docs/first-incremental-window-kv-rotary-validation.md'
ALLOWED={
 'tools/run_native_first_incremental_window_kv_rotary_validation.py',
 'tools/check_boundary13c_incremental_window_kv.py',
 'docs/first-incremental-window-kv-rotary-validation.md',
 'artifacts/native-first-incremental-window-kv-rotary-validation.json',
}

def dirty():
    return {line[3:] for line in subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()}

def main():
    assert ART.exists() and DOC.exists()
    d=dirty()
    if not d:
        for tool in ['check_boundary13_decode_state_audit.py','check_boundary13a_prefill_state.py','check_boundary13b_incremental_ngram.py']:
            assert subprocess.run([sys.executable,str(ROOT/'tools'/tool)],cwd=ROOT).returncode==0, tool
    else:
        assert d <= ALLOWED, d
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-first-incremental-window-kv-rotary-validation.v1'
    assert r['ok'] is True and r['not_omlx_derived'] is True
    assert r['base_head']=='850c14139195f17908132fe988722c25bca38936'
    assert r['Boundary13a_manifest_digest']=='311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301'
    h=r['Boundary13b_handoff']
    assert h['post_ngram_cache_digest']=='04a3a0772a3b03dd471d3ab889112d78bd7661e05aa2073c16017fa43198983c'
    assert h['full_incremental_hash_digest']=='09c32d336e7a23d61ff9ac94674cb30039857eeeb3df82cf475157685d76c530'
    assert h['layer1_hash_digest']=='4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373'
    assert h['layer14_hash_digest']=='5d93f09bfecb5b8a3a722603bb5e1cd44710849cc4729223df92953386bf3ad7'
    assert h['recomputed_in_memory'] is True and h['tensor_artifact_injection'] is False
    dc=r['decode_call']; assert dc=={'B':1,'S':1,'input_ids':[[15]],'start_pos':2,'token_mask':None,'world_size':1}
    sid=r['source_identities']
    assert sid['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
    assert sid['Transformer_forward']['span_sha256']=='b63c6a5afb2a8b5a91e249e08c6272a94c6772a0dc8df9095261c6fda751bbeb'
    assert sid['Block_forward_attention_prelude']['span_sha256']=='ecd6c573295c5c4dbf30c2b627c418e50fec7d1ea69d30cec5a86d32f1dd8295'
    assert sid['Attention_forward_q_window']['span_sha256']=='3db617863c8838676e1aa57cbb6ddd6fc182ad588e9765ec402d8cdf9ed32259'
    assert sid['Attention_window_kv']['span_sha256']=='1361c5743bb67565029539d01668c6071e35d819321907da6c93d180e7f6f664'
    assert sid['apply_rotary_emb']['span_sha256']=='b29090f12c23f30254d4f0f5b3f966a61c1d1e7e5f590b52b1a2777bbcab5a61'
    assert sid['get_window_topk_idxs']['span_sha256']=='6f29e174995e5f5d3f194b335d17d68a8f65fdca5ed35c2f4ec9576f2099e03f'
    te=r['Transformer_entry']
    assert te['embedding']['digest']=='75fc2a390b761925546540151bdf4735e79289d75e8a4d234d1ac0dc2ae7ba06'
    assert te['prefill_h_pre_mix_main_hidden_reused'] is False
    assert r['Block0_attention_prelude']['attention_input']['digest']=='9f3d5433e680798d5f32fe10b8384476664078668b234c0780c7bce20ef775be'
    pos=r['absolute_rotary_position_evidence']
    assert pos['start_pos']==2 and pos['end_pos']==3 and pos['S']==1 and pos['freqs_slice']=='freqs_cis[2:3]'
    assert pos['cos_pos2_digest']=='08014ebec49df6b660eef9154f73ef63d91cb5d75f860f64523aecd0289e0a45'
    assert pos['sin_pos2_digest']=='1665e30c199582ec502e76320ff61d1c628212766584b38afd060cf2585fe62f'
    assert pos['q_position0_control_differs'] and pos['kv_position0_control_differs']
    q=r['layer0_Q_path']; kv=r['layer0_KV_path']
    assert q['post_rotary']['digest']=='7ba5035746cd43e25016dd0069c2bb5b4bc64ccb229e35e89741e8d88cd6cab9'
    assert q['independent_post_rotary_digest']==q['post_rotary']['digest']
    assert q['position0_control_digest']!=q['post_rotary']['digest']
    assert kv['post_rotary']['digest']=='eb0d334e615e729d6e9e5765e1ca352f5072067048dcfc32f6384484b4f3afd1'
    assert kv['independent_post_rotary_digest']==kv['post_rotary']['digest']
    assert kv['position0_control_digest']!=kv['post_rotary']['digest']
    assert kv['new_window_kv']['digest']=='636e9636e288fd3bac5e8cf5aa3095187f715dc0c71195d72283c086be00c12b'
    wc=r['window_cache_transition']
    assert wc['slot0_before']['digest']=='1c6a13138e16d4abcf19c1c93313ce8da7f41adac3a31035f83132f2b2edf524'
    assert wc['slot0_after']['digest']==wc['slot0_before']['digest']
    assert wc['slot1_before']['digest']=='5cdf50ce91d245204f50db483466d8aee56795efc1313da868c4ea4c4f204e30'
    assert wc['slot1_after']['digest']==wc['slot1_before']['digest']
    assert wc['slot2_after']['digest']=='636e9636e288fd3bac5e8cf5aa3095187f715dc0c71195d72283c086be00c12b'
    assert wc['post_visible_cache']['digest']=='9a39b3b26a0b352b067fd5fbac0487911478f335078041d84ce3ef47a309920a'
    assert wc['valid_absolute_positions']==[0,1,2] and wc['ring_slots']==[0,1,2]
    assert wc['unused_capacity_promoted'] is False
    top=r['window_topk']
    assert top['source']['shape']==[1,1,128] and top['source']['dtype']=='int32'
    assert top['source']['digest']=='fae2c9f0a421368ce686c681d9d7c1ec97be4c343b8cc03c92df7d3e70fb2bc0'
    assert top['source_vs_independent_exact'] is True
    assert top['valid_slots']==[0,1,2]
    assert top['source']['values'][0][0][-10:]==[-1,-1,-1,-1,-1,-1,-1,0,1,2]
    assert all(r['non_mutation_table'].values())
    sf=r['STOP_flags']; assert all(v is False for v in sf.values())
    assert r['tensor_artifact_injection'] is False
    assert all(r['gates'].values()), [k for k,v in r['gates'].items() if not v]
    doc=DOC.read_text()
    for phrase in ['Boundary13c', 'freqs_cis[2:3]', 'slot2/new window KV', 'STOP before sparse_attn', 'No compressed']:
        assert phrase in doc, phrase
    print(f'Boundary13c first incremental window-KV/rotary check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
