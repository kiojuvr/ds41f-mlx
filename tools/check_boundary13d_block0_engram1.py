#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-first-incremental-block0-engram1-validation.json'
DOC=ROOT/'docs/first-incremental-block0-engram1-validation.md'
ALLOWED={
 'tools/run_native_first_incremental_block0_engram1_validation.py',
 'tools/check_boundary13d_block0_engram1.py',
 'docs/first-incremental-block0-engram1-validation.md',
 'artifacts/native-first-incremental-block0-engram1-validation.json',
}

def dirty():
    return {line[3:] for line in subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()}

def main():
    assert ART.exists() and DOC.exists()
    d=dirty()
    if not d:
        for tool in ['check_boundary13_decode_state_audit.py','check_boundary13a_prefill_state.py','check_boundary13b_incremental_ngram.py','check_boundary13c_incremental_window_kv.py']:
            assert subprocess.run([sys.executable,str(ROOT/'tools'/tool)],cwd=ROOT).returncode==0, tool
    else:
        assert d <= ALLOWED, d
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-first-incremental-block0-engram1-validation.v1'
    assert r['ok'] is True and r['not_omlx_derived'] is True
    assert r['base_head']=='27078843c63af010544a36b9d7a17ef4ba868e72'
    assert r['classification']=='current official-source-derived bounded first-incremental Block0 completion plus Engram@1 handoff authority'
    up=r['upstream_state_reconstruction']
    assert up['Boundary13a_manifest_digest']=='311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301'
    assert up['Boundary13b_recomputed_in_memory'] and up['Boundary13c_replayed_in_memory'] and up['tensor_artifact_injection'] is False
    h=r['handoff_regressions']
    assert h['Boundary13b_cache']=='04a3a0772a3b03dd471d3ab889112d78bd7661e05aa2073c16017fa43198983c'
    assert h['Boundary13b_full_hash']=='09c32d336e7a23d61ff9ac94674cb30039857eeeb3df82cf475157685d76c530'
    assert h['Boundary13b_layer1_hash']=='4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373'
    assert h['Boundary13c_q']=='7ba5035746cd43e25016dd0069c2bb5b4bc64ccb229e35e89741e8d88cd6cab9'
    assert h['Boundary13c_kv']=='eb0d334e615e729d6e9e5765e1ca352f5072067048dcfc32f6384484b4f3afd1'
    assert h['Boundary13c_topk']=='fae2c9f0a421368ce686c681d9d7c1ec97be4c343b8cc03c92df7d3e70fb2bc0'
    assert h['Boundary13c_window_visible']=='9a39b3b26a0b352b067fd5fbac0487911478f335078041d84ce3ef47a309920a'
    sid=r['source_identities']
    assert sid['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
    assert sid['kernel_py']['sha256']=='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
    sp=r['sparse_attn']
    assert sp['inputs']['q_digest']==h['Boundary13c_q']
    assert sp['inputs']['kv_visible_digest']==h['Boundary13c_window_visible']
    assert sp['inputs']['topk_digest']==h['Boundary13c_topk']
    assert sp['inputs']['attn_sink']['digest']=='8017bf7692016398f5b28fee5afe99d3f2f3a54cb68fc66d86a5b7a58c241457'
    assert abs(sp['inputs']['softmax_scale'] - 0.04419417306780815) < 1e-12
    assert sp['output']['digest']=='4cff07ee5c4455690cc6b2d2139d4c815c5d0ade0897eb759020bdec44bedb69'
    assert sp['minus_one_mask_present'] is True
    ao=r['attention_output_path']
    assert ao['inverse_rotary']['digest']=='a041e24df7c96e7ee4de7133ebe3019d91e221951d07ee4f6b2ade415d9752a7'
    assert ao['woa_out']['digest']=='23e1edca1ef7db213475b23113c09703efac93c851392b389ae1f222e3b62422'
    assert ao['attention_output']['digest']=='0814f25097ba620c760e3db144f55d25b0f9434d03d8edd291a7a81a19105add'
    b=r['Block0_completion']
    assert b['x_after_attn']['digest']=='9f230b41307ccff6fe28d99cbb8550860c726b2687830e14be3b768160eeb95f'
    assert b['moe_input']['digest']=='7510cf767ee815ddf37f15c2b48eaed1c479cdca788721a36880017eb637bb28'
    assert b['moe_output']['digest']=='ea6fefed4813adbefb8aba2e49b54a0fc1847a06ae45ea0f6b1cde0efe654ab7'
    assert b['x_out']['digest']=='ecbf76fb8c272dc20399cf4e9dc839e291adeac096d05556891fc30c3169740f'
    assert b['ffn_pre']['digest']=='036a63ede630537263cbe5125569fc36680270c7bd35e841bc789705d992cf74'
    assert b['moe_routing_indices']==[[206,278,208,105,307,38]]
    assert b['selected_expert_set']==[38,105,206,208,278,307]
    e=r['Engram1']
    assert e['incoming_Block0_x_out_digest']==b['x_out']['digest']
    assert e['layer1_hash_digest']=='4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373'
    assert e['sparse_embedding']['unique_row_count']==24
    assert e['sparse_embedding']['embedding_byte_exact'] is True
    assert e['gate']['gate_digest']==e['gate']['independent_gate_digest']
    assert e['residual_update']['post_engram1_h']['digest']=='b651ee96bcf7b1f82a5ea5f18678efc85668b4766a69c2fc2b117ee75be0794b'
    assert e['post_Engram1_h']['digest']=='b651ee96bcf7b1f82a5ea5f18678efc85668b4766a69c2fc2b117ee75be0794b'
    assert all(r['persistent_state_before_after_table'].values())
    assert all(v is False for v in r['STOP_flags'].values())
    assert r['call_local_state_table']['Block0_x_out'].startswith('call-local')
    assert all(r['gates'].values()), [k for k,v in r['gates'].items() if not v]
    doc=DOC.read_text()
    for phrase in ['Boundary13d', 'Sparse output digest', 'Block0 x_out digest', 'post-Engram@1 h', 'STOP before Block1']:
        assert phrase in doc, phrase
    print(f'Boundary13d Block0+Engram@1 check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
