#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CK=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
KERNEL_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
METHOD='raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1'
EXPECTED={
 'x36_out':'499ab386c1c3a547619c1fec569b74712dcc68804148975ba8439bcc35377c05',
 'ffn_pre36':'cb4af2a88e07151442d5fb92ccdebac977b8223f3e80a8a47200af64f0593170',
 'x39_out':'c705421c2422458f36a570338c51e70d50967f028da424abf38ef4bb30bf86d3',
 'ffn_pre39':'8b20fab6cfdec82baf94938ec3e56428cae42a0997bcb0f3aa9afa5e7115f1dc',
 'compress_kv':'fdb027edf978cebd05926802259c639f106f673c997da85129dbaa5ce3f0df41',
 'index_k':'70b8711d0fdf54876d56ff9bd993b7504f5c78463b1bf912634803c194bff1f5',
 'candidates':'27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1',
 'topk_idxs':'f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f',
}
FORBIDDEN=[
 'full Transformer validated', 'full forward validated', 'full model validated', 'prefill fully correct'
]

def sha(p:Path):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def walk_strings(o):
    if isinstance(o,str): yield o
    elif isinstance(o,dict):
        for v in o.values(): yield from walk_strings(v)
    elif isinstance(o,list):
        for v in o: yield from walk_strings(v)

def main():
    close=ROOT/'artifacts/boundary8-closeout.json'
    d=ROOT/'artifacts/native-layer0-39-all-blocks-connected-prefill-validation.json'
    assert close.exists(), close
    assert d.exists(), d
    rec=json.loads(close.read_text()); art=json.loads(d.read_text())
    assert rec['ok'] is True
    assert art['ok'] is True
    assert rec['not_omlx_derived'] is True
    assert art['not_omlx_derived'] is True
    assert rec['current_integrated_numerical_authority']['artifact']==str(d.relative_to(ROOT))
    assert rec['current_integrated_numerical_authority']['commit']=='dae92ab0d9b7cbc6c77d8dac878d3e81b22492ab'
    assert rec['scope']['tokens']==[[0,3]]
    assert rec['scope']['batch']==1 and rec['scope']['sequence']==2 and rec['scope']['start_pos']==0
    assert rec['scope']['world_size']==1 and rec['scope']['layers']=='0..39'
    assert rec['source_topology']['num_hidden_layers']==40
    assert rec['source_topology']['kv_source_layer_ids']==[2,8,14,20]
    assert rec['source_topology']['index_source_layer_ids']==[2,8,14,20,24,28,32,36]
    assert rec['source_topology']['candidate_source_layer_id']==20
    assert rec['source_authority']['model_py_sha256']==MODEL_SHA
    assert rec['source_authority']['kernel_py_sha256']==KERNEL_SHA
    assert rec['source_authority']['source_identity_hash_method']==METHOD
    assert sha(CK/'inference/model.py')==MODEL_SHA
    assert sha(CK/'inference/kernel.py')==KERNEL_SHA
    assert rec['checks']['boundary8d_ok'] is True
    assert rec['checks']['boundary8c_regression_pass'] is True
    assert rec['checks']['all_39_x_carries_exact'] is True
    assert rec['checks']['all_39_pre_mix_carries_exact'] is True
    assert rec['checks']['post_loop_stop_boundary_exact'] is True
    assert rec['checks']['source_identity_guard_pass'] is True
    assert rec['checks']['not_omlx_derived'] is True
    assert rec['top_level_digest_summary']['digest_gates']['x39_out']==EXPECTED['x39_out']
    assert rec['top_level_digest_summary']['digest_gates']['ffn_pre39']==EXPECTED['ffn_pre39']
    assert rec['top_level_digest_summary']['digest_gates']['x36_out']==EXPECTED['x36_out']
    assert rec['top_level_digest_summary']['digest_gates']['ffn_pre36']==EXPECTED['ffn_pre36']
    assert rec['top_level_digest_summary']['final_shared_state_snapshot']=={k:EXPECTED[k] for k in ['compress_kv','index_k','candidates','topk_idxs']}
    assert art['minimum_digest_table']['final_endpoint']['x_out']==EXPECTED['x39_out']
    assert art['minimum_digest_table']['final_endpoint']['ffn_pre']==EXPECTED['ffn_pre39']
    assert art['shared_state_snapshots']['after_layer39']=={k:EXPECTED[k] for k in ['compress_kv','index_k','candidates','topk_idxs']}
    for i in range(39):
        c=art['carries'][f'{i}->{i+1}']
        assert c['x_exact'] is True and c['pre_mix_exact'] is True
    assert rec['post_loop_boundary']['stop']['after']=='Block39.forward returns x39_out, ffn_pre39'
    assert rec['post_loop_boundary']['stop']['before']=='h = layer.hc_pre(h, pre_mix)'
    assert rec['closed_claims']['scope']['tokens']==[[0,3]]
    assert rec['still_open'] and rec['non_claims']
    txt='\n'.join(walk_strings(rec))
    for phrase in FORBIDDEN:
        assert phrase not in txt, phrase
    for artpath in ['artifacts/native-layer0-28-connected-prefill-validation.json','artifacts/native-layer0-32-connected-prefill-validation.json','artifacts/native-layer0-36-connected-prefill-validation.json','artifacts/native-layer0-39-all-blocks-connected-prefill-validation.json','artifacts/boundary7-closeout.json']:
        assert (ROOT/artpath).exists(), artpath
    rc=subprocess.run([sys.executable, str(ROOT/'tools/check_boundary7_closeout.py')], check=False).returncode
    assert rc==0
    print('boundary8 closeout PASS')
    return 0
if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'boundary8 closeout FAIL: {e}', file=sys.stderr)
        raise
