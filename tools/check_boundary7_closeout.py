#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CK=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
KERNEL_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
METHOD='raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1'
EXPECTED={
 'embedding_output':'e785817ca379b27e5a5d1c905c6261b46c6158815b55a29ecfaf7850b5925ef1',
 'initial_hc_x':'5d0d812064ce2fc25ea149ef428b80354094b902d0212b9077e33d8c3ff087cb',
 'initial_pre_mix':'56e94d4f8d9e543d1260250ae1fdc346aea6fc5468f1a7865e77534e2fce98a1',
 'x1_to_x2':'a9abb13060e3b5986de38d653b16d9e91189f68c8815bbaeba6d8cd647fca8a9',
 'pre_mix1_to_2':'393d4e5fd0d9209d0374f1896d2504e008d806ede1ec4b23b4260fba3816b4b3',
 'compress_kv_2':'b0344a39d07f2603a5c05828987655cd23a1d5173c7373d1a7522825c2c96d44',
 'index_k_2':'e619fe853544e92679d9705aa2cef4c3199d02d84c2d946e924f2365fc76fcaf',
 'compress_kv_8':'9b94dcb8f668c171fcbff6fbe62ec04f53289a800aa10909db1ea9df0ff181dd',
 'index_k_8':'e14360a0265915b6b292ffca18d678e034f0dbf9de05b1fb59e20b2b871fc725',
 'compress_kv_14':'a582748a87b83769b08b0ec08fcd40c4a72407ffb2ce794830e65ab9c0eac5ed',
 'index_k_14':'51ccd331ca8f5e2bd92471d9c8cc24de087d1623307b192441ba20a0b95a23db',
 'compress_kv_20':'fdb027edf978cebd05926802259c639f106f673c997da85129dbaa5ce3f0df41',
 'index_k_20':'70b8711d0fdf54876d56ff9bd993b7504f5c78463b1bf912634803c194bff1f5',
 'candidates_20':'27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1',
 'topk_24':'f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f',
 'x25_out':'703ad30300805df5266836a8a57427409853deb3402973bab703977fab1cea4b',
 'ffn_pre25':'41cd033507e0dcc64e47d1375d3c6c5dea7a90abce1515de4c6ef05bcac33504',
}

def sha(p:Path):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def walk_strings(o):
    if isinstance(o,str):
        yield o
    elif isinstance(o,dict):
        for v in o.values(): yield from walk_strings(v)
    elif isinstance(o,list):
        for v in o: yield from walk_strings(v)

def main():
    path=ROOT/'artifacts/boundary7-closeout.json'
    rec=json.loads(path.read_text())
    assert rec['ok'] is True
    assert rec['not_omlx_derived'] is True
    assert rec['source_authority']['model_py_sha256']==MODEL_SHA
    assert rec['source_authority']['kernel_py_sha256']==KERNEL_SHA
    assert rec['source_authority']['source_identity_hash_method']==METHOD
    assert sha(CK/'inference/model.py')==MODEL_SHA
    assert sha(CK/'inference/kernel.py')==KERNEL_SHA
    for item in rec['authority_chain']:
        art=item.get('artifact') or item.get('validation')
        if art:
            p=ROOT/art
            assert p.exists(), art
            a=json.loads(p.read_text())
            if 'ok' in a:
                assert a.get('ok') is True, art
    g=json.loads((ROOT/'artifacts/native-layer0-25-transformer-entry-validation.json').read_text())
    assert g['ok'] is True
    td=g['transformer_entry_digest_table']; md=g['minimum_digest_table']
    checks={
      'embedding_output':td['embedding_output'], 'initial_hc_x':td['initial_hc_x'], 'initial_pre_mix':td['initial_pre_mix'],
      'x1_to_x2':td['block2']['input_x'], 'pre_mix1_to_2':td['block2']['incoming_pre_mix'],
      'compress_kv_2':md['layer2']['compress_kv_publication'], 'index_k_2':md['layer2']['index_k_publication'],
      'compress_kv_8':md['layer8']['compress_kv_publication'], 'index_k_8':md['layer8']['index_k_publication'],
      'compress_kv_14':md['layer14']['compress_kv_publication'], 'index_k_14':md['layer14']['index_k_publication'],
      'compress_kv_20':md['layer20']['compress_kv_publication'], 'index_k_20':md['layer20']['index_k_publication'],
      'candidates_20':md['layer20']['candidates_publication'], 'topk_24':md['layer24']['topk_publication'],
      'x25_out':md['layer25']['x25_out'], 'ffn_pre25':md['layer25']['ffn_pre25'],
    }
    assert checks==EXPECTED
    assert rec['top_level_digest_summary']['digest_gates']==EXPECTED
    forbidden=['omlx-derived semantic authority','deepseek-v41-flash-mlx semantic authority']
    text='\n'.join(walk_strings(rec)).lower()
    for f in forbidden:
        assert f not in text
    for phrase in ['no layer26+','no Transformer final norm','no connected ParallelHead/logits','no full-model correctness']:
        assert phrase in rec['non_claims']
    print('boundary7 closeout PASS')
    return 0
if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'boundary7 closeout FAIL: {e}', file=sys.stderr)
        raise
