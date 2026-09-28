#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/m4/layer2-sparse-topology/result.json'
def main():
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.m4.layer2-sparse-topology.v1'
    a=r['actual_layer2_inputs']
    assert a['q_dtype_target']=='bfloat16' and a['q']['shape']==[1,1,64,512]
    assert a['wi']['shape']==[1,1,128] and a['wi']['count_minus_one']==125
    assert a['wi']['valid_positions']==[125,126,127] and a['wi']['valid_values']==[0,1,2]
    assert a['ci']['values']==[[[0]]]
    assert a['packed_window']['shape']==[1,3,528]
    assert a['packed_compressed']['shape']==[1,1,288]
    d=r['rounded_target_dispatch']
    assert d['function']=='rounded_packed_attention -> _fused_attention'
    assert d['chunk']==32 and d['block_count']==5 and d['total_sparse_slots']==129
    assert r['coordinate_spaces']['indexer_compressed_local']==[[[0]]]
    assert r['coordinate_spaces']['numpy_concatenated']['indices']==[[[0,1,2,3]]]
    assert r['comparisons']['S_vs_OC']['within_contract'] is True
    assert r['comparisons']['OC_vs_OP']['within_contract'] is False
    assert r['layer2_sparse_status']=='INCOMPLETE'
    assert r['invalid_padding_semantic_check']['all_invalid_are_minus_one'] is True
    print(f'Layer2 sparse topology check PASS (expected incomplete frontier): {ART}')
if __name__=='__main__': main()
