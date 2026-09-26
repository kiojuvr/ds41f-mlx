#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-engram-connected-main-hidden-validation.json'

def main():
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b4_engram_connected_logits.py')],cwd=ROOT).returncode==0
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b5_engram_sampling.py')],cwd=ROOT).returncode==0
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-engram-connected-main-hidden-validation.v1'
    assert r['ok'] is True
    assert r['not_omlx_derived'] is True
    assert r['base_head']=='e95e1a3b2c6bb03029916645dfcffc0bc747782d'
    assert r['source_identities']['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
    assert r['source_identities']['Transformer_forward_lines_1242_1272']['span_sha256']=='6f309bf24d79649472588cb0fda651646f9d8ee46c5b5f0dc5de482cd6f94a0c'
    assert r['source_identities']['config_json']['sha256']=='2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809'
    assert r['config_regression']['dspark_target_layer_ids']==[37,38,39]
    assert r['config_regression']['hc_mult']==4 and r['config_regression']['dim']==5120
    assert r['scope']['tokens']==[[0,3]] and r['scope']['target_layer_ids']==[37,38,39]
    assert r['pytorch_role']=='reference-only bounded framework semantic oracle'
    assert r['pytorch_is_production_dependency'] is False
    pr=r['pytorch_reference']
    assert pr['cpu_reference_executed'] is True
    assert pr['platform_machine']=='arm64'
    assert pr['torch_version']
    assert pr['pytorch_is_production_dependency'] is False
    assert r['preflight']['pass'] is True
    assert r['synthetic_mean_characterization']['exact'] is True
    for layer,dig in {'37':'0cd9f441dbacb6e4eba6e72787130c894fa170b05a34a475924a0c15d79b8f18','38':'0bad9cf92b73548f3881353e6d284d063f7f4d494f1c8b7d4607cc97a3da14e9','39':'1eb3466a1e3be7be796eaf2b6661f9757418c9d8960517fe9baed1f690fc561e'}.items():
        c=r[f'capture{layer}']
        assert c['capture_h']['shape']==[1,2,4,5120]
        assert c['capture_h']['dtype']=='BF16(uint16)'
        assert c['capture_h']['digest']==dig
        assert c['capture_equals_Block_x_in'] is True
        assert c['mean_non_mutating'] is True
        assert c['source_reference_output']['shape']==[1,2,5120]
        assert c['source_reference_output']['dtype']=='torch.bfloat16'
        assert c['pytorch_input_roundtrip_exact'] is True
        assert c['framework_vs_independent_bf16_exact'] is True
        assert c['max_bf16_ulp']==0
    assert r['main_hiddens_append_order']==[37,38,39]
    mh=r['main_hidden']
    assert mh['shape']==[1,2,15360]
    assert mh['dtype']=='torch.bfloat16'
    assert mh['digest']=='4956b1b8bce5101fd567b3f03a4db9dc7739776f7d18d54d26eadffc597e7997'
    assert mh['source_vs_independent_exact'] is True
    assert mh['segment_exact']=={'37':True,'38':True,'39':True}
    assert r['empty_main_hiddens_branch_reviewed'] is True and r['empty_main_hiddens_branch_exercised'] is False
    assert r['logits_regression']['digest']=='7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd'
    assert r['logits_regression']['exact'] is True
    assert r['logits_regression']['sample_or_rng_executed'] is False
    assert r['source_order_distinction']['Transformer_forward_return_executed'] is False
    assert r['main_hidden_capture_artifact_injection'] is False
    for v in r['gates'].values(): assert v is True
    print(f'Boundary12c main_hidden check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
