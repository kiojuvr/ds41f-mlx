#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-integrated-transformer-forward-return-validation.json'

def main():
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b4_engram_connected_logits.py')],cwd=ROOT).returncode==0
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b5_engram_sampling.py')],cwd=ROOT).returncode==0
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12c_main_hidden.py')],cwd=ROOT).returncode==0
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-integrated-transformer-forward-return-validation.v1'
    assert r['ok'] is True and r['not_omlx_derived'] is True
    assert r['base_head']=='58b81ed400ca488f497661682252b62117a7a582'
    s=r['source_identities']
    assert s['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
    assert s['Transformer_forward_lines_1242_1272']['span_sha256']=='6f309bf24d79649472588cb0fda651646f9d8ee46c5b5f0dc5de482cd6f94a0c'
    assert s['main_hidden_span_lines_1259_1272']['span_sha256']=='56ba7227f4202c0791c33e08300943a0675ef07b62ee824af92889d1e7dd57a3'
    assert s['return_packaging_span_lines_1269_1272']['line_start']==1269 and s['return_packaging_span_lines_1269_1272']['line_end']==1272
    assert r['scope']['tokens']==[[0,3]] and r['scope']['temperature']==0.0
    assert r['single_integrated_execution'] is True and r['artifact_tensor_injection'] is False
    assert r['Boundary12c_checker_PASS'] is True
    assert r['torch_reference_runtime_used_by_Boundary12d_execution'] is False
    assert r['torch_is_production_dependency'] is False
    u=r['upstream_regressions']
    assert u['full_Ngram_hash']=='f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d'
    assert u['post_engram1_h']=='3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9'
    assert u['post_engram14_h']=='ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd'
    assert u['pre_Block37_h']=='0cd9f441dbacb6e4eba6e72787130c894fa170b05a34a475924a0c15d79b8f18'
    assert u['pre_Block38_h']=='0bad9cf92b73548f3881353e6d284d063f7f4d494f1c8b7d4607cc97a3da14e9'
    assert u['pre_Block39_h']=='1eb3466a1e3be7be796eaf2b6661f9757418c9d8960517fe9baed1f690fc561e'
    caps=r['captures']
    assert caps['append_order']==[37,38,39]
    assert caps['37']['digest']=='dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb'
    assert caps['38']['digest']=='8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c'
    assert caps['39']['digest']=='b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80'
    lg=r['logits']
    assert lg['shape']==[1,129280] and lg['dtype']=='FP32'
    assert lg['digest']=='7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd'
    assert lg['argmax']==15 and lg['max']==13.22089958190918
    smp=r['sampling']
    assert smp['temperature']==0.0 and smp['output_ids']==[15]
    assert smp['shape']==[1] and smp['dtype']=='int64' and smp['tie_count']==1
    assert smp['independent_full_vocab_argmax']==15
    assert smp['stochastic_rng_executed'] is False and smp['mlx_rng_executed'] is False and smp['pytorch_rng_executed'] is False
    eo=r['event_order']
    assert eo['capture37']<eo['capture38']<eo['capture39']<eo['logits']<eo['sample']<eo['main_hidden_concat']<eo['return']
    mh=r['main_hidden']
    assert mh['shape']==[1,2,15360] and mh['dtype']=='BF16(uint16)'
    assert mh['digest']=='4956b1b8bce5101fd567b3f03a4db9dc7739776f7d18d54d26eadffc597e7997'
    assert mh['segment_digests']=={'0:5120':'dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb','5120:10240':'8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c','10240:15360':'b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80'}
    ret=r['return_packaging']
    assert ret['container_type']=='tuple' and ret['length']==3
    assert ret['member_order']==['output_ids','logits','main_hidden']
    assert ret['member_shapes']==[[1],[1,129280],[1,2,15360]]
    assert ret['member_dtypes']==['int64','FP32','BF16(uint16)']
    assert all(ret['byte_exact'].values()) and ret['packaging_performs_no_cast_or_mutation'] is True
    assert r['empty_main_hiddens_branch_reviewed'] is True and r['empty_main_hiddens_branch_exercised'] is False
    assert all(r['gates'].values())
    print(f'Boundary12d Transformer.forward return check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
