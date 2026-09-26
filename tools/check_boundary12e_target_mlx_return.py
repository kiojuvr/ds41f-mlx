#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-target-mlx-stochastic-integrated-return-validation.json'

def main():
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary11a_rng.py')],cwd=ROOT).returncode==0
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b5_engram_sampling.py')],cwd=ROOT).returncode==0
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12c_main_hidden.py')],cwd=ROOT).returncode==0
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12d_transformer_return.py')],cwd=ROOT).returncode==0
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-target-mlx-stochastic-integrated-return-validation.v1'
    assert r['ok'] is True and r['not_omlx_derived'] is True
    assert r['base_head']=='ffce669112e4a9090d6c1d9b5c34d2e767b08359'
    assert r['classification']=='qualified target-MLX-runtime runtime/key-specific stochastic integrated return authority'
    assert r['scope']['tokens']==[[0,3]] and r['scope']['temperature']==1.0 and r['scope']['seed']==289513473
    assert r['source_identities']['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
    assert r['single_integrated_execution'] is True and r['artifact_tensor_injection'] is False and r['sampled_token_artifact_injection'] is False
    assert r['official_pytorch_rng_executed'] is False and r['pytorch_rng_parity_claimed'] is False
    u=r['upstream_regressions']
    assert u['full_Ngram_hash']=='f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d'
    assert u['post_engram1_h']=='3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9'
    assert u['post_engram14_h']=='ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd'
    caps=r['captures']
    assert caps['append_order']==[37,38,39]
    assert caps['37']['digest']=='dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb'
    assert caps['38']['digest']=='8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c'
    assert caps['39']['digest']=='b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80'
    assert r['logits']['shape']==[1,129280] and r['logits']['dtype']=='FP32'
    assert r['logits']['digest']=='7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd'
    assert r['target_runtime']['mlx_version']=='0.32.2' and r['target_runtime']['device']=='Metal GPU / mx.gpu'
    assert r['host_to_mlx_logits']['roundtrip_digest']==r['logits']['digest']
    pr=r['primary_rng']
    assert pr['session_key']['values']==[0,289513473]
    assert pr['session_key']['digest']=='39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630'
    assert pr['draw_key']['digest']=='4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1'
    assert pr['next_session_key']['digest']=='175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4'
    assert r['noise']['digest']=='38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4'
    assert r['noise']['all_positive'] is True and r['noise']['all_finite'] is True
    smp=r['sampling']
    assert smp['mlx_probs']['digest']=='722acab2f31289fc2d2de5cc8ed588a94efd7a45bae12879a4b0ed08df3072f0'
    assert smp['mlx_probs']['comparison']['pass'] is True
    assert smp['target_mlx_sampled_token']==9468 and smp['independent_token']==9468 and smp['tie_count']==1
    assert smp['pytorch_rng_executed'] is False
    seam=r['output_ids_representation_seam']['native_output_ids']
    assert seam['value']==[9468] and seam['dtype']=='int64' and seam['shape']==[1]
    eo=r['event_order']
    assert eo['capture37']<eo['capture38']<eo['capture39']<eo['logits']<eo['mlx_sampling']<eo['main_hidden_concat']<eo['return']
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
    assert ret['next_session_key_in_transformer_return_tuple'] is False
    assert r['runtime_state_after']['next_session_key']['digest']=='175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4'
    rep=r['same_key_return_tail_reproducibility']
    assert rep['same_generated_noise'] and rep['same_sampled_output_ids'] and rep['same_next_session_key'] and rep['same_main_hidden'] and rep['same_returned_tuple_member_digests']
    assert all(r['gates'].values())
    print(f'Boundary12e target MLX stochastic return check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
