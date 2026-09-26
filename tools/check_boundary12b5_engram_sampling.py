#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-engram-connected-sampling-rebind-validation.json'

def main():
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b4_engram_connected_logits.py')],cwd=ROOT).returncode==0
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-engram-connected-sampling-rebind-validation.v1'
    assert r['ok'] is True and r['not_omlx_derived'] is True
    assert r['base_head']=='3b22edba773485d0e1c34799f8b9bdea555b5c74'
    assert r['source_identity']['sample_lines_1285_1292_sha256']=='da6030c7ebf858d615fcdf6b7efb88b5a98f53849b98ffc0815b4eccd467955a'
    lg=r['logits_regression']
    assert lg['logits_digest']=='7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd'
    assert lg['logits_artifact_tensor_injection'] is False
    assert lg['current_engram_connected_logits_regenerated'] is True
    A=r['branch_A_temperature_zero']
    assert A['temperature']==0.0 and A['native_token']==15 and A['independent_token']==15
    assert A['max_logit']==13.22089958190918 and A['tie_count']==1
    assert A['output_ids']==[15] and A['shape']==[1] and A['dtype']=='int64'
    B=r['branch_B_supplied_noise']
    assert B['temperature']==1.0
    assert B['supplied_noise']['digest']=='fe4cf22185a06020968728fcd88706683ac8c16a6f1b0b4f1f561893ee5a629f'
    assert B['supplied_noise']['all_positive'] is True and B['supplied_noise']['all_finite'] is True
    assert B['rng_guard']['official_rng_call_executed'] is False
    assert B['rng_guard']['rng_draw_replaced_by_supplied_fixture_for_arithmetic_validation'] is True
    assert B['probs']['comparison']['pass'] is True
    assert B['probs']['comparison']['max_abs']<=1e-6 and B['probs']['comparison']['sum_abs_error']<=1e-6
    assert B['conditional_output_id']==B['independent_token']==795
    assert B['tie_count']==1 and B['score_gap']>0 and B['log_score_gap']>0
    C=r['branch_C_target_mlx']
    assert C['mlx_version']=='0.32.2' and C['device']=='Metal GPU / mx.gpu'
    assert C['host_to_mlx']['roundtrip_digest']==lg['logits_digest']
    assert C['primary_rng']['session_key']['values']==[0,289513473]
    assert C['primary_rng']['session_key']['digest']=='39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630'
    assert C['primary_rng']['draw_key']['digest']=='4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1'
    assert C['primary_rng']['next_key']['digest']=='175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4'
    assert C['noise']['digest']=='38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4'
    assert C['noise']['all_positive'] is True and C['noise']['all_finite'] is True
    assert C['probs']['comparison']['pass'] is True
    assert C['target_mlx_sampled_token']==C['independent_token']==9468
    assert C['tie_count']==1 and C['score_gap']>0 and C['log_score_gap']>0
    assert all(C['same_key_repeat'].values())
    assert C['secondary_key']['noise_differs'] is True
    assert r['historical_token_non_use_guard']['old_no_engram_sampling_outputs_used_as_expected'] is False
    assert r['stop_boundary']['main_hidden_concat_executed'] is False
    assert r['stop_boundary']['Transformer_forward_return_executed'] is False
    for phrase in ['no PyTorch/MLX RNG bitwise parity','no backend-independent sampled-token identity','no main_hidden concat authority','no Transformer.forward return correctness','no full-model correctness']:
        assert phrase in r['non_claims']
    assert all(r['gates'].values())
    print(f'Boundary12b5 Engram sampling check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
