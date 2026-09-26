#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-engram-connected-deterministic-logits-validation.json'

def main():
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b3_engram_layer14.py')],cwd=ROOT).returncode==0
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-engram-connected-deterministic-logits-validation.v1'
    assert r['ok'] is True
    assert r['base_head']=='4b9985f9fc2cb42cba6e20f6f1538e82e834beda'
    assert r['not_omlx_derived'] is True
    u=r['upstream_regressions']
    assert u['full_hash']=='f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d'
    assert u['layer1_hash']=='8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5'
    assert u['layer14_hash']=='33e046238287e6e8b7c3466bdb4b7b47182ad19f9115afac8a1983ede219ba80'
    assert u['post_engram1_h']=='3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9'
    assert u['post_engram14_h']=='ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd'
    assert u['ffn_pre13']=='950063b739a1c4bb0e75c14850d1cb4921ce27cc366878ebe0b9d1d92e9c1bec'
    assert u['shared_state_after_Block13']==u['expected']['shared_state_after_Block13']
    assert r['carries']['Engram14->Block14']['x_exact'] is True
    assert r['carries']['Engram14->Block14']['pre_mix_exact'] is True
    for i in range(14,40):
        layer=r['layers'][str(i)]
        for k in ['x_in','pre_mix_in','attention_input','attention_output','x_after_attn','moe_input','moe_output','x_out','ffn_pre','window_kv']:
            assert layer[k]
    for i in range(15,40):
        assert r['carries'][f'{i-1}->{i}']['x_exact'] is True
        assert r['carries'][f'{i-1}->{i}']['pre_mix_exact'] is True
    gen=r['generation_summary']
    assert gen['old_no_engram_generation14_used_as_expected'] is False
    assert gen['generation14']['compress_kv']=='e2c045f500a5776d647fc22ebc99faf2bb2f7131ba2e332e256fed3aadfa5edb'
    assert gen['generation14']['index_k']=='82f4b94a61be422936f51e142f786be31a37df14412ac700ef804a98f56f5c85'
    assert gen['generation20']['compress_kv']=='17eacfb671aa2a37c302b5e6f097346b7951f7bd3ff23bd0ed5193c00695900d'
    assert gen['generation20']['index_k']=='a2ec36dab41f04ef5f7ab63fabcaa823a22232b9a2fdb3d28d990dfa75e4a896'
    assert gen['generation20']['candidates']=='27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1'
    for k in ['topk24','topk28','topk32','topk36']:
        assert gen[k]=='f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f'
    caps=r['target_layer_capture_inputs']
    assert caps['37']['pre_block_h']['digest']=='0cd9f441dbacb6e4eba6e72787130c894fa170b05a34a475924a0c15d79b8f18'
    assert caps['38']['pre_block_h']['digest']=='0bad9cf92b73548f3881353e6d284d063f7f4d494f1c8b7d4607cc97a3da14e9'
    assert caps['39']['pre_block_h']['digest']=='1eb3466a1e3be7be796eaf2b6661f9757418c9d8960517fe9baed1f690fc561e'
    assert r['block39']['x39_out']['digest']=='691569afdbde87a410893559805d72e031f5b8f91a321711e733af0ad1018777'
    assert r['block39']['ffn_pre39']['digest']=='786c17ed3cf248105f4456b28e693ded5912973d73b11b154501cd1a66a384e7'
    assert r['post_loop_hc_collapse']['collapsed_fp32_exact'] is True
    assert r['post_loop_hc_collapse']['collapsed_bf16_exact'] is True
    assert r['post_loop_hc_collapse']['post_loop_h']['digest']=='7a06d0ac4010cb2ddd4eb0d12b937bc310bffefd492b0ffec98684dcdfd70bd9'
    assert r['final_rmsnorm']['bf16_exact'] is True
    assert r['final_rmsnorm']['norm_weight_observed_digest']==r['final_rmsnorm']['norm_weight_digest']
    assert r['final_rmsnorm']['normalized_h']['digest']=='075115019d3f243d4fb2de85a56c4a2ed69b3d8b27d872a06b4384cff461f1a7'
    ph=r['parallel_head']
    assert ph['head_weight_observed_digest']==ph['head_weight_raw_bf16_digest']=='68f446ddda4243d5c8d57d2a9729c125f7fb8b2ee050c78ac6ff5e0cacde6789'
    assert ph['selected_final_position_hidden']['digest']=='514a4c013d818a8a64117a5784ca2e8e771680ae66b3a23b9ebca8e901269eaa'
    assert ph['logits']['digest']=='7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd'
    assert ph['logits']['argmax_token']==15
    assert ph['full_vocab_pass'] is True
    assert ph['anchor_pass'] is True
    assert ph['max_abs']<=0.005
    assert ph['max_rel_where_abs_independent_gte_1']<=0.0001
    assert ph['anchor_max_abs']<=0.005
    assert r['stop_boundary']['sample_executed'] is False
    assert r['stop_boundary']['main_hidden_concat_executed'] is False
    assert r['authority_transition']['current_engram_connected_authority'] is True
    assert r['authority_transition']['old_sampling_tokens_valid_as_expected_for_new_logits'] is False
    assert r['authority_contamination_guard']['old_no_engram_numeric_outputs_used_as_expected'] is False
    assert all(r['gates'].values())
    print(f'Boundary12b4 Engram-connected logits check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
