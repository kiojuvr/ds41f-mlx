#!/usr/bin/env python3
"""Check Boundary12b3 connected Engram@layer14 validation artifact."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-engram-layer14-validation.json'
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main()->int:
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b2_engram_layer1.py')],cwd=ROOT).returncode==0
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-engram-layer14-validation.v1'
    assert r['ok'] is True
    assert r['base_head']=='bafe36b46a627ab18ce23a72d9a7f1453a798a09'
    assert r['classification']=='official-reference-derived bounded connected Engram@layer14 numerical authority with upstream connected Engram@layer1 state'
    assert r['not_omlx_derived'] is True
    assert r['source_identities']['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'==sha(CKPT/'inference/model.py')
    assert r['source_identities']['kernel_py']['sha256']=='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'==sha(CKPT/'inference/kernel.py')
    assert r['source_identities']['ParallelEngramEmbedding']['span_sha256']=='23f3b61ac17a7fe38947171365138fb294d61ee73a41973aff322a45a655b23d'
    assert r['source_identities']['Engram.forward']['span_sha256']=='ff04df4f5312b4d5170b844364ff3e0e823a51f94b7c527656d4e5b08beecaf9'
    assert r['upstream_authorities']['native_fp8_validation_ok'] is True
    assert r['upstream_authorities']['native_fp8_max_bf16_ulp']==0
    h=r['hash_regression']
    assert h['full_digest']=='f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d'
    assert h['layer1_digest']=='8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5'
    assert h['layer14_digest']=='33e046238287e6e8b7c3466bdb4b7b47182ad19f9115afac8a1983ede219ba80'
    assert h['regenerated_from_tokens_in_runner'] is True
    assert h['no_hash_artifact_tensor_injection'] is True
    p1=r['post_engram1_regression']
    assert p1['digest']=='3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9'
    assert p1['exact'] is True
    b=r['blocks1_13']
    assert b['all_x_carries_exact'] is True and b['all_pre_mix_carries_exact'] is True
    assert b['carries']['Engram1->Block1']['x_exact'] is True
    assert b['carries']['Engram1->Block1']['pre_mix_exact'] is True
    for i in range(1,14):
        layer=b['layers'][str(i)]
        for key in ['x_in','pre_mix_in','attention_input','attention_output','x_after_attn','moe_input','moe_output','x_out','ffn_pre']:
            assert layer[key]
    ss=r['shared_attention_state']
    assert ss['same_logical_object_persists'] is True
    assert ss['generation2']['compress_kv']=='838e6e26d9889ef668bc6be7d345be10542466bb86a8c656f3aa5665376143ae'
    assert ss['generation2']['index_k']=='0a02a69899257836865cacec8a1a6d0d1bfb590cf07b8a3f602300e02ee7875d'
    assert ss['generation8']['compress_kv']=='29f479332ce47e4429d7c46d4fc1a952efa16f74ca110b1a9a025f032a9e2213'
    assert ss['generation8']['index_k']=='cd1c51ca26f6404bde0eb2230908af462e9440a923885236f35ba92df6542d87'
    assert ss['all_fields_unchanged_across_Engram14'] is True
    pre=r['pre_engram14_h']
    assert pre['shape']==[1,2,4,5120] and pre['dtype']=='BF16(uint16)'
    assert pre['digest']=='063e002c44f246805e4630678b705cb7987c9064693605fba9297b385e8bb931'
    pm=r['pre_block14_pre_mix']
    assert pm['shape']==[1,2,4] and pm['dtype']=='FP32'
    assert pm['digest']=='950063b739a1c4bb0e75c14850d1cb4921ce27cc366878ebe0b9d1d92e9c1bec'
    sp=r['layer14_sparse_embedding']
    assert len(sp['ordered_row_ids'])==48 and sp['unique_row_count']==48
    assert sp['full_engram_embedding_table_read'] is False and sp['sparse_random_access_rows_only'] is True
    assert sp['weight_bytes_read']==12288 and sp['scale_bytes_read']==384
    assert sp['ordered_rows_raw_payload_digest']=='8d1bbfa560fb056ce548c6044bc5a19bab573fedd33dc0b5ae6d3ecfe2301470'
    assert sp['source_output']['shape']==[1,2,24,256]
    assert sp['source_output']['digest']=='15536731a8a4241b2a7c525111fe0aac17b444c57b68af2f12a1b36bad23534f'
    assert sp['source_vs_independent_byte_exact'] is True
    assert r['layer14_flatten_seam']['byte_preserving'] is True
    w=r['layer14_wkv']
    assert w['raw_weight_digest']=='9abdfe920afa0528c9987a7d01450f5022cc765fa8b115fb132597ddb25cb429'
    assert w['raw_scale_digest']=='8c8d7aa1ccc5a6c24f3e454de2e884db32890874263567889a0e182b3584642a'
    assert w['activation_fp8_quantized_digest']=='fb455db73774caf9124ff2f2093be89338a86ef35ebf8ab325cf7ff04898b3bc'
    assert w['activation_e8m0_scale_digest']=='284e9f6e7e9e32e47c4fd049da557ae3c72854e843ffc7f18e451837e0f29339'
    assert w['output_digest']=='4a8e2062d543cba990a51e49b5106be2820b6bcd8cf0adb60f1d6565681aa7ec'
    assert w['anchor_max_bf16_ulp']==0 and all(x['bf16_ulp']==0 for x in w['anchor_comparisons'])
    kv=r['key_value_split']
    assert kv['key_shape']==[1,2,20480] and kv['key_reshaped_shape']==[1,2,4,5120]
    assert kv['key_digest']=='409b22feb44ea7b1154352677c77f185410e4eec1c3bb79886e8d23cf6bda169'
    assert kv['key_reshape_byte_preserving'] is True
    assert kv['value_digest']=='bb4641ddceb3abfe82cbb2af9197b222798bff8e2539d2faf4efb06dabb10142'
    qk=r['qk_weights']
    assert qk['q_weight_digest']=='a49e74aa912b88f56df4183862c729bc8539ce4a7b76e6107da858b59e782e28'
    assert qk['k_weight_digest']=='fa3128a62fef3630e32a0953e2dc32eb6f6870492ba447bb1877ff37f4f6cd93'
    gate=r['gate']
    assert len(gate['source_records'])==8 and len(gate['independent_records'])==8
    assert gate['dot_max_abs_diff']<=2e-5 and gate['gate_max_abs_diff']<=1e-5
    post=r['residual_update']['post_engram14_h']
    assert post['shape']==[1,2,4,5120] and post['dtype']=='BF16(uint16)'
    assert post['digest']=='ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd'
    assert r['residual_update']['post_bf16_max_ulp']==0 and r['residual_update']['post_byte_exact'] is True
    hand=r['block14_handoff']
    assert hand['post_engram14_h_digest']==post['digest']
    assert hand['block14_pre_mix_digest']==pm['digest']
    assert hand['pre_mix_unchanged_across_engram14'] is True
    assert all(r['connected_seams'].values())
    io=r['io_accounting']
    assert io['layer14_embedding_logical_rows']==48 and io['layer14_embedding_unique_rows']==48
    assert io['total_layer14_engram_checkpoint_bytes_read']==157534592
    assert io['no_full_giant_embedding_table_scan'] is True
    assert r['stop_boundary']['Block14_executed'] is False
    assert r['stop_boundary']['next_operation'].startswith('Block14.forward')
    for phrase in ['no Block14-and-later Engram-connected authority','no main_hidden numeric authority','no Transformer.forward return correctness','no full-model correctness']:
        assert phrase in r['non_claims']
    assert r['next_boundary'].startswith('Boundary 12b4')
    assert all(r['authority_source_guards'].values())
    assert all(r['gates'].values())
    print(f'Boundary12b3 Engram@layer14 check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
