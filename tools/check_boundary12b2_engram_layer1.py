#!/usr/bin/env python3
"""Check Boundary12b2 connected Engram@layer1 validation artifact."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-engram-layer1-validation.json'
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main()->int:
    assert subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b1_ngram_hash_state.py')],cwd=ROOT).returncode==0
    r=json.loads(ART.read_text())
    assert r['schema']=='ds41f.native-engram-layer1-validation.v1'
    assert r['ok'] is True
    assert r['base_head']=='4a9dce24bd263ede0f41cabfbbf819c729ffc4a2'
    assert r['classification']=='official-reference-derived bounded connected Engram@layer1 numerical authority'
    assert r['not_omlx_derived'] is True
    assert r['source_identities']['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'==sha(CKPT/'inference/model.py')
    assert r['source_identities']['kernel_py']['sha256']=='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'==sha(CKPT/'inference/kernel.py')
    assert r['source_identities']['ParallelEngramEmbedding']['span_sha256']=='23f3b61ac17a7fe38947171365138fb294d61ee73a41973aff322a45a655b23d'
    assert r['source_identities']['Engram.forward']['span_sha256']=='ff04df4f5312b4d5170b844364ff3e0e823a51f94b7c527656d4e5b08beecaf9'
    assert r['source_identities']['linear']['span_sha256']=='c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada'
    assert r['source_identities']['act_quant']['span_sha256']=='563a82836450bfefe3f5f176636dec0e1f4d126c76d8007b42bb7abb877d30cb'
    assert r['source_identities']['fp8_gemm']['span_sha256']=='cfd550d8b02ee127760ac26b39accae603302be6bf8e97c0cce29d8993af0657'
    assert r['upstream_regression']['fp8_reference_artifact_ok'] is True
    assert r['upstream_regression']['native_fp8_validation_ok'] is True
    assert r['upstream_regression']['native_fp8_max_bf16_ulp']==0
    assert r['upstream_regression']['embedding_digest']=='e785817ca379b27e5a5d1c905c6261b46c6158815b55a29ecfaf7850b5925ef1'
    assert r['upstream_regression']['block0_x_out_pass'] is True
    assert r['hash_regression']['full_digest']=='f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d'
    assert r['hash_regression']['layer1_digest']=='8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5'
    assert r['hash_regression']['regenerated_from_tokens_in_runner'] is True
    assert r['hash_regression']['no_hash_artifact_tensor_injection'] is True
    pre=r['pre_engram1_h']
    assert pre['shape']==[1,2,4,5120] and pre['dtype']=='BF16(uint16)'
    assert pre['digest']=='ba2e6acdac3178115513c81e871f0106541cba2810f7dbc5c4a0c8c309dbf938'
    assert pre['producer']=='tokens -> embedding -> HC repeat -> Block0; no prior Engram'
    sp=r['sparse_embedding']
    assert len(sp['ordered_row_ids'])==48
    assert sp['unique_row_count']==48
    assert sp['full_engram_embedding_table_read'] is False
    assert sp['sparse_random_access_rows_only'] is True
    assert sp['weight_bytes_read']==12288 and sp['scale_bytes_read']==384
    assert len(sp['weight_row_provenance'])==48 and len(sp['scale_row_provenance'])==48
    assert sp['ordered_rows_raw_payload_digest']=='13274474cc2cc860c598b6f238e3b2c4c84a9263378d916ed1be9d89879d4fdf'
    assert sp['source_output']['shape']==[1,2,24,256]
    assert sp['source_output']['digest']=='7a203d55dd97c4d0398bb81b70409d299a246289f60e8a2ecb209305cc180ed4'
    assert sp['source_vs_independent_byte_exact'] is True
    assert r['flatten_seam']['flatten_shape']==[1,2,6144]
    assert r['flatten_seam']['byte_preserving'] is True
    w=r['wkv']
    assert w['raw_weight_digest']=='44d799d6444a3f728ae96c873c8199bdff49cafbc0b5d703a0bf07be948d0447'
    assert w['raw_scale_digest']=='2c395fc76e65fad2f33e5a3d8f532f40dc232754a0e637e42ca89a674e3b44e0'
    assert w['activation_fp8_quantized_digest']=='b6279f2d20dab04c953cc8ed88aeafafaa0fa83bc645515169e3b0d5443517d7'
    assert w['activation_e8m0_scale_digest']=='ec3abae604bf0f0888638de597a70b8dd7e6eea88e883523bda23c28023c8136'
    assert w['output_digest']=='0d2fb1ad830f2c0e9c8ee6407bc22dc06fc172a361665ea833880efaacb7907d'
    assert w['output_shape']==[2,25600]
    assert w['anchor_rows']==[0,31,32,5119,5120,10239,10240,15359,15360,20447,20448,20479,20480,20511,25568,25599]
    assert w['anchor_max_bf16_ulp']==0 and all(x['bf16_ulp']==0 for x in w['anchor_comparisons'])
    kv=r['key_value_split']
    assert kv['key_shape']==[1,2,20480] and kv['key_reshaped_shape']==[1,2,4,5120]
    assert kv['key_digest']=='a38eeaf361caf47b12802f77d2932c1c5f800421ace0cef810fae69eff0dfb0a'
    assert kv['key_reshaped_digest']==kv['key_digest']
    assert kv['value_shape']==[1,2,5120]
    assert kv['value_digest']=='3d0617311e3dec2726b699741e812f6271fb641e76259131c394d580baa945e6'
    qk=r['qk_weights']
    assert qk['q_weight_digest']=='0aca22a679f1a2479e8065a7b3c64e035b13fe1fa63189644e3556b0a333da56'
    assert qk['k_weight_digest']=='61152efdabab20ca8c29fe2e087d7de9c5c1684bfef930fa00541efe58fff422'
    gate=r['gate']
    assert len(gate['source_records'])==8 and len(gate['independent_records'])==8
    assert gate['dot_max_abs_diff']<=gate['dot_max_abs_lte']<=2e-5
    assert gate['gate_max_abs_diff']<=gate['gate_max_abs_lte']<=1e-5
    post=r['residual_update']['post_engram1_h']
    assert post['shape']==[1,2,4,5120] and post['dtype']=='BF16(uint16)'
    assert post['digest']=='3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9'
    assert r['residual_update']['post_bf16_max_ulp']==0
    assert r['residual_update']['post_byte_exact'] is True
    assert all(r['connected_seams'].values())
    assert r['stop_boundary']['Block1_executed'] is False
    assert r['stop_boundary']['Engram14_executed'] is False
    assert r['stop_boundary']['next_operation']=='Block1.forward'
    io=r['io_accounting']
    assert io['embedding_logical_rows']==48 and io['embedding_unique_rows_requested']==48
    assert io['no_full_giant_embedding_table_scan'] is True
    assert io['total_checkpoint_bytes_read']==157534592
    for phrase in ['no Engram@layer14 numeric authority','no Block1-and-later Engram-connected authority','no main_hidden numeric authority','no Transformer.forward return correctness','no full-model correctness']:
        assert phrase in r['non_claims']
    assert r['next_boundary']=='Boundary 12b3: Engram@layer14 with upstream connected Engram@1 state'
    assert all(r['authority_source_guards'].values())
    assert all(r['gates'].values())
    print(f'Boundary12b2 Engram@layer1 check PASS: {ART}')
    return 0
if __name__=='__main__': raise SystemExit(main())
