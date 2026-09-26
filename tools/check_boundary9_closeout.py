#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CK=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
KERNEL_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
METHOD='raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1'
EXP={
 'post_loop_h':'6b99fb26048a577ce78756397101be15441d1ddb7fa9c7aa205007e10869061a',
 'normalized_h':'00186e76a7a7bd78ce40de4a9a912025c9d6724b1600a30f171eca9a2cb65eb5',
 'x39_out':'c705421c2422458f36a570338c51e70d50967f028da424abf38ef4bb30bf86d3',
 'ffn_pre39':'8b20fab6cfdec82baf94938ec3e56428cae42a0997bcb0f3aa9afa5e7115f1dc',
 'norm_weight':'9cd3b57cd9513541b9771bf66c9b356bf1a7b20ff050ed69f7e97cff9fedd428',
 'head_raw':'68f446ddda4243d5c8d57d2a9729c125f7fb8b2ee050c78ac6ff5e0cacde6789',
 'head_runtime':'ea729fc899cda136019ea91bbd85502ccddcf58bab8c7482029d8834415eb8cd',
 'selected_hidden':'79999590fe7867d7f398be3cce6cd4bda7dac7e2fcdf8c1571019175ffb9d746',
 'logits':'b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d',
 'hc_pre_source':'103935a48b2cba8d9e34507bafa52db0f21e2a84448a1a83cdd4f3c5a7e727a4',
 'rmsnorm_source':'ac829397ad0c5f99412def7adb54ba0334397baa5c2ab795d4531f072fc47ecc',
 'parallelhead_source':'673f828a099975e80e8524b8a13ababd456151fe9f588a4f14b45c8ece1b74df',
 'transformer_head_call_source':'bbdcbbf784a2f261ec02408f90c0b97be0d6a1054e2ec83eca63819efe25f1b7',
}
ANCHORS=[0,1,2,3,4096,8192,16384,32768,65536,98304,123456,129279]
FORBIDDEN=['full model correctness validated','full Transformer.forward correctness validated','sampling validated','generation correctness validated']

def sha(p:Path):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def strings(o):
    if isinstance(o,str): yield o
    elif isinstance(o,dict):
        for v in o.values(): yield from strings(v)
    elif isinstance(o,list):
        for v in o: yield from strings(v)

def main():
    close_p=ROOT/'artifacts/boundary9-closeout.json'
    a9_p=ROOT/'artifacts/native-post-loop-hc-collapse-validation.json'
    b9_p=ROOT/'artifacts/native-final-rmsnorm-validation.json'
    c9_p=ROOT/'artifacts/native-parallel-head-logits-validation.json'
    b8_p=ROOT/'artifacts/boundary8-closeout.json'
    for p in [close_p,a9_p,b9_p,c9_p,b8_p]: assert p.exists(), p
    rec=json.loads(close_p.read_text()); a9=json.loads(a9_p.read_text()); b9=json.loads(b9_p.read_text()); c9=json.loads(c9_p.read_text()); b8=json.loads(b8_p.read_text())
    assert rec['ok'] is True
    assert rec['not_omlx_derived'] is True
    assert all(x.get('ok') is True for x in [a9,b9,c9,b8])
    assert all(x.get('not_omlx_derived') is True for x in [a9,b9,c9])
    assert rec['current_integrated_numerical_authority']['artifact']=='artifacts/native-parallel-head-logits-validation.json'
    assert rec['current_integrated_numerical_authority']['commit']=='acf587776f7b9b98ccc6a60b4b34eafeb1ce744e'
    assert rec['scope']=={'tokens':[[0,3]],'B':1,'S':2,'start_pos':0,'prefill':True,'world_size':1,'full_logits':False,'output':'final-position generation logits only'}
    assert rec['source_authority']['model_py_sha256']==MODEL_SHA
    assert rec['source_authority']['kernel_py_sha256']==KERNEL_SHA
    assert rec['source_authority']['source_identity_hash_method']==METHOD
    assert sha(CK/'inference/model.py')==MODEL_SHA
    assert sha(CK/'inference/kernel.py')==KERNEL_SHA
    ids=rec['boundary9_source_identities']
    assert ids['Block.hc_pre']['source_sha256']==EXP['hc_pre_source']
    assert ids['RMSNorm']['source_sha256']==EXP['rmsnorm_source']
    assert ids['ParallelHead.__init__/forward']['source_sha256']==EXP['parallelhead_source']
    assert ids['Transformer final norm/head/sample order']['source_sha256']==EXP['transformer_head_call_source']
    assert rec['boundary9a_facts']['x39_out']==EXP['x39_out']
    assert rec['boundary9a_facts']['ffn_pre39']==EXP['ffn_pre39']
    assert rec['boundary9a_facts']['post_loop_h']['digest']==EXP['post_loop_h']
    assert rec['boundary9a_facts']['post_loop_h']['shape']==[1,2,5120]
    assert rec['boundary9a_facts']['native_vs_independent_exact'] is True
    assert rec['boundary9b_facts']['norm_weight']['raw_digest']==EXP['norm_weight']
    assert rec['boundary9b_facts']['normalized_h']['digest']==EXP['normalized_h']
    assert rec['boundary9b_facts']['independent_arithmetic_exact'] is True
    ph=rec['boundary9c_facts']
    assert ph['head_weight']['raw_bf16_digest']==EXP['head_raw']
    assert ph['head_weight']['runtime_fp32_digest']==EXP['head_runtime']
    assert ph['input_selection']['normalized_h_digest']==EXP['normalized_h']
    assert ph['input_selection']['selected_final_position_hidden_digest']==EXP['selected_hidden']
    assert ph['logits']['shape']==[1,129280]
    assert ph['logits']['dtype']=='FP32'
    assert ph['logits']['digest']==EXP['logits']
    assert ph['parallelhead_contract']['full_logits_default'] is False
    assert ph['parallelhead_contract']['world_size']==1
    assert ph['parallelhead_contract']['all_gather_executed'] is False
    pol=rec['parallelhead_numerical_comparison_policy']
    assert pol['policy']['full_vocab_max_abs_lte']==0.005
    assert pol['policy']['full_vocab_max_rel_lte_where_abs_independent_gte_1']==0.0001
    assert pol['policy']['anchor_max_abs_lte']==0.005
    assert pol['actual']['max_abs'] <= pol['policy']['full_vocab_max_abs_lte']
    assert pol['actual']['max_rel_meaningful'] <= pol['policy']['full_vocab_max_rel_lte_where_abs_independent_gte_1']
    assert pol['actual']['anchor_max_abs'] <= pol['policy']['anchor_max_abs_lte']
    assert pol['native_logits_digest_equals_independent_logits_digest'] is False
    assert rec['predeclared_anchor_authority']['anchor_rows']==ANCHORS
    assert rec['predeclared_anchor_authority']['all_pass'] is True
    assert rec['stop_boundary']['stopped_after']=='logits = self.head(self.norm(h))'
    assert rec['stop_boundary']['next_source_operation']=='output_ids = sample(logits, self.temperature)'
    assert all(x in rec['stop_boundary']['not_executed'] for x in ['sample()','temperature transform','random draw','output_ids'])
    assert rec['checks']['boundary9c_fixed_current_integrated_numerical_authority'] is True
    assert rec['checks']['bounded_comparison_policy_recorded'] is True
    assert rec['checks']['actual_comparison_pass'] is True
    assert rec['checks']['source_identity_guard_pass'] is True
    assert rec['checks']['not_omlx_derived'] is True
    assert rec['authority_contamination_guard']['boundary9c_current_integrated_numerical_authority'] is True
    assert rec['closed_claims']['scope']['full_logits'] is False
    assert 'sampling()' in rec['still_open']
    assert 'full Transformer.forward correctness' in rec['still_open']
    txt='\n'.join(strings(rec))
    for phrase in FORBIDDEN: assert phrase not in txt, phrase
    rc=subprocess.run([sys.executable, str(ROOT/'tools/check_boundary8_closeout.py')], check=False).returncode; assert rc==0
    rc=subprocess.run([sys.executable, str(ROOT/'tools/check_authority_labels.py')], check=False).returncode; assert rc==0
    print('boundary9 closeout PASS')
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print(f'boundary9 closeout FAIL: {e}', file=sys.stderr); raise
