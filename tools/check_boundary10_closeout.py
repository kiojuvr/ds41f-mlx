#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXP={
 'sample_sha':'da6030c7ebf858d615fcdf6b7efb88b5a98f53849b98ffc0815b4eccd467955a',
 'logits':'b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d',
 'noise':'fe4cf22185a06020968728fcd88706683ac8c16a6f1b0b4f1f561893ee5a629f',
 'softmax':'be8593220e2e73092a1a129875f8d331fea446af127b1ef6704aee607f2cca26',
 'scores':'5bdcb0b61b5a72eba6ed429b091e4144356b0b335a4f3df3d5f0fc48f451c564',
}
FORBIDDEN=[
 'official stochastic sampling validated',
 'official sampled token is 795',
 'default generated token is 795',
 'full sample() validated',
 'RNG correctness validated',
 'full Transformer.forward validated',
 'full model validated',
]

def strings(o):
    if isinstance(o,str): yield o
    elif isinstance(o,dict):
        for v in o.values(): yield from strings(v)
    elif isinstance(o,list):
        for v in o: yield from strings(v)

def run(cmd):
    return subprocess.run([sys.executable, str(ROOT/cmd)], cwd=ROOT, check=False).returncode == 0

def main():
    p_close=ROOT/'artifacts/boundary10-closeout.json'
    p9=ROOT/'artifacts/boundary9-closeout.json'
    p10a=ROOT/'artifacts/native-sampling-temperature-zero-validation.json'
    p10b=ROOT/'artifacts/native-sampling-supplied-noise-validation.json'
    for p in [p_close,p9,p10a,p10b]: assert p.exists(), p
    rec=json.loads(p_close.read_text()); b9=json.loads(p9.read_text()); a=json.loads(p10a.read_text()); b=json.loads(p10b.read_text())
    assert rec['ok'] is True
    assert rec['not_omlx_derived'] is True
    assert b9['ok'] is True and a['ok'] is True and b['ok'] is True
    assert a['not_omlx_derived'] is True and b['not_omlx_derived'] is True
    assert rec['authority_hierarchy']['boundary9']['relationship']=='current integrated deterministic model-forward authority through final-position logits'
    assert rec['authority_hierarchy']['boundary10']['relationship']=='sampling branch semantics authority layered on Boundary9 logits'
    assert rec['authority_hierarchy']['boundary10']['does_not_supersede_boundary9'] is True
    assert rec['source_identity']['sample']['source_sha256']==EXP['sample_sha']
    assert rec['source_identity']['sample']['source_lines']==[1285,1292]
    assert rec['source_identity']['sample']['source_hash_method']=='raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1'
    assert a['source_review']['model_py']['spans'][3]['source_sha256']==EXP['sample_sha']
    assert b['source_review']['model_py']['sample_span']['source_sha256']==EXP['sample_sha']
    a10=rec['boundary10a']
    assert a10['artifact']=='artifacts/native-sampling-temperature-zero-validation.json'
    assert a10['artifact_ok'] is True
    assert a10['classification']=='validated explicit parameterized deterministic sampling branch'
    assert a10['not_default_transformer_forward_sampling_authority'] is True
    assert a10['input_logits_digest']==EXP['logits']
    assert a10['temperature']==0.0
    assert a10['max_logit']==12.943254470825195
    assert a10['argmax']==372
    assert a10['tie_count']==1
    assert a10['output_ids']==[372]
    assert a10['output_ids_dtype']=='int64'
    assert a10['output_ids_shape']==[1]
    assert a10['native_argmax_equals_independent_scan'] is True
    b10=rec['boundary10b']
    assert b10['artifact']=='artifacts/native-sampling-supplied-noise-validation.json'
    assert b10['artifact_ok'] is True
    assert b10['classification']=='official-reference-derived sampling arithmetic conditional on supplied deterministic positive noise'
    assert b10['is_official_rng_authority'] is False
    assert b10['is_default_sampled_token_authority'] is False
    assert b10['input_logits_digest']==EXP['logits']
    assert b10['temperature']['input']==1.0
    assert b10['temperature']['effective']==1.0
    assert b10['temperature']['floor_reviewed'] is True
    assert b10['temperature']['floor_exercised'] is False
    n=b10['supplied_noise']
    assert n['shape']==[1,129280]
    assert n['dtype']=='FP32'
    assert n['digest']==EXP['noise']
    assert n['all_positive'] is True and n['all_finite'] is True
    assert n['classification']['not_rng_output'] is True
    assert b10['arithmetic']['scaled_logits_digest']==EXP['logits']
    assert b10['arithmetic']['softmax']['digest']==EXP['softmax']
    assert b10['arithmetic']['scores_digest']==EXP['scores']
    assert b10['arithmetic']['conditional_output_id']==795
    assert b10['independent_winner']['source_order_token']==795
    assert b10['independent_winner']['independent_log_domain_token']==795
    assert b10['independent_winner']['tokens_agree'] is True
    pol=b10['softmax_comparison']['policy']; act=b10['softmax_comparison']['actual']
    assert pol['predeclared_before_execution'] is True
    assert pol['softmax_max_abs_lte']==1e-6 and pol['softmax_sum_abs_error_lte']==1e-6
    assert act['max_abs'] <= pol['softmax_max_abs_lte']
    assert act['sum_abs_error'] <= pol['softmax_sum_abs_error_lte']
    rng=rec['rng_boundary']
    assert rng['official_rng_call_in_source']=='torch.empty_like(probs).exponential_(1)'
    assert rng['official_rng_call_executed'] is False
    assert rng['rng_draw_replaced_by_supplied_fixture_for_arithmetic_validation'] is True
    assert rng['supplied_noise_is_not_claimed_as_official_rng_output'] is True
    assert 'RNG state' in rng['still_open']
    assert rec['boundary9_regression_relationship']['boundary10a_logits_digest']==EXP['logits']
    assert rec['boundary9_regression_relationship']['boundary10b_logits_digest']==EXP['logits']
    assert rec['boundary9_regression_relationship']['artifact_injection'] is False
    stop=rec['stop_boundary']
    assert stop['next_source_operation']=='main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None'
    assert stop['main_hidden_assembly']=='not executed'
    assert stop['transformer_forward_return']=='not executed'
    guard=rec['authority_contamination_guard']
    assert guard['boundary9_remains_current_integrated_deterministic_model_forward_numerical_authority_through_logits'] is True
    assert guard['boundary10b_supplied_noise_evidence_never_official_rng_evidence'] is True
    assert guard['boundary10b_conditional_token_never_official_sampled_token'] is True
    assert guard['omlx_old_evidence_excluded_from_semantic_correctness_authority'] is True
    assert 'official RNG generation' in rec['explicitly_not_closed']
    assert 'main_hidden assembly' in rec['explicitly_not_closed']
    assert 'Transformer.forward return' in rec['explicitly_not_closed']
    txt='\n'.join(strings(rec))
    for phrase in FORBIDDEN: assert phrase not in txt, phrase
    assert run('tools/check_boundary9_closeout.py')
    assert run('tools/check_boundary8_closeout.py')
    assert run('tools/check_authority_labels.py')
    assert run('tools/check_source_identity_hashes.py')
    print('boundary10 closeout PASS')
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print(f'boundary10 closeout FAIL: {e}', file=sys.stderr); raise
