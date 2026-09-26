#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/native-prefill-end-persistent-state-validation.json'
DOC = ROOT / 'docs/prefill-end-persistent-state-validation.md'

EXPECTED_SOURCES = {
    'model_py': '4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65',
    'engram_py': '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897',
    'config_json': '2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809',
    'generate_py': '8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0',
}
EXPECTED_SPANS = {
    'NgramHashState': '4948807de7a25944b0c3dfaf09ea0d8fc9c9e77cc8a9efaddc2226a3e5413bbe',
    'Compressor': 'f8f76a2dad907c2468a82fb0dc07550cbc696fbc17ede4c24ad9a5cf58c3410c',
    'Indexer': 'e59d197a495bb2af96d44190ad4b8664d24f7460f36db1ea24b29807b050ac07',
    'Attention': '6922d64f9accc5af89f8e5f23c21870beaac5d2a8793ab9ca09dfb2599021b82',
    'SharedAttentionRuntime': '4e20039aaeb433f9d3ecbd569bc3da0bdee45e09f5f5eac2cb541f6ab86b4748',
    'Transformer_forward': '3503e2aca986abba129ded558eb3db80d5045900f0c2cbe78cec768827f5c02c',
    'generate_loop': '109c68796bf2265de649cd8a7a10953d920dc79a0985002d1346a4870a402425',
}
EXP_COMP = {
    '2': '838e6e26d9889ef668bc6be7d345be10542466bb86a8c656f3aa5665376143ae',
    '8': '29f479332ce47e4429d7c46d4fc1a952efa16f74ca110b1a9a025f032a9e2213',
    '14': 'e2c045f500a5776d647fc22ebc99faf2bb2f7131ba2e332e256fed3aadfa5edb',
    '20': '17eacfb671aa2a37c302b5e6f097346b7951f7bd3ff23bd0ed5193c00695900d',
}
EXP_INDEX = {
    '2': '0a02a69899257836865cacec8a1a6d0d1bfb590cf07b8a3f602300e02ee7875d',
    '8': 'cd1c51ca26f6404bde0eb2230908af462e9440a923885236f35ba92df6542d87',
    '14': '82f4b94a61be422936f51e142f786be31a37df14412ac700ef804a98f56f5c85',
    '20': 'a2ec36dab41f04ef5f7ab63fabcaa823a22232b9a2fdb3d28d990dfa75e4a896',
}
ALLOWED_DIRTY = {
    'tools/native_decode_session_state.py',
    'tools/run_native_prefill_end_persistent_state_validation.py',
    'tools/check_boundary13a_prefill_state.py',
    'docs/prefill-end-persistent-state-validation.md',
    'artifacts/native-prefill-end-persistent-state-validation.json',
}

def dirty_paths() -> set[str]:
    out = subprocess.run(['git','status','--short'], cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    return {line[3:] for line in out}

def main() -> int:
    assert ART.exists(), ART
    assert DOC.exists(), DOC
    dirty = dirty_paths()
    if not dirty:
        assert subprocess.run([sys.executable, str(ROOT/'tools/check_boundary13_decode_state_audit.py')], cwd=ROOT).returncode == 0
    else:
        assert dirty <= ALLOWED_DIRTY, dirty
    assert subprocess.run([sys.executable, str(ROOT/'tools/check_boundary12d_transformer_return.py')], cwd=ROOT).returncode == 0

    r = json.loads(ART.read_text())
    assert r['schema'] == 'ds41f.native-prefill-end-persistent-state-validation.v1'
    assert r['ok'] is True and r['not_omlx_derived'] is True
    assert r['base_head'] == '2a0eefcb6d4d0d512759a24672ee383ca06c9db0'
    assert r['classification'] == 'current official-source-derived bounded prefill-end persistent-state snapshot authority'
    assert r['decode_forward_executed'] is False
    assert r['state_container_structure']['class'] == 'NativeDecodeSessionState'
    assert r['state_container_structure']['partitions'] == ['official_model_persistent_state','generation_loop_control','target_runtime_session_state']

    m = r['manifest']
    assert m['fixture'] == {'B':1,'S':2,'engram_mask':None,'start_pos':0,'tokens':[[0,3]],'world_size':1}
    assert m['persistent_state_tensor_artifact_injection'] is False
    assert m['decode_forward_executed'] is False
    assert r['snapshot_manifest_digest'] == m['snapshot_manifest_digest'] == '311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301'
    for k, v in EXPECTED_SOURCES.items():
        assert m['source_identities'][k]['sha256'] == v, k
    for k, v in EXPECTED_SPANS.items():
        assert m['source_identities']['Boundary13_spans'][k]['span_sha256'] == v, k

    official = m['official_model_persistent_state']
    ng = official['ngram_cache']
    assert ng['visible_slice_values'] == [[0,3]]
    assert ng['visible_slice_digest'] == '96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a'
    assert ng['allocated_shape'] == [4,4096]
    assert 'excluded' in ng['unused_capacity']
    assert ng['future_decode_step0_dependency']['history_reads'] == ['pos1','pos0','sequence_beginning_pad']

    win = official['per_layer_window_kv_cache']
    assert len(win) == 40
    for i in range(40):
        e = win[str(i)]
        assert e['owner'] == f'layers.{i}.attn.window_kv_cache'
        assert e['allocated_shape'] == [4,128,512]
        assert e['ring_slots'] == [0,1] and e['valid_absolute_positions'] == [0,1]
        assert e['visible_slice_shape'] == [1,2,512]
        assert e['visible_slice_digest'] == e['producer_window_kv_digest']
        assert e['cache_visible_slots_equal_producer_window_kv'] is True
        assert 'excluded' in e['unused_capacity']

    comp = official['kv_source_compress_kv_cache']
    idx = official['owner_indexer_k_cache']
    assert sorted(comp.keys(), key=int) == ['2','8','14','20']
    assert sorted(idx.keys(), key=int) == ['2','8','14','20']
    assert r['upstream_regressions']['compress_kv'] == EXP_COMP
    assert r['upstream_regressions']['index_k'] == EXP_INDEX
    for layer, ratio, vlen in [('2',2,1),('8',2,1),('14',2,1),('20',1,2)]:
        assert comp[layer]['ratio'] == ratio and comp[layer]['valid_compressed_index_range'] == [0,vlen]
        assert comp[layer]['visible_prefix_digest'] == EXP_COMP[layer]
        assert idx[layer]['ratio'] == ratio and idx[layer]['valid_compressed_index_range'] == [0,vlen]
        assert idx[layer]['visible_prefix_digest'] == EXP_INDEX[layer]

    partial = official['ratio_gt1_compressor_partial_state']
    assert sorted(partial.keys(), key=int) == ['2','8','14']
    for v in partial.values():
        assert v['read_by_decode_step0_bytes'] == 0
        assert v['visible_digest_recorded'] is False
        assert v['slot_classification']['0'] == 'OVERWRITTEN_BEFORE_READ'
        assert v['slot_classification']['1'] == 'UNOBSERVED'

    excl = m['excluded_call_local_fields']
    assert excl['call_local_values_excluded_from_persistent_snapshot'] is True
    assert excl['candidates_in_persistent_state'] is False and excl['topk_idxs_in_persistent_state'] is False
    for k in ['h','pre_mix','main_hiddens']:
        assert 'excluded' in excl[k]
    for k in ['compress_kv','index_k','candidates','topk_idxs']:
        s = excl[f'shared_attn.{k}']
        assert s['physical_attribute_may_exist_after_prefill'] is True
        assert s['semantic_cross_call_value_required'] is False

    gen = m['generation_loop_control']
    assert gen['current_prompt'] == [[0,3]] and gen['prefill_output_token'] == 15
    assert gen['next_input_ids'] == [[15]] and gen['next_start_pos'] == 2 and gen['next_sequence_length'] == 1
    rng = m['target_runtime_session_state']
    assert rng['stochastic_boundary12e_next_session_key_digest'] == '175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4'
    assert rng['not_part_of_official_model_persistent_state'] and rng['not_a_Transformer_forward_return_member']

    regs = r['upstream_regressions']
    assert regs['full_Ngram_hash'] == 'f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d'
    assert regs['post_engram1_h'] == '3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9'
    assert regs['post_engram14_h'] == 'ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd'
    assert regs['final_logits'] == '7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd'
    assert regs['output_ids'] == [15]
    comp13 = r['completeness_against_Boundary13']
    assert comp13['bidirectional_completeness_PASS'] is True
    assert comp13['every_source_persistent_field_classified'] is True
    assert comp13['every_snapshotted_field_source_proven_persistent'] is True
    assert all(r['gates'].values()), [k for k, v in r['gates'].items() if not v]
    doc = DOC.read_text()
    for phrase in ['Boundary13a', 'NativeDecodeSessionState', 'Ngram visible slice', 'Compressor partial-state classification', 'Snapshot manifest digest', 'No first-decode']:
        assert phrase in doc, phrase
    print(f'Boundary13a prefill-end persistent-state check PASS: {ART}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
