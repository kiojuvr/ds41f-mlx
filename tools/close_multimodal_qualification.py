"""Fail-closed canonical closeout of multimodal development/core evidence."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'artifacts/vision'


def main():
    a=json.loads((EVIDENCE/'initial.json').read_text())
    b=json.loads((EVIDENCE/'restored.json').read_text())
    reference=json.loads((EVIDENCE/'official-reference.json').read_text())
    assert a['status']=='PASS_INITIAL' and b['status']=='PASS_RESTORE' and reference['status']=='PASS'
    assert (EVIDENCE/'sequence.exit').read_text().strip()=='0'
    for receipt in (a,b):
        for path,digest in receipt['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        assert receipt['admission']['resource_set_sha256']==hashlib.sha256((ROOT/'ds41f_mlx/runtime/admitted_resources.json').read_bytes()).hexdigest()
        assert receipt['resource_policy_restored_exactly'] and not receipt['retired_admission']['active']
        assert receipt['active_generation_sessions_after_close']==0
        assert receipt['displaced_numerical_imports_and_calls_forbidden']
    # Active P7 overlap is observed on the native/max path. Fresh restored M8
    # uses the existing direct SSD/P6 path; absence of overlap is not fallback
    # from an activated prefetch and cannot be advertised as overlap evidence.
    assert a['p7_events']['engram_consume']>0
    assert len(a['turns'])==3
    answers=[t['turn']['response_json']['choices'][0]['message']['content'].lower() for t in a['turns']]
    assert 'corn' in answers[0] and 'yellow' in answers[1] and 'green' in answers[1]
    assert 'carrot' in answers[2] and 'orange' in answers[2]
    assert [t['trace']['image_encoded_count'] for t in a['turns']]==[1,0,1]
    for t in a['turns']:
        trace=t['trace']
        assert trace['all_cache_offsets_equal_frontier'] and trace['cache_layer_count']==40
        assert trace['prompt_replay_count']==trace['full_cache_repack_count']==0
    assert a['changed_image_rejected_without_mutation'] and a['cancelled_tokens']
    for key in ('cancel_trace','encoding_cancel_trace','prefill_cancel_trace'):
        assert a[key]['cancelled'] and a[key]['cleanup_called']
    assert a['discarded_prefills']==[dict(burned_layers=40,certificate_detached=True)]
    assert a['stateful_cancel_trace']['cancelled'] and a['stateful_cancel_trace']['ok']
    assert a['stateful_cancel_record']['state']=='idle' and a['stateful_cancel_record']['request_count']==1
    assert a['stateful_cancel_record']['last_turn'] is not None
    assert b['restore_exact'] and not b['no_image_reencode_calls']
    assert len(a['saved_slots'])==len(a['probe_slots'])==len(b['probe_slots'])==280
    assert a['probe_tokens']==b['probe_tokens'] and a['probe_slots']==b['probe_slots']
    assert 'yes' in b['restored_additional_image_turn']['response_json']['choices'][0]['message']['content'].lower()
    assert b['restored_additional_image_trace']['image_encoded_count']==1
    assert b['restored_additional_image_trace']['all_cache_offsets_equal_frontier']
    assert b['restored_additional_image_trace']['prompt_replay_count']==b['restored_additional_image_trace']['full_cache_repack_count']==0
    maximum=a['maximum_matrix']
    assert maximum['initial_prompt_tokens']==8160 and maximum['consumed_frontier']==8192
    assert len(maximum['spans'])==4 and max(s['length'] for s in maximum['spans'])==1014
    assert maximum['diagnostics']['m8']['total_prompt_replay_count']==maximum['diagnostics']['m8']['total_full_cache_repack_count']==0
    for variant,span,oracle in zip(maximum['variants'],maximum['spans'],reference['maximum_preprocessing'],strict=True):
        for key in ('size','format','sha256'):assert variant[key]==oracle[key]
        assert span['grid']==oracle['grid'] and span['length']==oracle['tokens']
        assert oracle['pixels_exact'] and oracle['layout_exact']
    for image in reference['images']:
        assert image['pixels_exact']
        for key in ('fp32_tower','fp32_aligner'):
            assert image[key]['rms_relative']<.0002 and image[key]['cosine']>.99999
        for key in ('tower','aligner'):
            assert image[key]['rms_relative']<.1 and image[key]['cosine']>.995
    text=a['text_trace']
    assert text['production_prefill_selector']=='DENSE_P0_P7' and text['cleanup_called']
    assert text['prompt_replay_count']==text['full_cache_repack_count']==0
    tests=(EVIDENCE/'affected-tests.txt').read_text()
    assert 'failed' not in tests.lower() and re.search(r'\b\d+ passed',tests)
    result=dict(schema='ds41f.multimodal.closeout.v1',
        decision='QUALIFIED_BOUNDED_FIRST_PARTY_OFF_MULTIMODAL_4_IMAGES_8192',
        source_sha256=a['source_sha256'],
        receipts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (EVIDENCE/'initial.json',EVIDENCE/'restored.json',EVIDENCE/'official-reference.json',EVIDENCE/'affected-tests.txt')},
        max_images=4,max_source_pixels=4194304,max_expanded_image_positions=1014,max_consumed_positions=8192,
        initial_turn_timings=[t['trace'] for t in a['turns']],
        image_stream_timing=a['stateless_image_stream_timing'],
        maximum_matrix=dict(preprocessing_seconds=maximum['preprocessing_seconds'],elapsed_seconds=maximum['elapsed_seconds'],
                            trace=maximum['trace'],memory=maximum['memory']),
        loaded_memory=a['loaded_memory'],closed_memory=a['closed_memory'],
        restore_canonical_exact=True,restore_image_reencode_count=0,
        resource_policy_restored_exactly=True,
        exclusions=['CUDA/full-LM logit parity','full visual quality benchmark','multimodal beyond measured envelope',
                    'low detail','remote/file images','multimodal Responses/Messages','batching/concurrency','active/cross-backend restore',
                    'automatic protocol-partial recovery','MTP/DSpark promotion','full R1','release/packaging/promotion'])
    (EVIDENCE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['decision'])


if __name__=='__main__':main()
