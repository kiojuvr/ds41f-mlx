"""Immutable M33 evidence checks; no model loading or import-path side effects."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/m33'
def load(name):return json.loads((OUT/(name+'.json')).read_text())

def test_gate_and_release_policies():
    q=load('qualification')
    assert q['decision']=='PROTOCOL_GATE_SOLVED'
    assert q['binding']=='FULL_BINDING_QUALIFIED'
    assert q['authorized_next']=='M34 operational qualification only'
    assert q['production_mtp_default']=='OFF' and q['public_mtp_option']=='disabled'
    assert q['mtp_persistence']=='fail closed' and q['token_exact_immediate_abort']=='unsupported'
    assert not q['public_serving_promotion'] and not q['broad_operational_soak']

def test_candidate_identity_and_patch():
    i=load('runtime-identities')
    assert i['ds41f_base_commit']=='fff84bdd1acd814ae84b994bec4ed110f31f4bb1'
    assert i['recipe_candidate']=='29dabb5a55b7b2c6a68e18bbb3eb14495623e81a'
    assert i['omlx_candidate']=='fbe18e8fe68e5bb7b9b1971652ed330f752b6afc'
    assert i['recipe_native_sha256']=='454413afdcee2916795e1c5f7ce1b94a8346e76bffd6b45024f28f69f8d0a73f'
    assert i['DOCS_RS'] is None and i['preserved_release_unchanged']
    assert len(i['preserved_release_cpp_artifacts'])==16
    assert hashlib.sha256((OUT/'omlx-semantic-horizon.patch').read_bytes()).hexdigest()==i['omlx_patch_sha256']
    assert len(i['omlx_source_hashes'])==4

def test_native_parity_and_nonmutation():
    p=load('canonical-parity');n=load('native-preview')
    assert p['equal'] and p['comparisons']==64 and p['cases']==22
    assert n['row_count']==77 and n['source_agreement'] and n['nonmutation'] and n['deterministic']
    assert n['preview_latency']['samples']==15400
    assert load('source-preview-clone')['clone_latency']['samples']==15400

def test_real_semantics_and_cross_cycle_closure():
    tools=load('tools-live');stops=load('stops-live')
    assert tools['status']==stops['status']=='PASS'
    for c in tools['cases']:
        assert c['protocol']['choices'][0]['finish_reason']=='tool_calls'
        if c['mtp']:
            assert c['prediction_matches'][0]['identity'][0]=='DSML_TOOL_CALL_BLOCK_END'
            assert c['quiescence_actual_counter_delta']['verify']==c['quiescence_actual_counter_delta']['proposal']==0
    multi=next(c for c in tools['cases'] if c['name']=='multiple_tools' and c['mtp'])
    assert any(r['terminal_kind'] and r['start']<r['source_bytes_before'] for r in multi['preview']['rows'])
    assert all(c['prediction_matches'][0]['identity'][0]=='STOP_SEQUENCE' for c in stops['cases'])
    text=next(c for c in load('init-live')['cases'] if c['name']=='text' and c['mtp'])
    assert not text['prediction_matches']  # EOS is suppressed backend control, not recipe text.
    assert text['protocol']['choices'][0]['message']['content']=='Hello there!'

def test_interruptions_and_real_tool_reentry():
    x=load('interruptions');r=load('tool-reentry')
    assert x['status']==r['status']=='PASS' and len(x['cases'])==16
    assert sum(c['status']=='PASS_FAIL_CLOSED' for c in x['cases'])==3
    late=next(c for c in x['cases'] if c['name']=='canonical_terminal_transport_pending')
    assert late['delivery_ack']['metadata_only'] and not any(late['delivery_ack']['counter_delta'].values())
    assert len(late['quiescence']['recovery_suffix_tokens'])==1
    assert r['counts']['replay']==r['counts']['repack']==0 and len(r['turns'])==2
    assert r['turns'][0]['protocol']['choices'][0]['finish_reason']=='tool_calls'
    assert r['turns'][1]['protocol']['choices'][0]['finish_reason']=='stop'
    assert all(s['P7_events'] and s['full_cache_repack']==0 for s in r['suffixes'])

def test_performance_and_manifest():
    p=load('performance');m=load('metadata-overhead')
    assert p['status']=='PASS' and p['M26_class_materially_intact']
    plain=next(r for r in p['runs'] if r['mode']=='ON_no_guard')
    guarded=next(r for r in p['runs'] if r['mode']=='ON_guarded')
    assert plain['tokens']==guarded['tokens'] and plain['stats']['accepts']==guarded['stats']['accepts']
    assert p['guard_vs_no_guard_ratio']>.9 and guarded['tok_s']>34
    assert m['current_owned_terminal_limit']==1 and m['decision_history_limit']==16
    for path,digest in load('file-manifest')['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
