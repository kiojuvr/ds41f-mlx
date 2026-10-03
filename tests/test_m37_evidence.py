"""Canonical current-source M37 evidence gates (no checkpoint needed)."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/m37'
def load(name):return json.loads((OUT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_source_and_evidence_hashes():
    q=load('qualification.json')
    assert q['decision']=='QUALIFIED_BOUNDED_INTERNAL_LOCAL_CLIENT'
    for group in ['source_hashes','evidence_hashes','historical_evidence_hashes']:
        for path,digest in q[group].items():assert sha(ROOT/path)==digest,path
    for path,digest in load('workflow.json')['sources'].items():assert sha(ROOT/path)==digest,path


def test_coherent_transcript_fencing_and_effects():
    w=load('workflow.json');assert w['status']=='PASS'
    assert [t['identity']['sequence'] for t in w['turns']]==[1,2,3,4,5,6,7,1]
    assert len({t['identity']['session_id'] for t in w['turns']})==2
    for turn in w['turns']:
        i=turn['identity'];raw=i['body_utf8'].encode();request=json.loads(raw)
        assert hashlib.sha256(raw).hexdigest()==i['body_sha256']
        out=turn['outcome'];assert out['sequence']==i['sequence']
        if out['outcome_state']=='recoverable':
            assert out['certificate']['representable'] is True
            assert turn['transcript']==request['messages']+[out['response']['choices'][0]['message']]
        else:
            assert out['outcome_state']=='unrecoverable'
            assert out['certificate']['representable'] is False
            assert turn['transcript']==request['messages']
    assert len(w['effects'])==len(w['ledger'])==2
    keys=[tuple(e['key']) for e in w['ledger']];assert len(set(keys))==2
    assert all(e['status']=='completed' and e['result']['role']=='tool' for e in w['ledger'])
    assert [a['observations'] for a in w['actions'] if 'observations' in a]==[4,4]
    expired=next(a for a in w['actions'] if a['name']=='expired-outcome')
    assert expired['result']['outcome_state']=='expired' and expired['no_generation']
    restart=next(a for a in w['actions'] if a['name']=='delete-fresh')
    assert restart['old_id_and_request_denied'] and not restart['native_state_carried']
    fresh=json.loads(w['turns'][-1]['identity']['body_utf8'])
    assert fresh['messages'][:-1]==restart['preserved_application_messages']
    assert 'tools' not in fresh and fresh['stream'] is False
    assert w['turns'][-1]['state']=='ready'


def test_native_no_replay_repack_and_fresh_prefill_accounting():
    w=load('workflow.json');q=load('qualification.json')
    assert w['counts']['replay']==w['counts']['repack']==0
    assert w['counts']['fresh_target_allocations']==2
    for t in w['traces']:
        assert t['prompt_replay']==t['full_cache_repack']==0
        assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
        assert set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']}
        assert t['queue_empty'] and t['prediction_retired']
        assert all(t['quiescence']['counters'][k]==0 for k in ['history_replay','full_cache_repack','new_verify_cycles','new_proposals'])
    assert q['replay_repack']['post_delete_new_session_prefill_turns']==1
    assert len(q['observations']['ledger_only_s'])==6
    assert len(q['observations']['latencies'])==8
    assert q['unsupported'] and 'release' in q['next_milestone']


def test_current_native_and_previous_contract_gates():
    commands=load('native-gate-commands.json')
    assert len(commands)==6 and all(c['returncode']==0 for c in commands)
    assert load('m35-http.json')['status']=='PASS' and len(load('m35-http.json')['cases'])==18
    assert load('m36-probe.json')['status']=='PROBE_COMPLETE'
    assert load('m36r-http.json')['status']=='PASS'
    assert load('native-interruptions.json')['status']=='PASS' and len(load('native-interruptions.json')['cases'])==16
    assert len(load('recipe-matrix.json')['rows'])==28
