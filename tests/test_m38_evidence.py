"""Current-source M38 receipts; a finite passing run must not erase the blocker."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/m38'
def load(name):return json.loads((OUT/name).read_text())


def test_current_receipt_hashes_and_nonpromotion():
    q=load('qualification.json')
    assert q['decision']=='BLOCKED_OPERATIONAL_LIFETIME_RETENTION'
    assert q['bounded_workload']=='PASS_BOUNDED_WORKLOAD' and not q['release_promoted']
    for group in ['source_hashes','evidence_hashes','historical_evidence_hashes']:
        for path,digest in q[group].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    assert not q['blocker_fixed'] and len(q['defects_fixed'])==3


def test_living_client_end_of_turn_and_phase_authority():
    w=load('soak.json');assert w['status']=='PASS_BOUNDED_WORKLOAD'
    assert len(w['turns'])==76 and len(w['effects'])==len(w['ledger'])==14
    assert len(w['phases'])==3
    for p in range(3):
        turns=[t for t in w['turns'] if t['phase']==p]
        assert [t['identity']['sequence'] for t in turns]==list(range(1,len(turns)+1))
        assert len({t['identity']['session_id'] for t in turns})==1
    for t in w['turns']:
        raw=t['identity']['body_utf8'].encode();out=t['outcome'];request=json.loads(raw)
        assert hashlib.sha256(raw).hexdigest()==t['identity']['body_sha256']
        assert out['sequence']==t['identity']['sequence']
        expected=request['messages']
        if out['outcome_state']=='recoverable':
            assert all(out['certificate'][k] is True for k in ['representable','exact_prefix','semantic_complete'])
            expected=expected+[out['response']['choices'][0]['message']]
        else:
            assert out['outcome_state']=='unrecoverable' and out['certificate']['representable'] is False
        assert t['transcript']==expected
        trace=t['trace']
        assert trace['prompt_replay']==trace['full_cache_repack']==0
        assert len(trace['target_offsets'])==40 and len(trace['dspark_offsets'])==3
        assert set(trace['target_offsets']+trace['dspark_offsets'])=={trace['canonical_frontier']}
        assert trace['queue_empty'] and trace['prediction_retired']
        assert all(trace['quiescence']['counters'][k]==0 for k in ['history_replay','full_cache_repack','new_verify_cycles','new_proposals'])
        assert t['resources']['timings']<=128 and t['resources']['server_trace_length']<=32
    assert w['counts']['fresh_target_allocations']==3
    assert w['counts']['replay']==w['counts']['repack']==0
    assert len({tuple(e['key']) for e in w['ledger']})==14
    assert all(e['status']=='completed' for e in w['ledger'])


def test_real_transport_loss_and_pressure_and_synthetic_ledger_capacity():
    life=load('lifecycle-transport.json');assert life['status']=='PASS'
    assert [r['action'] for r in life['rows']]==['delete','create']
    assert all(r['error']['type']=='IncompleteRead' and r['client_state']=='stopped' and not r['retry_or_replacement'] for r in life['rows'])
    pressure=load('client-pressure.json');assert pressure['status']=='PASS'
    assert pressure['client_final_state']=='retired' and len(pressure['traces'])==2
    assert max(s.get('wait_s',0) for s in pressure['sends'])>1
    assert pressure['samples'][-1]['generated']==pressure['samples'][-10]['generated']
    assert pressure['counts']['replay']==pressure['counts']['repack']==0
    ledger=load('ledger-exhaustion.json')
    assert ledger['effects']==ledger['capacity']==len(ledger['ledger'])==128
    assert ledger['overflow_effects']==0 and ledger['no_eviction'] and ledger['results_unchanged']
    assert ledger['state']=='stopped' and not ledger['cross_session_alias']
    assert [e['result']['content'] for e in ledger['ledger']]==[f'result-{i}' for i in range(128)]
    assert all(s['timings']==128 for s in ledger['samples'])


def test_retirement_retention_is_open_not_hidden_by_trace_bound():
    w=load('soak.json');r=load('retention-probe.json');q=load('qualification.json')
    assert r['finding']=='NO_LIFETIME_SESSION_CAP'
    assert [s['retained_sessions'] for s in r['samples']]==list(range(32,257,32))
    assert [s['cycles'] for s in r['samples']]==list(range(32,257,32))
    assert [p['resources']['closed_sessions'] for p in w['phases']]==[1,2,3]
    assert [p['resources']['server_trace_length'] for p in w['phases']]==[25,32,32]
    assert w['phases'][-1]['resources']['server_session_json_bytes']>w['phases'][0]['resources']['server_session_json_bytes']
    assert q['resources']['open_blocker'] and q['recommended_next'] and q['unsupported']
    assert len(q['gates']['commands'])==7 and all(r['returncode']==0 for r in q['gates']['commands'])
    assert len(q['gates']['native'])==6 and all(r['returncode']==0 for r in q['gates']['native'])
