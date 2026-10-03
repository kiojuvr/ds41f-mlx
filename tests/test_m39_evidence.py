"""Canonical M39 evidence consistency, not a substitute for runtime gates."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/m39'

def test_m39_canonical_decision_sources_and_raw_evidence():
    d=json.loads((OUT/'qualification.json').read_text())
    assert d['decision']=='QUALIFIED_FINITE_PROCESS_LIFETIME_ADMISSION'
    assert all(d['checks'].values())
    for p,h in d['sources'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for p,h in d['evidence'].items():assert hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h,p


def test_m39_retirement_is_not_a_diagnostic_tombstone_archive():
    d=json.loads((OUT/'qualification.json').read_text());b=d['budgets']
    assert b['live_sessions']==b['active_requests']==1
    assert b['retired_authority_records']==b['retired_outcome_payload_slots']==0
    assert b['diagnostic_records']==16 and d['lifetimes']>=20000
    for s in d['structural_samples']:
        assert s['identity_storage_bytes']<=117
        assert s['diagnostic_records']<=16 and s['closed_outcome_payload_bytes']==0
        assert s['live_records']==s['native_owners']==s['retired_authority_records']==0
        assert s['session_trace_limit']==s['trace_limit']==32
    for r in d['checkpoint_resources']:
        assert r['live_records']==r['native_owners']==r['retired_authority_records']==0
        assert r['mlx_cache_bytes']==0


def test_m39_finite_no_wrap_and_no_release_promotion():
    d=json.loads((OUT/'qualification.json').read_text())
    assert int(d['budgets']['lifetimes_max'])==2**128-1
    assert d['budgets']['request_sequence_bits']==64
    assert 'public/default/release MTP' in d['unsupported']
    assert 'release/admission/provenance' in d['next_task']
