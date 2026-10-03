"""Current M36 blocker evidence, not inherited M35 source qualification."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/m36'


def test_m36_stays_blocked_without_protocol_reconstruction_certificate():
    q = json.loads((OUT / 'qualification.json').read_text())
    assert q['decision'] == 'BLOCKED_CANONICAL_PROTOCOL_REPRESENTABILITY'
    assert not q['contract']['established']
    assert not q['contract']['public_api_added']
    assert not q['tools']['exactly_once_agent_qualification']
    assert q['blocker']['ordinary_conversion']['succeeded']
    assert not q['blocker']['ordinary_conversion']['exact_prefix']
    assert q['reconciliation']['tool_poisoned']
    assert q['reconciliation']['text_exact_prefix']
    assert q['reconciliation']['original_request_retry_mutation_atomic']
    assert q['backpressure']['max_send_wait_s'] > 1
    assert q['backpressure']['plateau_responses'] < q['backpressure']['response_limit']
    assert q['gates']['real_m35_http_turns'] == 18
    assert q['gates']['real_native_interruption_cases'] == 16
    for scope in ['probe', 'm35_rerun']:
        assert q['replay_repack'][scope]['replay'] == q['replay_repack'][scope]['repack'] == 0


def test_current_sources_and_raw_evidence_match():
    q = json.loads((OUT / 'qualification.json').read_text())
    for name, digest in q['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for name, digest in q['evidence_hashes'].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
