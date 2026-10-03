"""Durable M32 gate boundaries; live binaries are tested by qualification tools."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/m32'

def load(name):
    return json.loads((OUT / f'{name}.json').read_text())

def test_real_full_binding_identity_and_patch():
    identity = load('runtime-identities')
    gate = load('qualification')
    assert gate['build_used'] == 'FULL_STANDARD_NATIVE_BUILD'
    assert not gate['protocol_only_feature_gate']
    assert identity['native_architecture'] == 'arm64'
    assert identity['opencv_version'] == '4.14.0'
    assert identity['recipe_lock_sha256'] == identity['base_lock_sha256']
    assert identity['recipe_patch_sha256'] == hashlib.sha256((OUT/'recipe-semantic-preview.patch').read_bytes()).hexdigest()
    assert identity['preserved_release_unchanged']
    assert identity['omlx_patch_revision'] is None
    assert identity['omlx_tool_import_ok']
    assert identity['native_sha256'] != identity['preserved_release_native_sha256']
    for name in ['canonical-patched', 'native-preview', 'init-boundary']:
        assert load(name)['native_sha256'] == identity['native_sha256']

def test_native_canonical_and_preview_parity():
    parity = load('canonical-parity')
    assert parity['equal'] and parity['count'] == 64 and parity['case_count'] == 22
    assert all(c['equal'] for c in parity['comparisons'])
    assert load('canonical-base')['records'] == load('canonical-patched')['records']
    preview = load('native-preview')
    assert preview['source_agreement'] and preview['nonmutation'] and preview['deterministic']
    assert preview['row_count'] == 77
    for row in preview['rows']:
        assert row['before'][2] == len(row['before'][1])
        assert row['after'][2] == len(row['after'][1])
        assert row['preview']['terminal'] == row['canonical_terminal']

def test_actual_init_commit_edge_is_not_chain_qualification():
    probe = load('init-boundary')
    assert probe['status'] == 'CHAIN_ONLY_CLAMP_INSUFFICIENT_AT_INIT'
    assert probe['chain_verify_calls'] == 0
    assert probe['prediction']['safe'] == probe['prediction']['index'] == 0
    before, after = probe['before_init'], probe['after_init_before_emission']
    assert before['history'] == after['history']
    assert before['target'] == [before['history']]*40
    assert after['target'] == [before['history']+1]*40
    assert after['dspark'] == [before['history']+1]*3
    assert before['parser'] == after['parser']
    assert after['queue'][0][0] == probe['first_model_token']
    assert probe['canonical_observation_matches']
    assert not probe['required_unforwarded_terminal_invariant']

def test_blocked_gate_keeps_production_off_and_no_soak_authorization():
    gate = load('qualification')
    assert gate['decision'] == 'MTP_SEMANTIC_CLAMP_BLOCKED'
    assert gate['native_binding_gate_passed']
    assert not gate['full_protocol_gate_passed']
    assert not gate['mtp_semantic_clamp_implemented']
    assert not gate['terminal_queue_metadata_implemented']
    assert gate['model_replay_repack_proof'] is None
    assert gate['semantic_quiescence_counters'] is None
    assert gate['guarded_mtp_throughput'] is None
    assert gate['production_mtp'] == 'OFF'
    assert gate['public_mtp_option'] == 'disabled'
    assert gate['mtp_persistence'] == 'fail closed'
    assert not gate['m33_authorized']
    assert gate['operational_soak'] == 'NOT RUN'
    assert all(c['status'] == 'BLOCKED_SEMANTIC_COMMIT_BOUNDARY' for c in gate['live_qualification'].values())
