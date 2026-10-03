"""Fail-closed M35 evidence aggregation; never rewrite M33/M34 evidence."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/m35'
def read(name): return json.loads((OUT/name).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
http = read('http.json')
identity = read('identities.json')
interruptions = read('native-interruptions.json')
assert http['status'] == identity['status'] == interruptions['status'] == 'PASS'
assert all(sha(ROOT/p) == digest for p,digest in http['sources'].items())
assert http['counts']['replay'] == http['counts']['repack'] == 0
assert http['counts']['fresh_target_allocations'] == 11
cases = http['cases']
byname = {c['name']: c for c in cases}
required = ['nonstream', 'nonstream_continuation', 'tool_sse', 'tool_result_reentry',
            'semantic_terminal_disconnect', 'terminal_tool_result_reentry', 'normal_sse',
            'slow_sse', 'disconnect_long', 'disconnect_long_continuation',
            'undelivered_suffix', 'undelivered_suffix_continuation',
            'terminal_disconnect', 'terminal_disconnect_continuation',
            'rust_iterator_drop', 'rust_drop_continuation']
assert all(n in byname for n in required)
for c in cases:
    r = c['runtime']
    assert r['prompt_replay'] == r['full_cache_repack'] == 0
    assert r['client_acknowledged_ordinal'] == 0
    assert r['queue_empty'] and r['prediction_retired']
    assert len(r['target_offsets']) == 40
    assert set(r['target_offsets']+r['dspark_offsets']) == {r['canonical_frontier']}
    assert all(r['quiescence']['counters'][k] == 0 for k in ['new_verify_cycles','new_proposals','history_replay','full_cache_repack'])
    events = c['transport'].get('received_events')
    if events is None:
        events = [json.loads(e['data']) for e in c['transport'].get('rows',[]) if e['data'] != '[DONE]']
    assert events == r['protocol_events'][:len(events)]
for n in ['disconnect_long', 'rust_iterator_drop']:
    r = byname[n]['runtime']
    assert r['cancelled'] and r['generated'] < 100
    assert r['quiescence']['canonical_frontier'] == r['canonical_frontier']
assert byname['semantic_terminal_disconnect']['runtime']['terminal_matches']
assert not byname['undelivered_suffix']['transport']['rows']
assert byname['undelivered_suffix']['runtime']['canonical_generated']
normal = byname['normal_sse']['runtime']
direct = http['direct_control']
assert normal['canonical_generated'] == direct['canonical_generated']
assert normal['response']['choices'] == direct['response']['choices']
assert direct['prompt_replay'] == direct['full_cache_repack'] == 0
model_ratio = (normal['generated']/normal['decode_s'])/(direct['generated']/direct['decode_s'])
assert .90 < model_ratio < 1.10
performance=[]
for c in cases:
    r=c['runtime'];t=c['transport'];s=r.get('mtp_stats');den=sum(s['depth_drafted']) if s else 0
    rows=[e for e in t.get('rows',[]) if e['data']!='[DONE]']
    # Residual is dispatch/scheduling/protocol-loop work, not a pure profiler.
    residual=r['elapsed_s']-r['load_s']-r['prefill_handoff_s']-r['decode_s']-r['cleanup_s']-r['formatting_s']-r['delivery_wait_s']
    performance.append(dict(case=c['name'],responses=r['generated'],model_phase_tok_s=r['generated']/r['decode_s'],
        model_phase_s=r['decode_s'],model_load_s=r['load_s'],prefill_handoff_s=r['prefill_handoff_s'],
        first_canonical_from_admission_s=r['first_canonical_s'],first_iterator_yield_s=r['first_chunk_s'],
        first_client_event_s=rows[0]['received_s'] if rows else None,formatting_s=r['formatting_s'],
        controller_delivery_delay_s=r['delivery_wait_s'],cleanup_s=r['cleanup_s'],
        residual_dispatch_loop_s=residual,client_wall_s=t['client_wall_s'],
        idle_observation_wait_s=c['post_disconnect_idle_wait_s'],
        acceptance=None if not den else s['accepts']/den,
        acceptance_scope='unchanged native considered-draft metric; completed-owner stats only',
        max_observed_client_lag_input_watermarks=max([e['server_canonical_at_receive']-e['canonical_input_upper_bound'] for e in rows] or [0])))
logs=['regressions.log','lease-tests.log','horizon-regressions.log','off-policy-regressions.log','release-off-regressions.log','rust-regressions.log']
for name in logs:
    text=(OUT/name).read_text()
    assert 'passed' in text or 'test result: ok' in text
    assert 'FAILED' not in text
sources=list(http['sources'])+['tools/run_m35_native_interruptions.py','tools/record_m35_evidence.py',
    'tests/test_m35_transport_lease.py','tests/test_m35_http_evidence.py',
    'rust/ds41f_api/src/bin/m35_internal_transport.rs','rust/ds41f_api/src/lib.rs',
    'Cargo.toml','Cargo.lock','rust/ds41f_api/Cargo.toml','.gitignore',
    'docs/milestone-35-http-sse-qualification.md',
    'docs/README.md','docs/implementation-plan.md','docs/architecture.md','docs/qualification.md',
    'docs/session-state.md','docs/doc-classification.json']
result=dict(schema='ds41f.m35.qualification.v1',decision='HTTP_SSE_QUALIFIED_BOUNDED_INTERNAL_SINGLETON',
    base_commit=http['base_commit'],final_commit_resolver='git log -1 --format=%H -- artifacts/m35/qualification.json',
    identities=identity,server=http['server'],
    transport_identities={name:metadata.version(name) for name in ['uvicorn','fastapi','starlette','anyio']},
    rust_binary_sha256=sha(ROOT/'target/debug/m35_internal_transport'),
    cases=required,total_http_turns=len(cases),
    returned_responses=sum(c['runtime']['generated'] for c in cases),counters=http['counts'],
    admission=http['admission'],native_interruption_cases=len(interruptions['cases']),
    performance=performance,direct_same_request_control=dict(responses=direct['generated'],
        model_phase_tok_s=direct['generated']/direct['decode_s'],http_model_phase_ratio=model_ratio,
        direct_elapsed_s=direct['elapsed_s'],http_elapsed_s=normal['elapsed_s'],identical_tokens=True),
    frontier_contract=dict(canonical='M33/M34 emitted/drained inputs; never a socket-write frontier',
        client='received recipe event prefix with canonical input upper-bound watermark; not a token-exact delivery claim',
        acknowledgement='no application ACK protocol; runtime delivered-token metadata remains ordinal zero for the turn',
        recovery='GET internal canonical response, ordinary recipe exact-prefix continuation; no prompt replay or cache reconstruction'),
    defects=[dict(name='shielded-loop cancellation starvation',evidence='cancellation-checkpoint-defect.json',
        fix='unshielded AnyIO checkpoints between synchronous canonical boundaries; shield only native mutation and cleanup'),
        dict(name='protocol chunk transfer / double-settlement failure',evidence='chunk-transfer-error.json',
        fix='serialize native chunks before response.append consumes them; retire native runtime ownership exactly once before protocol accumulation; protocol faults poison session'),
        dict(name='unstarted iterator admission lease',evidence='lease-tests.log',scope='synthetic response-header send failure',
        fix='response-level cleanup owns pre-iteration reservation too; exact per-response lease retirement cannot release a subsequent request; DELETE owns the same lease; retirement faults release admission but poison singleton reuse')],
    attempts=dict(pilot='pre-fix exploratory transport, not final qualification',
        pre_final='pre-checkpoint exploratory transport, not final qualification',
        pre_lease_hardening='post-checkpoint pilot before exact per-response lease hardening; excluded from final decision',
        pre_retirement_hardening='pilot before failed-retirement lease cleanup; excluded from final decision',
        pre_delete_hardening='pilot before serialized DELETE ownership; excluded from final decision',
        harness_name_error='client helper shadowed http import; pre-model failure',
        chunk_transfer_error='real native transfer defect; secondary retry hid primary error',
        cancellation_checkpoint_defect='real ordinary and Rust disconnect ignored until completion; excluded'),
    unsupported=['public/default/release MTP','persistence/restore/process restart','token-exact immediate abort',
        'concurrent/shared generation','batching','200K/long-context HTTP MTP','distributed serving',
        'portable native wheels','unbounded socket saturation','arbitrary partial Unicode/stop-string client recovery'],
    next_milestone='M36: client/agent canonical recovery and admission qualification, including pending-byte recovery and larger real-socket backpressure; not automatic release promotion',
    source_hashes={p:sha(ROOT/p) for p in sources},
    evidence_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['qualification.json','record.log','evidence-tests.log']})
(OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['decision','total_http_turns','returned_responses','counters','direct_same_request_control']},indent=2))
