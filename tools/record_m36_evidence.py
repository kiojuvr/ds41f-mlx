"""Record the M36 BLOCKED decision without promoting exploratory observations."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/m36'
def load(name): return json.loads((OUT / name).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p = load('probe.json')
h = load('m35-http.json')
n = load('native-interruptions.json')
assert p['status'] == 'PROBE_COMPLETE'
assert h['status'] == n['status'] == 'PASS'
for name, digest in p['sources'].items(): assert sha(ROOT / name) == digest
for name, digest in h['sources'].items(): assert sha(ROOT / name) == digest
by = {c['name']: c for c in p['cases']}
tool, text, pressure = (by[k] for k in ['partial-tool', 'utf8-byte', 'socket-pressure'])
assert tool['recovered']['state'] == 'poisoned'
assert tool['ordinary_conversion']['succeeded'] and not tool['ordinary_conversion']['exact_prefix']
assert tool['continuation_status'] == 400 and tool['rejection_mutation_atomic']
assert tool['observed']['tool_executions'] == 0
assert tool['observed']['events'] == tool['recovered']['last_turn']['protocol_events'][:len(tool['observed']['events'])]
raw = bytes.fromhex(text['observed']['bytes_hex'])
assert raw[-1] >= 0xc0
complete_lines = raw.split(b'\n')[:-1]
complete_events = [json.loads(line[6:]) for line in complete_lines if line.startswith(b'data: ')]
assert complete_events == text['recovered']['last_turn']['protocol_events'][:len(complete_events)]
assert text['observed']['incomplete_utf8'] and text['ordinary_conversion']['exact_prefix']
assert text['continuation_status'] == 200 and text['repeated_get_idempotent']
assert pressure['recovered']['state'] == 'idle'
assert max(s['wait_s'] for s in pressure['sends']) > 1
assert pressure['samples'][-1]['generated'] == pressure['samples'][-10]['generated']
for c in [tool, text]:
    assert c['duplicate_retry']['status'] == 400 and c['duplicate_retry']['mutation_atomic']
    assert c['repeated_get_idempotent']
traces = [c['recovered']['last_turn'] for c in p['cases']]
traces += [text['after']['last_turn']]
traces += [c['runtime'] for c in h['cases']] + [h['direct_control']]
for t in traces:
    assert t['prompt_replay'] == t['full_cache_repack'] == 0
    assert t['queue_empty'] and t['prediction_retired']
    assert len(t['target_offsets']) == 40 and len(t['dspark_offsets']) == 3
    assert set(t['target_offsets'] + t['dspark_offsets']) == {t['canonical_frontier']}
    assert all(t['quiescence']['counters'][k] == 0 for k in ['new_verify_cycles','new_proposals','history_replay','full_cache_repack'])
assert p['counts']['replay'] == p['counts']['repack'] == h['counts']['replay'] == h['counts']['repack'] == 0
for c in h['cases']:
    observed = c['transport'].get('received_events')
    if observed is None:
        observed = [json.loads(e['data']) for e in c['transport'].get('rows', []) if e['data'] != '[DONE]']
    assert observed == c['runtime']['protocol_events'][:len(observed)]
assert len(h['cases']) == 18
normal = next(c['runtime'] for c in h['cases'] if c['name'] == 'normal_sse')
assert normal['canonical_generated'] == h['direct_control']['canonical_generated']
model_ratio = (normal['generated']/normal['decode_s']) / (h['direct_control']['generated']/h['direct_control']['decode_s'])
assert .90 < model_ratio < 1.10
assert h['counts']['fresh_target_allocations'] == 11
assert len(n['cases']) == 16
logs = ['runtime-regressions.log', 'horizon-regressions.log', 'off-policy-regressions.log',
        'release-off-regressions.log', 'rust-regressions.log', 'prior-evidence-regressions.log']
for name in logs:
    log = (OUT / name).read_text()
    assert 'passed' in log or 'test result: ok' in log
    assert 'FAILED' not in log
identity = json.loads((OUT / 'identities.log').read_text())
assert identity['status'] == 'PASS' and all(identity['checks'].values())
source_paths = list(p['sources']) + ['tools/record_m36_evidence.py',
    'tests/test_m36_recovery_boundary.py', 'tests/test_m36_evidence.py',
    'docs/milestone-36-client-recovery-qualification.md', 'docs/README.md',
    'docs/doc-classification.json', 'docs/implementation-plan.md',
    'docs/qualification.md', 'docs/session-state.md']
resource = {}
for key in ['rss_kib', 'mlx_active_bytes', 'mlx_cache_bytes', 'generated']:
    samples = [s[key] for s in pressure['samples']]
    resource[key] = dict(min=min(samples), max=max(samples), final=samples[-1])
performance=[]
for c in p['cases']:
    t=c['recovered']['last_turn']
    performance.append(dict(case=c['name'], model_load_s=t['load_s'],
        prefill_handoff_s=t['prefill_handoff_s'], canonical_model_phase_s=t['decode_s'],
        formatting_s=t['formatting_s'], cleanup_s=t['cleanup_s'], response_elapsed_s=t['elapsed_s'],
        recovery_poll_observation_s=c['recovery_wait_s'],
        recovery_timing_scope='client poll-to-settled observation, not a pure lookup microbenchmark'))
next_turn=text['after']['last_turn']
result=dict(schema='ds41f.m36.qualification.v1',decision='BLOCKED_CANONICAL_PROTOCOL_REPRESENTABILITY',
    base_commit=p['base_commit'],final_commit_resolver='git log -1 --format=%H -- artifacts/m36/qualification.json',
    identities=identity,transport=p['transport'],
    runtime_correction=dict(path='ds41f_mlx/serving/internal_mtp.py',
        base_sha256=hashlib.sha256(subprocess.check_output(['git','show',p['base_commit']+':ds41f_mlx/serving/internal_mtp.py'])).hexdigest(),
        current_sha256=sha(ROOT/'ds41f_mlx/serving/internal_mtp.py'),patch='runtime-correction.patch'),
    blocker=dict(canonical_frontier=tool['recovered']['last_turn']['canonical_frontier'],
        reconstructed_message=tool['recovered']['last_turn']['response']['choices'][0]['message'],
        ordinary_conversion=tool['ordinary_conversion'],
        explanation='native idle coherence does not imply an ordinary reconstructable/executable assistant result; incomplete canonical DSML re-encodes non-prefix'),
    contract=dict(established=False,public_api_added=False,rust_api_changed=False,
        positive='discard local partial transport bytes; use settled canonical recipe response; exact-prefix continuation in exercised text case',
        fail_closed='unfinished canonical tool deltas poison; busy remains visible until lease retirement completes',
        unqualified=['four-way request identity/outcome fence','never-committed ambiguous request discovery',
            'arbitrary decoder/parser-pending ordinary reconstruction','complete M36 client tool execution ledger']),
    partial_delivery=dict(utf8_byte_prefix=True,incomplete_json_sse_event=True,
        canonical_incomplete_dsml_tool=True,arbitrary_pending_recipe_recovery_qualified=False),
    tools=dict(negative_probe_executions=0,incomplete_tool_is_execution_permission=False,
        completed_tool_m35_gate_rerun=True,exactly_once_agent_qualification=False),
    reconciliation=dict(text_exact_prefix=True,text_frontiers=[text['recovered']['canonical_frontier'],text['after']['canonical_frontier']],
        tool_exact_prefix=False,tool_poisoned=True,repeated_observation_idempotent=True,
        original_request_retry_mutation_atomic=True),
    backpressure=dict(workload='256 KiB valid SSE comment per frame, real uvicorn h11, no body reads during 30 x 100 ms samples',
        qualification_only_wrapper=True,padding_bytes=pressure['comment_padding_bytes'],
        sample_wall_s=pressure['samples'][-1]['t']-pressure['samples'][0]['t'],
        max_send_wait_s=max(s['wait_s'] for s in pressure['sends']),
        plateau_responses=pressure['samples'][-1]['generated'],settled_responses=pressure['recovered']['last_turn']['generated'],
        disconnect_to_settled_observation_s=pressure['disconnect_to_idle_s'],
        response_limit=768,total_token_limit=8192,
        resource_observations=resource,
        bounds='pull-based single-flight; one protected in-flight phase; admitted response/event history bounded per request; comments not canonical or retained',
        limits='finite socket restriction only, not unbounded safety; diagnostic trace/closed-session lifetime retention not qualified for indefinite service'),
    replay_repack=dict(probe=p['counts'],m35_rerun=h['counts']),performance=performance,
    text_next_turn={k:next_turn[k] for k in ['prefill_handoff_s','first_canonical_s','decode_s','formatting_s','cleanup_s','elapsed_s']},
    defects=[dict(name='unfinished tool incorrectly advertised idle',evidence='pre-pressure-harness-fix.json',
        fix='authoritative guard finish is required; unfinished tool settlement poisons instead of inventing a complete canonical call'),
        dict(name='poison visible before response retirement settled',evidence=['poison-get-defect.log','early-poison-observation.json'],
        fix='busy precedes poisoned until exact lease release')],
    attempts=dict(pre_pressure_harness_fix='real pre-containment tool/Unicode observations preserved; pressure failure was SSE padding applied to JSON Content-Length, excluded',
        poison_get_defect='first containment run observed poison before cleanup completion; excluded from final stable observation',
        boundary_regressions='initial text-only native fixture lacked guard.finished; condition evaluates tool presence first; superseded by passing runtime-regressions.log'),
    gates=dict(logs=logs,real_m35_http_turns=len(h['cases']),real_native_interruption_cases=len(n['cases']),identities='PASS',
        matched_m35_http_direct_model_phase_ratio=model_ratio,
        historical_m35_head_hash_gate='one identity_and_sources test deselected: unchanged M35 artifact hashes describe its historical source; fresh M36 source/evidence hashes are gated instead'),
    unsupported=['public/default/release MTP','persistence/process restart','immediate token-exact abort',
        'concurrent/shared execution','batching','distributed recovery','200K HTTP MTP','unbounded backpressure','portable native wheels'],
    next_milestone='M36R canonical protocol representability and recovery admission design; not release promotion or longer-context qualification',
    source_hashes={name:sha(ROOT/name) for name in source_paths},
    evidence_hashes={f.name:sha(f) for f in OUT.iterdir() if f.is_file() and f.name not in ['qualification.json','record.log','evidence-tests.log']})
(OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(decision=result['decision'],blocker=result['blocker'],gates=result['gates']),indent=2))
