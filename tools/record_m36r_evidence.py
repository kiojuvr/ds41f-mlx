"""Record only current-source bounded certified-recovery evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/m36r'
def load(name):return json.loads((OUT/name).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
h=load('http.json');m=load('m35-http.json');p=load('m36-probe.json');n=load('native-interruptions.json');r=load('recipe-matrix.json')
assert h['status']==m['status']==n['status']=='PASS' and p['status']=='PROBE_COMPLETE'
for sample in [h,m,p]:
    for path,digest in sample['sources'].items():assert sha(ROOT/path)==digest,path
    assert sample['counts']['replay']==sample['counts']['repack']==0
assert len(m['cases'])==18 and len(n['cases'])==16 and len(r['rows'])==28
positive=[x['name'] for x in r['rows'] if x['certificate']['representable']]
negative=[x['name'] for x in r['rows'] if not x['certificate']['representable']]
assert len(positive)==9 and len(negative)==19
by={x['name']:x for x in h['cases']}
assert by['partial-tool']['recovered']['outcome_state']=='unrecoverable'
assert by['partial-tool']['tool_executions']==0 and by['partial-tool']['continuation_rejected_atomic']
assert by['completed-tool']['tool_executions']==1 and by['completed-tool']['execution_reobservations']==4
assert by['active']['active_retry_no_generation']
for name in ['unicode','completed-tool']:
    assert by[name]['continuation_exact_prefix'] and by[name]['expired_retry_atomic']
traces=[c['recovered']['last_turn'] for c in h['cases']]+[c['after']['last_turn'] for c in h['cases'] if 'after' in c]
traces += [c['runtime'] for c in m['cases']]+[m['direct_control']]
traces += [c['recovered']['last_turn'] for c in p['cases']]+[p['cases'][1]['after']['last_turn']]
for t in traces:
    assert t['prompt_replay']==t['full_cache_repack']==0
    assert t['queue_empty'] and t['prediction_retired']
    assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
    assert set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']}
    assert all(t['quiescence']['counters'][k]==0 for k in ['new_verify_cycles','new_proposals','history_replay','full_cache_repack'])
pressure=p['cases'][-1];assert pressure['name']=='socket-pressure'
assert max(s['wait_s'] for s in pressure['sends'])>1
assert pressure['samples'][-1]['generated']==pressure['samples'][-10]['generated']
assert p['cases'][0]['ordinary_conversion']['succeeded'] and not p['cases'][0]['ordinary_conversion']['exact_prefix']
assert p['cases'][0]['rejection_mutation_atomic'] and p['cases'][1]['ordinary_conversion']['exact_prefix']
logs=['runtime-regressions.log','controlled-admission-regressions.log','horizon-regressions.log',
      'off-policy-regressions.log','release-off-regressions.log','rust-regressions.log','prior-evidence-regressions.log']
for name in logs:
    text=(OUT/name).read_text();assert ('passed' in text or 'test result: ok' in text) and 'FAILED' not in text,name
identity=load('identities.log');assert identity['status']=='PASS' and all(identity['checks'].values())
source_paths=['ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/server.py','ds41f_mlx/serving/recovery_certificate.py',
              'ds41f_mlx/serving/request_fence.py','tools/run_m36r_http.py','tools/run_m36r_recipe_matrix.py',
              'tools/record_m36r_evidence.py','tests/test_m36r_recovery.py','tests/test_m36r_evidence.py',
              'docs/milestone-36r-recovery-admission.md','docs/README.md','docs/doc-classification.json',
              'docs/implementation-plan.md','docs/qualification.md','docs/session-state.md']
for path in ['ds41f_mlx/runtime/recipe_semantic_guard.py','ds41f_mlx/runtime/mtp_lifecycle.py']:
    assert subprocess.check_output(['git','show','df788d5:'+path])==(ROOT/path).read_bytes()
result=dict(schema='ds41f.m36r.qualification.v1',decision='QUALIFIED_BOUNDED_CERTIFIED_RECOVERY',
    base_commit=subprocess.check_output(['git','rev-parse','df788d5'],text=True).strip(),
    final_commit_resolver='git log -1 --format=%H -- artifacts/m36r/qualification.json',
    identities=identity,
    rule=dict(authority='sole native canonical H plus existing official recipe interpretation',
        certificate='ordinary official message witness converts/renders/encodes to a strictly longer request with prefix exactly H; tool outcomes additionally require canonical DSML block terminal and tool_calls finish',
        negative_policy='no certificate => retire native owners/caches, outcome_state unrecoverable, preserve diagnostic H, DELETE/fresh session required',
        completeness='conservative fixed-authoritative-message witness, not exhaustive search over hypothetical equivalent envelopes',
        native_settlement_required=True,syntactic_json_sufficient=False,cache_quiescence_sufficient=False,sse_accumulation_sufficient=False),
    tested_state_classes=dict(fixture_scope='real native official recipe/tokenizer; controlled token input, no sampled model or synthetic-cache execution claim',
        positive=positive,negative=negative,
        exact_but_incomplete=[x['name'] for x in r['rows'] if x['certificate'].get('exact_prefix') and not x['certificate'].get('semantic_complete')],
        complete_but_nonprefix='complete_tool_alternate_spelling',positive_and_negative_certificates='recipe-matrix.json'),
    state_model=dict(authoritative_field='outcome_state',states=['not_admitted','active','recoverable','unrecoverable','poisoned'],
        busy_until_native_and_protocol_settlement=True,legacy_state_poisoned_contains_unrecoverable=True,
        private_direct_controls_without_body='not protocol-certified; contract applies to ordinary HTTP requests through injected backend'),
    fence=dict(header='X-DS41F-Request-Sequence',start=1,identity='session ID / monotonic integer / SHA256 exact body bytes',
        scope='one living in-process local singleton and one well-behaved client; no process restart or concurrent client scope',
        storage='one latest outcome slot plus consumed monotonic sequence; older sequences reject as expired',
        active_retry='409 non-generating',settled_retry='200 JSON outcome wrapper, non-generating and non-mutating',
        not_admitted='pre-conversion/rejection leaves next sequence unchanged; unstarted response header failure keeps same-body sequence retryable',
        started_failure='consume sequence and poison, never retry generation',body_mismatch_and_nonprefix_atomic=True,
        preiteration_header_failure_evidence='synthetic lifecycle',active_and_committed_ambiguity_evidence='real checkpoint HTTP',
        public_api=False,legacy_no_header='no ambiguous identity guarantee; mixing modes rejected'),
    client_workflow=['retry/observe the same exact-body sequence after loss','wait until non-active',
        'discard local transport fragments; recover only certified canonical response','replace assistant entry',
        'execute only completed certified tools through living client ledger','build ordinary next messages and increment sequence',
        'official conversion plus actual exact canonical-prefix admission','on negative/poison DELETE and restart fresh'],
    tools=dict(completed_call_effects=by['completed-tool']['tool_executions'],reobservations=4,incomplete_effects=0,
        key=['session_id','sequence','call_index','call_id'],reserve_before_effect=True,
        ambiguous_effect_automatic_retry=False,active_wrong_identity_and_poisoned_denied=True,
        scope='bounded living single-client/session duplicate prevention, NOT distributed/crash-safe exactly-once'),
    continuation={name:dict(exact_prefix=True,frontiers=[by[name]['recovered']['canonical_frontier'],by[name]['after']['canonical_frontier']],
        retained_cache_continuation=True,expired_retry_atomic=True) for name in ['unicode','completed-tool']},
    negative_checkpoint=dict(frontier=by['partial-tool']['recovered']['canonical_frontier'],
        certificate=by['partial-tool']['reobserved']['certificate'],continuation_rejected_before_mutation=True,tool_executions=0),
    replay_repack=dict(all_instrumented_counts_zero=True,http=h['counts'],m35=m['counts'],m36=p['counts'],
        settlement_new_verify_and_proposal_zero=True),
    backpressure=dict(unchanged_m36_workload=True,listener_sndbuf=p['transport']['listener_sndbuf'],comment_padding_bytes=pressure['comment_padding_bytes'],
        max_send_wait_s=max(s['wait_s'] for s in pressure['sends']),plateau_responses=pressure['samples'][-1]['generated'],
        settled_responses=pressure['recovered']['last_turn']['generated'],disconnect_to_settled_s=pressure['disconnect_to_idle_s'],
        bound='pull-based finite socket-pressure case; 768 responses / 8192 total tokens; no independent generation queue',
        diagnostic_lifetime_retention='unchanged, not qualified for indefinite service'),
    defects=dict(new_native_ownership_mismatch=False,upstream_protocol_limitations_repaired=False,
        new_contract_hardenings=['ledger checks settled state and matching identity','DELETE retirement failure is internal poison, not protocol limitation'],
        historical_hash_gate_selection_attempt='preserved in prior-evidence-historical-hash-attempt.log; excluded',
        native_horizon_and_cache_transfer_unchanged=True),
    gates=dict(logs=logs,m35_http_turns=18,native_interruption_cases=16,controlled_native_recipe_cases=28,
        historical_hash_deselections=['M35 identity_and_sources','M36 current_sources_and_raw_evidence_match'],
        current_source_hash_gate='tests/test_m36r_evidence.py',real_m36_pressure_rerun=True),
    historical_evidence_hashes={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/f'artifacts/m{i}/qualification.json' for i in [33,34,35,36]]+[ROOT/'artifacts/m36/pre-pressure-harness-fix.json']},
    unsupported=['public/default/release MTP','persistence/process restart','token-exact immediate abort','concurrent/shared MTP',
        'batching','distributed recovery/idempotency','200K HTTP MTP','unbounded backpressure','portable native wheels',
        'universal partial canonical DSML/tool recovery','hypothetical alternate envelope search'],
    next_milestone='M37 bounded internal local-client recovery/execution-ledger integration, including DELETE/restart and expired outcome handling',
    source_hashes={path:sha(ROOT/path) for path in source_paths},
    evidence_hashes={str(path.relative_to(ROOT)):sha(path) for path in OUT.iterdir() if path.is_file() and path.name not in ['qualification.json','record.log','evidence-tests.log']})
(OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['decision','tested_state_classes','continuation','backpressure','gates']},indent=2))
