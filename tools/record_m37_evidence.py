"""Record bounded real-client evidence only after current-source gates pass."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/m37'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads((OUT/p).read_text())
w=load('workflow.json')
assert w['status']=='PASS'
for path,digest in w['sources'].items():assert sha(ROOT/path)==digest,path
assert len(w['turns'])==8 and len(w['effects'])==len(w['ledger'])==2
by={t['name']:t for t in w['turns']}
assert by['incomplete-tool']['outcome']['outcome_state']=='unrecoverable'
assert by['incomplete-tool']['transcript']==json.loads(by['incomplete-tool']['identity']['body_utf8'])['messages']
assert by['fresh-session-ordinary-prefill']['state']=='ready'
assert json.loads(by['fresh-session-ordinary-prefill']['identity']['body_utf8'])['stream'] is False
assert not by['ambiguous-complete-tool']['observed']
assert by['ambiguous-stream-text']['observed']
for t in w['turns']:
    i=t['identity'];raw=i['body_utf8'].encode()
    assert hashlib.sha256(raw).hexdigest()==i['body_sha256']
    assert t['outcome']['sequence']==i['sequence']
    if t['outcome']['outcome_state']=='recoverable':
        cert=t['outcome']['certificate'];msg=t['outcome']['response']['choices'][0]['message']
        assert cert['representable'] and cert['exact_prefix'] and cert['semantic_complete']
        assert t['transcript']==json.loads(raw)['messages']+[msg]
assert [t['identity']['sequence'] for t in w['turns']]==[1,2,3,4,5,6,7,1]
assert w['counts']['replay']==w['counts']['repack']==0 and w['counts']['fresh_target_allocations']==2
for t in w['traces']:
    assert t['prompt_replay']==t['full_cache_repack']==0
    assert t['queue_empty'] and t['prediction_retired']
    assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
    assert set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']}
    assert all(t['quiescence']['counters'][k]==0 for k in ['history_replay','full_cache_repack','new_verify_cycles','new_proposals'])
commands=load('native-gate-commands.json');assert len(commands)==6 and all(c['returncode']==0 for c in commands)
for name,status in [('m35-http.json','PASS'),('m36-probe.json','PROBE_COMPLETE'),('m36r-http.json','PASS'),('native-interruptions.json','PASS')]:
    sample=load(name)
    assert sample['status']==status,(name,sample['status'])
    if 'counts' in sample:assert sample['counts']['replay']==sample['counts']['repack']==0
    for path,digest in sample.get('sources',{}).items():assert sha(ROOT/path)==digest,path
matrix=load('recipe-matrix.json');assert len(matrix['rows'])==28
logs=['client-tests.log','runtime-regressions.log','rust-regressions.log','prior-evidence-regressions.log','release-off-regressions.log']
for name in logs:
    text=(OUT/name).read_text();assert ('passed' in text or 'test result: ok' in text) and 'FAILED' not in text
# Ownership layers are unchanged byte-for-byte relative to M36R.
unchanged=['ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/request_fence.py','ds41f_mlx/serving/recovery_certificate.py',
           'ds41f_mlx/serving/server.py','ds41f_mlx/runtime/mtp_lifecycle.py','ds41f_mlx/runtime/recipe_semantic_guard.py']
for p in unchanged:assert subprocess.check_output(['git','show','40e2303:'+p])==(ROOT/p).read_bytes()
latencies=[]
for turn,trace in zip(w['turns'],w['traces']):
    latencies.append(dict(name=turn['name'],request_wall_s=turn['wall_s'],recovery_wait_lookup_observe_s=turn['recovery_wait_and_lookup_s'],
        load_s=trace['load_s'],decode_s=trace['decode_s'],prefill_handoff_s=trace['prefill_handoff_s'],delivery_wait_s=trace['delivery_wait_s'],
        cleanup_s=trace['cleanup_s'],fresh_session_prefill=turn['identity']['sequence']==1,
        new_fresh_after_delete=turn['name']=='fresh-session-ordinary-prefill'))
ledger_times=[x['seconds'] for x in w['timings'] if x['action']=='tool_ledger_and_effect']
# Calls 0 and 4 include the stub; other calls are pure duplicate ledger lookup.
assert len(ledger_times)==8
source_paths=['ds41f_mlx/internal_local_client.py','ds41f_mlx/web_client.py','tools/run_m37_local_client.py',
              'tools/run_m37_gates.py','tools/record_m37_evidence.py','tests/test_m37_local_client.py','tests/test_m37_evidence.py',
              'docs/milestone-37-local-client-integration.md','docs/README.md','docs/implementation-plan.md',
              'docs/qualification.md','docs/session-state.md','docs/doc-classification.json']
result=dict(schema='ds41f.m37.qualification.v1',decision='QUALIFIED_BOUNDED_INTERNAL_LOCAL_CLIENT',base_commit='40e2303',
    final_commit_resolver='git log -1 --format=%H -- artifacts/m37/qualification.json',
    integrated_boundary='ds41f_mlx.internal_local_client.InternalLocalClient above existing local/browser RuntimeClient HTTP boundary; explicitly experimental; public browser and raw Rust API unchanged',
    authority=dict(server='canonical history, certificate, request/outcome identity, classification and session/cache ownership',
        client='ordinary local transcript, frozen bytes, expected sequence, transport observations and bounded living-client effect ledger',
        sse='display-only; incomplete fragments discarded',no_token_parser_cache_authority=True),
    fence=dict(sequences=[t['identity']['sequence'] for t in w['turns']],same_body_retry=True,
        settlement='same exact-byte POST JSON outcome; no generation',active='no sequence advancement; explicit wait',
        absent='same identity only, never absent-GET ACK',expired='explicit expired result; reconcile current conversation or explicit restart, no regeneration',
        wrong_body='state error/stop, never substitute work',current_displaced_expiry_evidence='synthetic client fault test; shared clients unsupported'),
    recovery=dict(positive_assistant_replacement=True,complete_tool_zero_events_lost=True,
        ordinary_next_turn_exact_prefix=True,negative_assistant_not_manufactured=True,internal_poison_distinct=True),
    tools=dict(effects=2,observations_per_call=4,ledger_entries=2,key=['session_id','sequence','call_index','call_id'],
        full_call_binding=True,reserve_before_effect=True,ordinary_stored_result_continuation=True,
        incomplete_effects=0,ambiguous_client_effect_retry=False,scope='one living single-thread client; not crash-safe/distributed exactly-once'),
    restart=dict(explicit_delete=True,old_id_and_request_denied=True,fresh_creation=True,
        preserved='prior certified assistants, actual completed tools/results and user messages including final submitted user; no unrepresentable final native assistant',
        native_state_carried=False,new_session_ordinary_prefill=True),
    replay_repack=dict(model_history_replay=0,full_cache_repack=0,fresh_target_allocations=2,
        retained_continuation_turns=6,new_session_prefill_turns=2,post_delete_new_session_prefill_turns=1),
    observations=dict(latencies=latencies,ledger_with_effect_s=[ledger_times[0],ledger_times[4]],
        ledger_only_s=[v for i,v in enumerate(ledger_times) if i not in (0,4)],
        lifecycle=[x for x in w['timings'] if x['action'] in ('create','delete')],resources=w['resources'],
        final_client_serialized_payload_bytes=w['client_payload_bytes'],
        max_measured_client_serialized_payload_bytes=max(t['retained_state_bytes'] for t in w['turns']),
        retention='one current frozen body/outcome, transcript, 128 ledger effects (64KiB result each), 128 timing samples; body <=1MiB; estimates exclude Python allocator overhead'),
    defects=dict(server_runtime_defects=False,owning_layers_unchanged=unchanged,
        excluded_attempts=['initial fresh tool-enabled model validly requested a new tool, violating harness text-only expectation',
                           'superseded in-development JSON serialization of diagnostic deque; fixed harness conversion to list'],
        failure_evidence_preserved=True),
    gates=dict(native=commands,logs=logs,m35_http_turns=18,m36r_http_sessions=4,native_interruption_cases=16,native_recipe_cases=28,
        historical_deselections=['M35 identity_and_sources','M36 current_sources_and_raw_evidence_match','M36R final_qualification'],
        current_hash_gate='tests/test_m37_evidence.py'),
    unsupported=['public/default/release MTP','persistence/process restart','client crash-safe tool exactly-once','distributed idempotency',
        'concurrent/shared MTP','batching','token-exact immediate abort','200K HTTP MTP','unbounded backpressure','portable native wheels','universal partial DSML recovery'],
    next_milestone='M38 bounded internal recovery-client operational soak and fault matrix; separate later release/admission evaluation',
    source_hashes={p:sha(ROOT/p) for p in source_paths},
    historical_evidence_hashes={f'artifacts/m{i}/qualification.json':sha(ROOT/f'artifacts/m{i}/qualification.json') for i in [33,34,35,36,'36r']},
    evidence_hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('qualification.json','record.log','evidence-tests.log')})
(OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['decision','replay_repack','observations','gates']},indent=2))
