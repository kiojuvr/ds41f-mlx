"""Record M38 only from completed current-source evidence; do not preselect PASS."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/m38'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(n):return json.loads((OUT/n).read_text())
w=load('soak.json');assert w['status']=='PASS_BOUNDED_WORKLOAD'
assert len(w['turns'])==76 and len(w['effects'])==len(w['ledger'])==14
assert [sum(t['phase']==p for t in w['turns']) for p in range(3)]==[25,26,25]
for t in w['turns']:
    raw=t['identity']['body_utf8'].encode();seq=t['identity']['sequence'];out=t['outcome']
    assert hashlib.sha256(raw).hexdigest()==t['identity']['body_sha256']
    assert seq==out['sequence']
    expected=json.loads(raw)['messages']
    if out['outcome_state']=='recoverable':expected=expected+[out['response']['choices'][0]['message']]
    else:assert out['outcome_state']=='unrecoverable'
    assert t['transcript']==expected
    trace=t['trace'];assert trace['prompt_replay']==trace['full_cache_repack']==0
    assert len(trace['target_offsets'])==40 and len(trace['dspark_offsets'])==3
    assert set(trace['target_offsets']+trace['dspark_offsets'])=={trace['canonical_frontier']}
    assert trace['queue_empty'] and trace['prediction_retired']
    assert all(trace['quiescence']['counters'][k]==0 for k in ['history_replay','full_cache_repack','new_verify_cycles','new_proposals'])
    assert t['resources']['timings']<=128 and t['resources']['server_trace_length']<=32
assert w['counts']['replay']==w['counts']['repack']==0 and w['counts']['fresh_target_allocations']==3
for path,digest in w['sources'].items():assert sha(ROOT/path)==digest,path
pressure=load('client-pressure.json');assert pressure['status']=='PASS'
assert pressure['counts']['replay']==pressure['counts']['repack']==0
assert max(s.get('wait_s',0) for s in pressure['sends'])>1
for path,digest in pressure['sources'].items():assert sha(ROOT/path)==digest,path
life=load('lifecycle-transport.json');assert life['status']=='PASS'
assert all(r['error']['type']=='IncompleteRead' and not r['retry_or_replacement'] for r in life['rows'])
ledger=load('ledger-exhaustion.json');assert ledger['status']=='PASS'
assert ledger['effects']==ledger['capacity']==len(ledger['ledger'])==128 and ledger['overflow_effects']==0
assert ledger['state']=='stopped' and ledger['results_unchanged'] and ledger['no_eviction']
assert len({e['key'][0] for e in ledger['ledger']})==4
retention=load('retention-probe.json')
assert retention['finding']=='NO_LIFETIME_SESSION_CAP' and retention['samples'][-1]['retained_sessions']==256
assert [p['resources']['closed_sessions'] for p in w['phases']]==[1,2,3]
# Block on the demonstrated structural lifetime gap, not workload or allocator variation.
decision='BLOCKED_OPERATIONAL_LIFETIME_RETENTION'
xml=ET.parse(OUT/'client-tests.xml');cases=[]
for case in xml.findall('.//testcase'):
    assert case.find('failure') is None and case.find('error') is None
    cases.append(dict(name=case.get('name'),classname=case.get('classname'),seconds=float(case.get('time'))))
commands=load('gate-commands.json');assert len(commands)==7 and all(c['returncode']==0 for c in commands)
native=load('native-gate-commands.json');assert len(native)==6 and all(c['returncode']==0 for c in native)
assert load('m37-workflow.json')['status']=='PASS'
for n,status in [('m35-http.json','PASS'),('m36-probe.json','PROBE_COMPLETE'),('m36r-http.json','PASS'),('native-interruptions.json','PASS')]:assert load(n)['status']==status
unchanged=['ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/server.py','ds41f_mlx/serving/request_fence.py','ds41f_mlx/serving/recovery_certificate.py','ds41f_mlx/runtime/mtp_lifecycle.py','ds41f_mlx/runtime/recipe_semantic_guard.py']
for path in unchanged:assert subprocess.check_output(['git','show','daf429e:'+path])==(ROOT/path).read_bytes(),path
# Inspect the complete historical artifact trees, including failed/excluded attempts.
historical={};audits=[]
for m in ['33','34','35','36','36r','37']:
    directory=ROOT/f'artifacts/m{m}';qual=json.loads((directory/'qualification.json').read_text())
    checked=0
    for group in ['evidence_hashes','evidence_sha256','historical_evidence_hashes']:
        for path,digest in qual.get(group,{}).items():
            if not path.startswith('artifacts/') and '/' in path:continue # historical SOURCE hashes, not current HEAD
            p=ROOT/path if path.startswith('artifacts/') else directory/path
            assert sha(p)==digest,str(p);checked+=1
    inventory={}
    for p in directory.rglob('*'):
        if not p.is_file():continue
        item=dict(sha256=sha(p),bytes=p.stat().st_size)
        if p.suffix=='.json':
            data=json.loads(p.read_text())
            if isinstance(data,dict):item.update({k:data[k] for k in ['schema','status','decision','error'] if k in data})
        inventory[str(p.relative_to(ROOT))]=item
        historical[str(p.relative_to(ROOT))]=item['sha256']
    audits.append(dict(milestone=m,decision=qual['decision'],checked_referenced_artifact_hashes=checked,files=inventory))
(OUT/'historical-evidence-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
performance=[]
for p in range(3):
    allturns=[t for t in w['turns'] if t['phase']==p]
    ordinary=[t for t in allturns if t['mode'] in ['normal','json'] and t['identity']['sequence']!=1]
    performance.append(dict(phase=p,turns=len(allturns),cohort='ordinary normal/json, excluding first sequence and injected loss/delay',
        median_wall_s=statistics.median(t['wall_s'] for t in ordinary),
        median_prefill_handoff_s=statistics.median(t['trace']['prefill_handoff_s'] for t in ordinary),
        weighted_decode_responses_s=sum(t['trace']['generated'] for t in ordinary)/sum(t['trace']['decode_s'] for t in ordinary),
        median_reconcile_wall_s=statistics.median(t['reconcile_wait_s'] for t in ordinary),
        fresh_prefill_handoff_s=allturns[0]['trace']['prefill_handoff_s'],load_s=allturns[0]['trace']['load_s']))
source_paths=unchanged+['ds41f_mlx/internal_local_client.py','ds41f_mlx/web_client.py',
    *[str(p.relative_to(ROOT)) for p in (ROOT/'tools').glob('*m38*.py')],
    'tests/test_m38_client_faults.py','tests/test_m38_evidence.py','docs/milestone-38-client-operational-soak.md',
    'docs/README.md','docs/implementation-plan.md','docs/qualification.md','docs/session-state.md','docs/doc-classification.json']
result=dict(schema='ds41f.m38.qualification.v1',base_commit=subprocess.check_output(['git','rev-parse','daf429e'],text=True).strip(),
    decision=decision,bounded_workload='PASS_BOUNDED_WORKLOAD',release_promoted=False,
    final_commit_resolver='git log -1 --format=%H -- artifacts/m38/qualification.json',
    scale=dict(main_turns=76,agent_sessions=3,all_main_process_sessions=5,certified_effects=14,returned_responses=2390,
        pressure_client_turns=2,synthetic_ledger_effects=128,synthetic_ledger_sessions=4,synthetic_retention_records=256),
    longitudinal=dict(one_current_identity=True,qualified_sequence_advance=True,frozen_retries_not_generation=True,
        immutable_outcomes=True,sse_display_only=True,positive_replacement_once=True,negative_execution=False,
        expired_regeneration=False,retired_identity_reuse=False,fresh_native_authority_inheritance=False,
        end_of_request_and_phase_assertions=True),
    recovery_matrix=dict(main_modes=dict(collections.Counter(t['mode'] for t in w['turns'])),
        synthetic_cases=cases,actual_lifecycle_transport=life['rows'],pressure_scope=pressure['scope'],
        native_poison='fresh 16-case native gate; controlled protected-phase exceptions, not naturally sampled poison'),
    tools=dict(main_effects=14,main_ledger_entries=14,capacity=128,overflow_effects=0,no_eviction=True,
        ambiguous_effect='reserved; repeated observation preserves tool_ambiguous; execution/new-work/results/retire/create blocked',
        resolution='explicit external application intervention; no supported reset/resolution API',
        ledger_exhaustion_scope=ledger['scope'],crash_safe=False,distributed_exactly_once=False),
    lifecycle=dict(successful_main_agent_retirements=3,retained_fresh_reconstruction_cycles=2,sequence_restart=1,
        uncertain_delete_create='sticky stopped/manual reconciliation; no silent retry or replacement',
        real_truncated_response_errors=[r['error']['type'] for r in life['rows']],
        external_controller_cleanup_not_client_authority=True),
    replay_repack=dict(main_model_replay=0,main_full_repack=0,main_fresh_prefills=3,main_retained_suffix_requests=73,
        pressure_fresh_prefills=1,pressure_retained_suffix_requests=1,settlement_new_verify_proposals=0),
    resources=dict(phases=[p['resources'] for p in w['phases']],final=w['final_resources'],
        max_measured_client_payload_bytes=max(t['resources']['client_bytes'] for t in w['turns']),
        ledger_capacity_samples=ledger['samples'],
        intended_bounds=dict(request_bytes=1048576,ledger_entries=128,result_bytes=65536,timing_samples=128,
            server_trace_samples=32,max_response_tokens=768,max_prompt_plus_budget_tokens=8192,max_live_sessions=1),
        open_blocker='uncapped closed-session map retains canonical IDs, last_turn/certificate/reconstruction/guard payloads; trace eviction does not release them',
        estimate_caveat='serialized payload estimate, excludes Python allocator and timing deque; model-resident MLX bytes != process RSS; not a universal leak proof'),
    performance=dict(phases=performance,retained_timing_samples=w['timings'],
        pressure_max_send_wait_s=max(s.get('wait_s',0) for s in pressure['sends']),
        pressure_disconnect_to_settlement_s=pressure['disconnect_to_settlement_s']),
    defects_fixed=['sticky lifecycle uncertainty prohibits subsequent lifecycle mutation','reserved effect cannot lose classification or bypass through retirement/fresh creation','non-dictionary additions reject explicitly before mutation'],
    blocker_fixed=False,owning_server_runtime_layers_unchanged=unchanged,
    gates=dict(commands=commands,native=native,client_test_cases=len(cases),historical_hash_deselections=[
        'identity_and_sources','current_sources_and_raw_evidence_match','final_qualification','current_source_and_evidence_hashes']),
    excluded_attempts=['import-attempt.log: direct script lacked repository import path; final invocation is python -m',
        'soak-fresh-tool-attempt.*: valid unsolicited certified fresh-session tool, harness tried new work before stored-result continuation',
        'excluded-gate-attempt/: interrupted overlapping in-development gate scheduling; no result used',
        'gate-commands-first-attempt.json/prior-evidence-first-attempt.log: missed historical M37 HEAD-hash deselection; corrected subset rerun',
        'pre-fix-client-faults.log: deliberate original M37 regressions, 10 failed/8 passed'],
    unsupported=json.loads((ROOT/'artifacts/m37/qualification.json').read_text())['unsupported'],
    readiness_gaps=['server lifetime/admission retirement budget','released manual lifecycle/effect workflow and envelope/admission policy','portable installation/native provenance','separate longer-context HTTP qualification'],
    recommended_next='operational lifetime/admission hardening: finite budget without retired-ID eviction, closed payload reduction and mutation-atomic capacity denial; then dedicated release/admission/provenance evaluation, not client-semantic micro-milestones',
    source_hashes={p:sha(ROOT/p) for p in source_paths},historical_evidence_hashes=historical,
    evidence_hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['qualification.json','record.log','evidence-tests.log']})
(OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['decision','scale','replay_repack','performance']},indent=2))
