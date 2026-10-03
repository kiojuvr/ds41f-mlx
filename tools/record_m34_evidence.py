"""Aggregate fresh M34 evidence; never turn missing/failed gates into PASS."""
import hashlib
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/m34'
def read(name):return json.loads((OUT/name).read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
required=['soak.json','recovery.json','interruptions.json','control-4096.json','control-12288.json','identities.json']
statuses={name:read(name).get('status') if (OUT/name).exists() else 'MISSING' for name in required}
if any(status!='PASS' for status in statuses.values()):
    blocked=dict(schema='ds41f.m34.qualification.v1',decision='BLOCKED',gate_statuses=statuses,reason='Required fresh operational evidence incomplete or failed; no promotion authorized.')
    (OUT/'qualification.json').write_text(json.dumps(blocked,indent=2)+'\n')
    print(json.dumps(blocked,indent=2));raise SystemExit(1)
soak=read('soak.json');recovery=read('recovery.json');interrupt=read('interruptions.json')
controls=[read(f'control-{c}.json') for c in (4096,12288)]
turns=soak['turns'];allturns=turns+recovery['turns']
checks={
 'exact_m33_identities':read('identities.json')['status']=='PASS',
 'soak_complete':soak['status']=='PASS' and len(turns)==90,
 'recovery_complete':recovery['status']=='PASS' and len(recovery['turns'])==12,
 'live_persistence_denial':len(recovery.get('unsupported_live',[]))==2 and all(row['fail_closed_before_io'] for row in recovery['unsupported_live']),
 'checkpoint_interruption_matrix':interrupt['status']=='PASS' and len(interrupt['cases'])==16,
 'corrected_interruption_runtime':interrupt.get('m34_runtime_sha256')==soak['runtime_correction']['m34_sha256'],
 'corrected_recovery_runtime':recovery['runtime_correction']['m34_sha256']==soak['runtime_correction']['m34_sha256'],
 'frontiers':all(set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']} for t in allturns),
 'prefix_and_ownership':all(t['prefix_preserving'] and t['cache_ownership_preserved'] and t['queue_empty'] and t['retired_prediction'] for t in allturns),
 'zero_replay_repack':all(p['counts']['replay']==p['counts']['repack']==0 for p in (soak,recovery)) and all(t['prompt_replay']==0 and not any(t['quiescence_counts'].values()) for t in allturns),
 'paired_controls':all(p['status']=='PASS' for p in controls),
 'cancel_reentry':recovery['turns'][8]['cancelled'] and recovery['turns'][9]['prefix_preserving'],
 'harness_identity':soak['harness_sha256']==digest(ROOT/'tools/run_m34_operational_soak.py'),
}
# Verify executable regression logs as well as checkpoint evidence.
for name,expected in [('regressions.log','81 passed'),('horizon-regressions.log','29 passed'),('rust-regressions.log','test result: ok.'),('release-off-regressions.log','5 passed'),('off-policy-regressions.log','5 passed')]:
 checks[name]=expected in (OUT/name).read_text()
performance=[]
for p in controls:
 rows={r['mode']:r for r in p['runs']};off=rows['OFF'];on=rows['ON_guarded'];plain=rows['ON_no_guard']
 checks[f"tokens-{on['context']}"]=on['tokens']==plain['tokens'] and on['stats']['accepts']==plain['stats']['accepts']
 performance.append(dict(scope='Fresh pre-correction M34 control; corrected session method not exercised, primitive/guard/model identities unchanged',context=on['context'],output_tokens=len(on['tokens']),off_tok_s=off['tok_s'],guarded_tok_s=on['tok_s'],no_guard_tok_s=plain['tok_s'],speedup=on['tok_s']/off['tok_s'],guard_vs_plain=on['tok_s']/plain['tok_s'],first_response_seconds=on['first_token_seconds'],off_first_response_seconds=off['first_token_seconds'],acceptance=on['acceptance'],accepted_per_cycle=on['accepted_per_cycle'],preview=on['preview']))
long=[t for t in turns if t['kind']=='long_text' and not t['cancelled']]
def acceptance(ts):
 stats=[t['stats'] for t in ts if t['stats']]
 return sum(s['accepts'] for s in stats)/sum(sum(s['depth_drafted']) for s in stats)
resources=soak['resources'];cleanup=[r for r in resources if 'cleanup' in r['label']]
checks['bounded_cleanup_memory']=len(cleanup)==3 and max(r['active_bytes'] for r in cleanup)-min(r['active_bytes'] for r in cleanup)<256*1024*1024
checks['material_control_speedup']=all(r['speedup']>1.5 and r['guard_vs_plain']>.9 for r in performance)
summary=dict(schema='ds41f.m34.qualification.v1',decision='OPERATIONALLY_QUALIFIED_BOUNDED_SINGLETON' if all(checks.values()) else 'BLOCKED',checks=checks,
 identities=soak['runtime_identities'],current_runtime_sources=soak['current_runtime_sources'],runtime_correction=soak['runtime_correction'],base_commit=soak['base_commit'],runtime_source_changed=True,scope=dict(fresh_sessions=3,turns=len(turns),tool_calls=sum(len(t['protocol']['choices'][0]['message'].get('tool_calls') or []) for t in turns),generated_tokens=sum(t['generated'] for t in turns),long_streams=len(long),max_generated=max(t['generated'] for t in turns),max_frontier=max(t['canonical_frontier'] for t in turns),final_frontiers=[t['canonical_frontier'] for t in turns if t['turn']==29],delivery_ack_batches=sum(len(t['delivery_acks']) for t in turns),midstream_cancels=sum(t['cancelled'] for t in turns),recovery_turns=len(recovery['turns']),recovery_final_frontier=recovery['turns'][-1]['canonical_frontier'],fresh_checkpoint_interruptions=len(interrupt['cases']),seconds=soak['elapsed_seconds']),
 operational_performance=dict(weighted_tok_s=sum(t['generated'] for t in turns)/sum(t['decode_seconds'] for t in turns),long_text_tok_s_range=[min(t['tok_s'] for t in long),max(t['tok_s'] for t in long)],acceptance=acceptance(turns),first_response_range_seconds=[min(t['first_response_seconds'] for t in turns),max(t['first_response_seconds'] for t in turns)],quiescence_range_seconds=[min(t['quiescence_seconds'] for t in turns),max(t['quiescence_seconds'] for t in turns)],long_streams=[dict(session=t['fresh_session'],turn=t['turn'],frontier=t['canonical_frontier'],tokens=t['generated'],tok_s=t['tok_s'],acceptance=acceptance([t])) for t in long]),
 interruptions=dict(protected_phase_fail_closed=sum(c['status']=='PASS_FAIL_CLOSED' for c in interrupt['cases']),fresh_native_idle_transitions=sum(c['status']=='PASS' for c in interrupt['cases']),midstream_cancel_seconds=recovery['turns'][8]['quiescence_seconds'],post_cancel_prefix_reentry_frontier=recovery['turns'][9]['canonical_frontier'],post_cancel_handoff_seconds=recovery['turns'][9]['prefill_handoff_seconds'],post_cancel_first_response_seconds=recovery['turns'][9]['first_response_seconds'],transport_ack_metadata_only=True),
 controls=performance,resources=dict(active_range_bytes=[min(r['active_bytes'] for r in resources),max(r['active_bytes'] for r in resources)],rss_range_kib=[min(r['rss_kib'] for r in resources),max(r['rss_kib'] for r in resources)],cleanup=cleanup,interpretation='Bounded measured trend, not a leak proof; active MLX bytes and OS RSS are distinct.'),replay=0,repack=0,
 defects=dict(runtime=['Active cancellation lacked the native completed-response ownership transfer: owner removal cleared its returned list; shallow retention kept old P6 sealed capability. Fixed with existing native singleton row-view extraction, without clearing capability flags/repacking/replay. Added regression and reran full recovery/soak/interruption gates.'],harness=['Initial over-strong Python target-wrapper identity assertion: native finish extraction returns row views. Corrected to prove actual native cache authority and forbid reconstruction; reran.']),
 other_audits=dict(repository_authority_labels='FAILED_PRE_EXISTING_UNCHANGED_FINDINGS',delegated_legacy_source_hashes='FAILED_PRE_EXISTING_UNCHANGED_ARTIFACTS',evidence='artifacts/m34/baseline-authority-findings.json',impact='No findings suppressed; outside directly checked M33/M34 runtime scope.'),
 unsupported=['production/default MTP','public MTP selector','MTP persistence/restore','token-exact immediate abort','shared/concurrent MTP','HTTP MTP integration','portable recipe wheels','200K MTP operational soak'],
 recommended_next='M35: explicit internal single-flight serving integration and transport/cancellation qualification; separate release identity/promotion gate before public or default MTP.')
files=[p for p in OUT.iterdir() if p.is_file() and p.name not in ('qualification.json','file-manifest.json','record.log','evidence-tests.log')]+list((ROOT/'tools').glob('*m34*.py'))+list((ROOT/'tests').glob('test_m34*.py'))
summary['evidence_sha256']={str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}
(OUT/'qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(decision=summary['decision'],checks=checks,scope=summary['scope']),indent=2))
