"""Derive M39 decision from current raw evidence; no promotion."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/m39'
def read(name):return json.loads((OUT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
lifetime=read('lifetimes.json');integration=read('integration.json');gates=read('gate-commands.json')
native=read('native-interruptions.json');transport=read('lifecycle-transport.json')
prior=json.loads((ROOT/'artifacts/m38/qualification.json').read_text())
preserved=['ds41f_mlx/serving/request_fence.py','ds41f_mlx/serving/recovery_certificate.py',
           'ds41f_mlx/runtime/mtp_lifecycle.py','ds41f_mlx/runtime/recipe_semantic_guard.py',
           'ds41f_mlx/internal_local_client.py','ds41f_mlx/web_client.py']
checks=dict(
    prior_ownership_layers_unchanged=all(sha(ROOT/p)==prior['source_hashes'][p] for p in preserved),
    lifetime_stress=lifetime['status']=='PASS' and lifetime['lifetimes']>=20000,
    diagnostic_plateau=all(s['diagnostic_records']==16 and s['diagnostic_payload_bytes']<=3216 for s in lifetime['samples'] if s['cycles']>=16),
    authoritative_plateau=all(s['live_records']==s['retired_authority_records']==s['closed_outcome_payload_bytes']==0 and s['identity_storage_bytes']<=117 for s in lifetime['samples']),
    eviction_and_exhaustion=lifetime['diagnostic_eviction_preserves_retirement'] and lifetime['exhaustion_fail_closed'],
    checkpoint_integration=integration['status']=='PASS' and integration['evicted_session_and_request_rejected'] and integration['counts']['fresh_target_allocations']==4,
    no_replay=integration['counts']['replay']==integration['counts']['repack']==0,
    admission_before_mutation=len(integration['admission'])==3,
    lifecycle_ambiguity=transport['status']=='PASS' and all(r['no_retry_or_replacement'] for r in transport['rows']),
    affected_gates=len(gates)==9 and all(r['returncode']==0 for r in gates),
    checkpoint_native=native['status']=='PASS' and len(native['cases'])==16,
    official_recipe_matrix=len(read('recipe-matrix.json')['rows'])==28,
    integration_sources=all(sha(ROOT/p)==h for p,h in integration['sources'].items()),
)
sources=['ds41f_mlx/serving/internal_mtp.py','ds41f_mlx/serving/server.py',
         'ds41f_mlx/serving/request_fence.py','ds41f_mlx/internal_local_client.py',
         'tests/test_m35_transport_lease.py','tests/test_m39_lifetime_admission.py',
         'tests/test_m39_evidence.py','docs/milestone-39-lifetime-admission.md',
         'docs/README.md','docs/doc-classification.json','docs/session-state.md',
         'docs/qualification.md','docs/implementation-plan.md',
         *[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tools').glob('*m39*.py'))]]
raws=['lifetimes.json','lifetimes.log','integration.json','integration.log','lifecycle-transport.json',
      'lifecycle-transport.log','gate-commands.json','native-interruptions.json','recipe-matrix.json',
      'admission-tests.xml','client-tests.xml',*[f'{r["name"]}.log' for r in gates]]
result=dict(schema='ds41f.m39.qualification.v1',
    decision='QUALIFIED_FINITE_PROCESS_LIFETIME_ADMISSION' if all(checks.values()) else 'BLOCKED',
    base_commit='998500b',checks=checks,
    primary_commands=[dict(command=[sys.executable,f'tools/{name}.py'],
                           environment=dict(DS41F_OMLX_PATH='/tmp/ds41f-m33-omlx',
                                            DS41F_RECIPE_PATH='/tmp/ds41f-m32-recipe'))
                      for name in ('run_m39_lifetimes','run_m39_integration',
                                   'run_m39_lifecycle_transport','run_m39_gates')],
    identity_model='single backend/process namespace plus contiguous fixed 128-bit no-wrap server issuance',
    invariant='only live dictionary membership executes; issued serials never reissued; diagnostics never authorize',
    budgets=dict(live_sessions=1,active_requests=1,identity_namespace_bits=128,identity_counter_bits=128,
                 lifetimes_max=str(2**128-1),request_sequence_bits=64,body_bytes=1048576,total_tokens=8192,
                 responses=768,retired_authority_records=0,retired_outcome_payload_slots=0,
                 diagnostic_records=16,diagnostic_summary_max_bytes=199,diagnostic_list_max_bytes=3216,
                 trace_slots=32,session_trace_slots=32),
    lifetimes=lifetime['lifetimes'],performance=lifetime['performance'],
    structural_samples=lifetime['samples'],checkpoint_resources=integration['resources'],
    preserved_ownership_layers={p:sha(ROOT/p) for p in preserved},
    historical_evidence={f'artifacts/{m}/qualification.json':sha(ROOT/f'artifacts/{m}/qualification.json')
                         for m in ('m33','m34','m35','m36','m36r','m37','m38')},
    sources={p:sha(ROOT/p) for p in sources},evidence={p:sha(OUT/p) for p in sorted(set(raws))},
    excluded_attempts={
        'initial-regressions.log':'tests called historical client-selected IDs; synthetic lease fixture labels adapted, real admission covered separately',
        'admission-tests-first-attempt.log':'ceiling fixture attempted no-header fenced retry, which correctly rejects mode mixing; fixture corrected',
        'lifetimes-first-attempt.log':'weakref checked before executor result callback release; next event-loop turn required, not backend retention',
        'integration-envelope-attempt.json':'tool schema changed on retained text-only session; exact-prefix correctly rejected; final lifetime integration keeps envelope unchanged',
        'integration-preliminary.json':'passing preliminary integration superseded by final source-hashed HTTP admission run'},
    unsupported=['public/default/release MTP','persistence/process restart/backend recreation','distributed identity/recovery',
                 'crash-safe tool exactly-once','concurrent/shared MTP','batching','token-exact immediate abort',
                 '200K HTTP MTP','unbounded backpressure','portable native wheels','universal partial DSML recovery'],
    next_task='dedicated release/admission/provenance readiness evaluation; not a recovery micro-milestone')
(OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['decision']);print(json.dumps(checks,indent=2))
if not all(checks.values()):raise SystemExit(1)
