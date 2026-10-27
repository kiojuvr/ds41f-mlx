"""Read-only reconciliation of the three model-free gates and frozen authorities."""
import hashlib,json,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];OLD=OUT.parent/'phase4-step4a-evidence'
def load(n):return json.loads((OUT/n).read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='2cb84bf06b2eb72e41effe65e484e66b655b912e'
# The CURRENT plan is deliberately updated; every retained Step4A input/report
# including scripts, fixtures and old receipts must otherwise remain unchanged.
frozen=json.loads((OLD/'inputs.sha256.json').read_text());checked=0
for name,expected in frozen.items():
 if name=='../../../docs/multimodal-runtime-plan.md':continue
 assert digest(OLD/name)==expected,name;checked+=1
b=json.loads((OLD/'budgets.json').read_text())
assert b['ordinary_generations']==4+4+4+1+5+1==19
assert b['total_output_allocation']==19*1200+2048==24848
assert b['runtime_http_requests']==20 and b['adapter_attempts_including_offload']==20+7
assert b['sequential_headless_invocations']==2+4+1+1+1==9
assert b['session_seconds']==20*180+420+180==4200
assert b['overall_seconds']==1200+4200==5400
assert load('identity.json')['actual_pre_woa_materialized']
assert load('gate-a-tests.json')['pass'] and len(load('gate-a-tests.json')['checks'])==29
assert load('gate-a-effects.json')['pass'] and len(load('gate-a-effects.json')['effects'])==10
assert load('wire-test.json')['pass']
assert len(load('supervisor-tests.json')['cases'])==17
assert load('control-output-tokens.json')['pass_']
p=load('gate-b-profile.json')['results'];h=load('gate-b-headroom.json')
for r in p:
 assert r['provider_requests_after_failed_summary']==0
 assert r['negative_adoption']['provider_submissions']==0 and r['negative_adoption']['second_summary_attempts']==0
 assert not r['summary_succeeded'] and not r['replacement']
 assert r['latest_header']['config']['maxTokens']==1200
 assert r['latest_header']['tools'][0]['parameters']['required']==['file_path']
 assert r['actual_image_prices']['represented'][0]['visualTokens']==184
 assert r['max_local_storage_bytes']<67108864
for name in ['concise','representative','usage-low','usage-high']:
 r=next(r for r in p if r['scenario']==name);assert r['read_count']==6 and len(r['offloads'])==3
 assert r['invocations'][0]['pressure']<8192
 assert r['invocations'][-1]['phase']=='trigger'
 assert r['selected_range']['measurement']['totalTokens']>=8192
assert h['effective']['adapter']['models'][0]['contextWindow']==1048576
assert h['effective']['adapter']['retryPolicy']['maxRetries']==0
assert h['round3'][0]['credible_no_second_pressure_trigger'] and h['round3'][1]['credible_no_second_pressure_trigger']
assert not h['round3'][2]['entry_screen_pass'] and not h['round3'][-1]['credible_no_second_pressure_trigger']
c=load('gate-c-layout.json');assert c['pass_'] and c['cases'][0]['end']==32773
assert c['cases'][0]['existing_projected_pair_bytes']==32292564
assert c['cases'][0]['conservative_padded_23_equivalents_bytes']==1698473272
assert load('gate-c-admission.json')['fresh_patch_limit']==36864
assert max(x['prepared_plus_output'] for x in load('gate-c-admission.json')['cases'])==25614
assert '64 passed' in (OUT/'python-tests.log').read_text()
assert '6 passed' in (OUT/'authority-tests.log').read_text()
assert '2 passed' in (OUT/'growth-tests.log').read_text()
assert load('integrity-after.json')['original_runtime_paths']==817
assert not subprocess.check_output(['git','diff','HEAD','--','ds41f_mlx'],cwd=ROOT,text=True)
result=dict(kind='MODEL_FREE_READINESS_NOT_NATIVE_ACCEPTANCE',starting_head='2cb84bf06b2eb72e41effe65e484e66b655b912e',gates={'A':'PASS','B':'PASS','C':'PASS'},overall='READY',step4b_technically_ready_for_separate_authorization=True,step4b_authorized=False,step4b_started=False,weights_loaded=0,production_changes=False,frozen_contract_revisions=[],frozen_step4a_inputs_verified=checked,original_operational_paths=817,effective_threshold=8192,round3_entry_limit=4608,budgets_sha256=digest(OLD/'budgets.json'),ordinary_projected_pair_bytes=32292564,padded_coexistence_bytes=1698473272,remaining_native_only=['real summary completion, strict shrink and persisted replacement','actual pressure after required ordinary reconstruction and before Round3','native paired/Vision/protocol/settlement/publication joins including fresh historical Vision after offload','native physical/MLX residency and ownership retirement','separate C7 same-process Backend/Model weakref death and MLX reclamation remains PARTIAL'],C8='NOT QUALIFIED; no native session in this task',C2='PARTIAL; numerical distinctions unchanged',C7='PARTIAL; FastAPI LRU diagnosis preserved, no lifetime/cache surgery')
(OUT/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
inputs={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='inputs.sha256.json'}
inputs['../phase4-step4ar1-readiness.md']=digest(OUT.parent/'phase4-step4ar1-readiness.md')
inputs['../../../docs/multimodal-runtime-plan.md']=digest(ROOT/'docs/multimodal-runtime-plan.md')
(OUT/'inputs.sha256.json').write_text(json.dumps(inputs,indent=2)+'\n')
print('PASS: Gates A/B/C; frozen budgets and historical inputs unchanged; READY for separate authorization only')
