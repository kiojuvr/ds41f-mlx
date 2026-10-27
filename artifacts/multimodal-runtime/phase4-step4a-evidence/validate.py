"""Planning receipt consistency only. No serving or model construction."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def load(n):return json.loads((OUT/n).read_text())
b=load('budgets.json');m=load('images.json');o=load('ordering.json');a=load('admission.json')
assert len(m)==len({x['sha256'] for x in m})==9
for x in m:
 raw=(ROOT/x['source']).read_bytes();assert hashlib.sha256(raw).hexdigest()==x['sha256']
assert b['ordinary_generations']==4+4+4+1+5+1
assert b['total_output_allocation']==19*1200+2048==24848
assert b['adapter_attempts_including_offload']==19+1+7
assert b['runtime_http_requests']==b['completed_generations']==20
assert b['session_seconds']>=20*b['generation_seconds']+600
assert b['overall_seconds']==b['startup_seconds']+b['session_seconds']
assert b['sequential_headless_invocations']==2+4+1+1+1
assert b['official_budget_retries']==b['image_offload_transitions']==10-3
assert b['read_image_effects']==10 and b['auxiliary_summary_attempts']==1
assert b['attachment_bytes']>sum(x['bytes'] for x in m)
assert o['options']['retryPolicy']['maxRetries']==0
assert o['info']['context']['contextWindow']==1048576
r=o['records'][1];assert r['spec']['thresholdTokens']==8192 and r['spec']['retainTokens']==1024
assert r['round1']<8192 and r['round2']<8192
assert r['attempts'][0]['reads']==6 and len(r['offloads'])==3
assert r['attempts'][0]['selected_tokens']==6695
assert max(x['prepared_plus_output'] for x in a['cases'])<=b['prepared_plus_reserved_output_positions']
assert max(x['body_bytes'] for x in a['cases'])<b['wire_body_bytes']
assert max(x.get('patches',0) for x in a['cases'])==6084<36864
assert load('stop-checks.json')['pass']
assert load('zero-weight.json')['proof']['actual_pre_woa_materialized']
assert load('integrity-before.json')['original_runtime_paths']==817
assert '41 passed' in (OUT/'admission-tests.log').read_text()
assert 'passed' in (OUT/'paired-accounting-tests.log').read_text()
# Do not silently declare the old native gate ready for ten effects.
readiness=dict(disposition='PARTIAL',weights_loaded=0,production_changes=False,chosen_threshold=8192,manifest_images=9,total_fixture_bytes=sum(x['bytes'] for x in m),controlled_offloads_before_summary=3,controlled_pressure=8258,controlled_selected_tokens=6695,conditional_discount_required=False,max_fresh_runtime_patches=4563,conservative_four_image_control_patches=6084,not_native_success=True,unverified_prerequisites=['Existing Step3B-R3 task guard allows only A/B, two effects, 4096 threshold; nine-image/ten-effect/8192 extension not implemented or qualified here','Controlled Agent uses small no-effect tool declaration and no real native usage anchors; exact full-profile pressure/headroom and final Round3 single-compaction route not yet qualified model-free','No current Step4B disposable installed payload/full checkpoint qualification; dependency zero-weight pre_woa is not that gate'],step4b_authorized=False)
(OUT/'validation.json').write_text(json.dumps(readiness,indent=2)+'\n')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name not in ['inputs.sha256.json']}
files['../phase4-step4a-acceptance-contract.md']=hashlib.sha256((OUT.parent/'phase4-step4a-acceptance-contract.md').read_bytes()).hexdigest()
files['../../../docs/multimodal-runtime-plan.md']=hashlib.sha256((ROOT/'docs/multimodal-runtime-plan.md').read_bytes()).hexdigest()
(OUT/'inputs.sha256.json').write_text(json.dumps(files,indent=2)+'\n')
print('PASS: proposed budgets, identities, actual ordering/range and refusal-fence consistency; readiness PARTIAL, no native authorization')
