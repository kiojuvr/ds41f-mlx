"""Validate final gate evidence and record isolated runtime/source identities."""
import dataclasses
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/m33'
OMLX=Path('/tmp/ds41f-m33-omlx');RECIPE=Path('/tmp/ds41f-m32-recipe')
sys.path[:0]=[str(ROOT),str(OMLX)]
import deepseek_recipe._native as native
import omlx.api.tool_calling as tools
from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.mlx_lm_mtp.semantic_horizon import Prediction,OwnedTerminal
BASE='fff84bdd1acd814ae84b994bec4ed110f31f4bb1'
OPIN='4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40'
OCAND='fbe18e8fe68e5bb7b9b1971652ed330f752b6afc'
RPIN='8cadfede7063c896b944e7bae05daa3549ae97ea'
RCAND='29dabb5a55b7b2c6a68e18bbb3eb14495623e81a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
def load(name):return json.loads((OUT/(name+'.json')).read_text())
def save(name,value):(OUT/(name+'.json')).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
assert git(OMLX,'rev-parse','HEAD')==OCAND and not git(OMLX,'status','--porcelain')
assert git(RECIPE,'rev-parse','HEAD')==RCAND and not git(RECIPE,'status','--porcelain')
assert not git(ROOT,'diff',BASE,'--',*[f'artifacts/m{n}' for n in range(25,33)])
m32=json.loads((ROOT/'artifacts/m32/runtime-identities.json').read_text())
assert sha(native.__file__)==m32['native_sha256']
release=Path.home()/'omlx-0.7.0.release'
assert git(release,'rev-parse','HEAD')==OPIN and not git(release,'status','--porcelain')
preserved=Path.home()/'.venvs/omlx-0.7.0.release/lib/python3.13/site-packages/deepseek_recipe/_native.abi3.so'
assert sha(preserved)==m32['preserved_release_native_sha256']
release_artifacts=json.loads((ROOT/'artifacts/m20/release-environment.json').read_text())['native_artifacts']
assert all(sha(a['path'])==a['sha256'] for a in release_artifacts)
patch=subprocess.check_output(['git','-C',str(OMLX),'diff','--binary',OPIN,OCAND])
(OUT/'omlx-semantic-horizon.patch').write_bytes(patch)
source_files=git(OMLX,'diff','--name-only',OPIN,OCAND).splitlines()
source_hashes={f:sha(OMLX/f) for f in source_files}
identity=dict(schema='ds41f.m33.runtime-identities.v1',ds41f_base_commit=BASE,
    ds41f_final_commit={'resolver':'git log -1 --format=%H -- artifacts/m33/qualification.json','reason':'avoid impossible self-referential committed SHA; final SHA also reported in completion'},
    recipe_base=RPIN,recipe_candidate=RCAND,recipe_native_path=native.__file__,recipe_native_sha256=sha(native.__file__),
    recipe_binding='FULL_BINDING_QUALIFIED: unchanged M32 full standard ARM64/OpenCV host-linked binding',
    recipe_build_provenance='artifacts/m32/runtime-identities.json',recipe_lock_sha256=sha(RECIPE/'Cargo.lock'),
    recipe_patch_sha256=m32['recipe_patch_sha256'],opencv_native_version=m32['opencv_version'],
    rust_tokenizers_version='0.23.2',python_tokenizers_version=md.version('tokenizers'),
    omlx_base=OPIN,omlx_candidate=OCAND,omlx_path=str(OMLX),omlx_patch_sha256=sha(OUT/'omlx-semantic-horizon.patch'),omlx_source_hashes=source_hashes,
    actual_tool_import=tools.__file__,actual_tool_import_ok=True,candidate_optional_cpp_dsa_available=fast.is_native_available(),
    model_kernel_scope='stock pinned model MLX/source-JIT kernels; no borrowed release C++ extension and no claim of a new compiled C++ model variant',
    checkpoint='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash',checkpoint_config_sha256=m32['checkpoint_config_sha256'],
    tokenizer_sha256=sha(RECIPE/'static/tokenizers/v41/tokenizer.json'),python=sys.version,python_executable=sys.executable,
    platform=platform.platform(),rustc=subprocess.check_output(['rustc','-Vv'],text=True).strip(),
    packages={n:md.version(n) for n in ['deepseek-recipe','tokenizers','mlx','mlx-lm','omlx','jsonschema','pytest']},DOCS_RS=os.environ.get('DOCS_RS'),
    preserved_release_native_sha256=sha(preserved),preserved_release_cpp_artifacts=release_artifacts,preserved_release_unchanged=True,
    historical_artifacts_m25_m32_unchanged=True)
assert identity['DOCS_RS'] is None
assert identity['recipe_lock_sha256']==m32['recipe_lock_sha256']
assert identity['checkpoint_config_sha256']==sha(Path(identity['checkpoint'])/'config.json')
assert str(OMLX) in tools.__file__
local_sources=[
    'ds41f_mlx/runtime/recipe_semantic_guard.py','ds41f_mlx/runtime/mtp_lifecycle.py','ds41f_mlx/prefill_fp8_mlx/handoff.py',
    'tools/m33_semantic_tests.py','tools/m33_semantic_extra_tests.py','tools/m33_recipe_clone_driver.rs',
    'tools/run_m33_live_semantics.py','tools/run_m33_interruptions.py','tools/run_m33_tool_reentry.py','tools/run_m33_guard_bench.py',
    'tools/record_m33_evidence.py','tests/test_m33_protocol_evidence.py',
    'docs/milestone-33-semantic-horizon.md','docs/milestone-33-protocol-qualification.md']
identity['ds41f_source_hashes']={p:sha(ROOT/p) for p in local_sources}
save('runtime-identities',identity)
parity=load('canonical-parity');preview=load('native-preview');clones=load('source-preview-clone')
assert parity['equal'] and parity['comparisons']==64
assert preview['source_agreement'] and preview['nonmutation'] and preview['row_count']==77
live_names=['init-live','tools-live','stops-live']
all_cases=[]
for name in live_names:
    result=load(name);assert result['status']=='PASS' and result['omlx_revision']==OCAND
    for c in result['cases']:
        assert c['status']=='PASS'
        if c['mtp']:
            q=c['quiescence'];assert q['dspark']['offsets']==[q['canonical_frontier']]*3
            # The primitive and live harness check all 40 final target offsets
            # before returning PASS; the final emit records the unforwarded
            # terminal frontier explicitly.
            assert c['emits'][-1]['target']==[q['canonical_frontier']-1]*40
            assert q['counters']['target_forwards']==1
            assert c['quiescence_actual_counter_delta']['verify']==c['quiescence_actual_counter_delta']['proposal']==0
        all_cases.append(c)
interrupt=load('interruptions');reentry=load('tool-reentry');perf=load('performance')
assert interrupt['status']==reentry['status']==perf['status']=='PASS'
assert len(interrupt['cases'])==16
assert sum(c['status']=='PASS_FAIL_CLOSED' for c in interrupt['cases'])==3
assert reentry['counts']['replay']==reentry['counts']['repack']==0
assert all(s['P7_events'] for s in reentry['suffixes'])
assert len(reentry['turns'])==2 and reentry['turns'][0]['protocol']['choices'][0]['finish_reason']=='tool_calls'
assert reentry['turns'][1]['protocol']['choices'][0]['finish_reason']=='stop'
assert perf['M26_class_materially_intact']
# Size only the response-associated prediction/owner graph, not parser, model,
# canonical protocol output, or the existing cache/ring references.
def tree_size(obj,seen=None):
    seen=set() if seen is None else seen
    if id(obj) in seen:return 0
    seen.add(id(obj));n=sys.getsizeof(obj)
    if dataclasses.is_dataclass(obj):return n+tree_size(vars(obj),seen)
    if isinstance(obj,dict):return n+sum(tree_size(k,seen)+tree_size(v,seen) for k,v in obj.items())
    if isinstance(obj,(tuple,list)):return n+sum(tree_size(v,seen) for v in obj)
    return n
owners=[]
for c in all_cases:
    for cycle in c['cycles']:
        if cycle['terminal']:owners.append(cycle['terminal'])
for c in interrupt['cases']:
    if c.get('before',{}).get('prediction'):owners.append(c['before']['prediction'])
    if 'alignment' in c:owners.append(c['alignment']['prediction'])
sizes=[]
for o in owners:
    p=o['prediction'];pred=Prediction(p['kind'],p['index'],p['safe_count'],p['token_id'],tuple(p['identity']))
    value=OwnedTerminal(o['ordinal'],pred,o['region'],tuple(o['candidate_ids']),o['queued_prefix'],o['owner_uid'])
    sizes.append(tree_size(value))
metadata=dict(current_owned_terminal_limit=1,decision_history_limit=16,actual_owner_samples=len(owners),
              max_owned_graph_bytes=max(sizes),max_owned_json_bytes=max(len(json.dumps(o).encode()) for o in owners),
              scope='recursive unique Python object sizing of reconstituted actual owner records, excludes shared model/parser/output/cache objects',
              actual_max_candidate_ids=max(len(o['candidate_ids']) for o in owners),max_preview_diagnostic_rows=64,max_preview_timing_samples=4096)
save('metadata-overhead',metadata)
reg=(OUT/'regressions.log').read_text();release_reg=(OUT/'release-off-runtime-regressions.log').read_text();rust=(OUT/'rust-tests.log').read_text()
assert '139 passed, 24 subtests passed' in reg and '5 passed' in release_reg and '10 passed; 0 failed' in rust
summary=dict(schema='ds41f.m33.qualification.v1',decision='PROTOCOL_GATE_SOLVED',binding='FULL_BINDING_QUALIFIED',
    authorized_next='M34 operational qualification only',production_mtp_default='OFF',public_mtp_option='disabled',
    mtp_persistence='fail closed',token_exact_immediate_abort='unsupported',broad_operational_soak=False,public_serving_promotion=False,
    identities='artifacts/m33/runtime-identities.json',native_parity=dict(cases=22,comparisons=64,equal=True),
    native_preview=dict(rows=77,repeated_samples=15400,nonmutation=True),
    rust_tests=10,candidate_python_tests=139,release_off_runtime_tests=5,subtests=24,
    semantic_tests=29,live_matrix_cases=len(all_cases),interruptions=dict(cases=16,exact_recovery=13,deliberate_internal_phase_fail_closed=3),
    tool_result_reentry=dict(turns=2,P7_enabled=True,P6_P5_same_forward_taps=True,model_history_replay=0,full_cache_repack=0),
    performance=dict(qualified_scope='single paired M26-shape 4K/128 benchmark; no operational soak',
        rates={r['mode']:r['tok_s'] for r in perf['runs']},guard_vs_no_guard_ratio=perf['guard_vs_no_guard_ratio'],
        native_preview_latency=preview['preview_latency'],same_core_source_clone_latency=clones['clone_latency'],
        standalone_python_binding_clone_timer=None,clone_timer_scope=clones['scope'],live_preview=perf['runs'][-1]['preview'],
        metadata_overhead=metadata,acceptance_metric='unchanged upstream considered-draft counter metric, same as M26'),
    proof_scope='non-mutating canonical forks, exact owner/ordinal/token/span match; safe-prefix native rollback and append-only context; bounded quiescence only after canonical emission',
    excluded_experiments=['init-live-unmatched-eos','stops-live-fixture-miss','tool-reentry-history-alias-error','performance-supervisor-timeout','regressions-initial-mixed-root-error','semantic-unit-fixture-error','alignment-live (pre-final candidate)','control-policy-config-error','tool-reentry-control-name-error','tool-reentry-missing-matcher-error'])
save('qualification',summary)
# Evidence files immutable after this manifest is produced; no circular hashes.
files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='file-manifest.json')
save('file-manifest',{'schema':'ds41f.m33.file-manifest.v1','sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}})
print(json.dumps(summary,indent=2))
