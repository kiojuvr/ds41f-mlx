"""Record M32's real full-native gate and uncovered initialization boundary."""
import ctypes
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
import sys
import deepseek_recipe._native as native

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/m32'
RECIPE = Path('/tmp/ds41f-m32-recipe')
BASE = Path('/tmp/ds41f-m31-base')
OMLX = Path('/tmp/ds41f-m32-omlx')
PIN = '8cadfede7063c896b944e7bae05daa3549ae97ea'
OMLX_PIN = '4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40'

def command(*args):
    return subprocess.check_output(args, text=True).strip()
def git(path, *args):
    return command('git', '-C', str(path), *args)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name, obj):
    (OUT / (name + '.json')).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

assert git(BASE, 'rev-parse', 'HEAD') == PIN
assert git(OMLX, 'rev-parse', 'HEAD') == OMLX_PIN
assert not git(RECIPE, 'status', '--porcelain')
assert not git(OMLX, 'status', '--porcelain')
assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert hasattr(__import__('deepseek_recipe').StreamProcessor, 'semantic_snapshot')
base = json.loads((OUT / 'canonical-base.json').read_text())
patched = json.loads((OUT / 'canonical-patched.json').read_text())
assert base['records'] == patched['records']
assert patched['native_sha256'] == sha(native.__file__)
assert base['native_sha256'] != patched['native_sha256']
preview = json.loads((OUT / 'native-preview.json').read_text())
init = json.loads((OUT / 'init-boundary.json').read_text())
assert preview['source_agreement'] and preview['native_sha256'] == sha(native.__file__)
assert init['native_sha256'] == sha(native.__file__)
assert init['status'] == 'CHAIN_ONLY_CLAMP_INSUFFICIENT_AT_INIT'
assert init['chain_verify_calls'] == 0
sys.path.insert(0, str(OMLX))
import omlx.api.tool_calling as tool_calling
# Enumerate the actual loaded native images, not just otool install names.
dyld = ctypes.CDLL(None)
dyld._dyld_image_count.restype = ctypes.c_uint32
dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
dyld._dyld_get_image_name.restype = ctypes.c_char_p
images = []
for i in range(dyld._dyld_image_count()):
    path = Path(dyld._dyld_get_image_name(i).decode())
    if str(path).startswith('/opt/homebrew') and path.is_file():
        images.append(dict(path=str(path), resolved=str(path.resolve()), sha256=sha(path)))
preserved = Path.home() / '.venvs/omlx-0.7.0.release/lib/python3.13/site-packages/deepseek_recipe/_native.abi3.so'
assert sha(preserved) == json.loads((ROOT / 'artifacts/m30/runtime-identities.json').read_text())['native_sha256']
assert git(Path.home()/'omlx-0.7.0.release', 'rev-parse', 'HEAD') == OMLX_PIN
assert not git(Path.home()/'omlx-0.7.0.release', 'status', '--porcelain')
files = git(RECIPE, 'diff', '--name-only', PIN, 'HEAD').splitlines()
identity = dict(ds41f_base_commit='269c392664b384255f0ad627402346b1606fba20', recipe_base=PIN,
                recipe_candidate=git(RECIPE,'rev-parse','HEAD'), m31_recipe_candidate='066d2ef2ed0deb574a0e6b5e6136316d6f296d55',
                recipe_patch_sha256=sha(OUT/'recipe-semantic-preview.patch'), recipe_lock_sha256=sha(RECIPE/'Cargo.lock'),
                base_lock_sha256=sha(BASE/'Cargo.lock'), rustc=command('rustc','-Vv'), cargo=command('cargo','-V'),
                python=sys.version, python_executable=sys.executable, platform=platform.platform(),
                packages={dist.metadata['Name']:dist.version for dist in md.distributions()},
                native_path=native.__file__, native_sha256=sha(native.__file__), native_architecture='arm64',
                wheel_path=str(next(Path('/tmp/ds41f-m32-wheels').glob('*.whl'))),
                wheel_sha256=sha(next(Path('/tmp/ds41f-m32-wheels').glob('*.whl'))),
                opencv_version='4.14.0', opencv_formula='opencv@4', opencv_linkage='external host libraries; skip-auditwheel, NOT portable distribution',
                rust_tokenizers_version='0.23.2', python_tokenizers_version=md.version('tokenizers'),
                loaded_homebrew_libraries=images, omlx_revision=OMLX_PIN, omlx_patch_revision=None,
                omlx_tool_import=tool_calling.__file__, omlx_tool_import_ok=True,
                preserved_release_native_sha256=sha(preserved), preserved_release_unchanged=True,
                recipe_source_hashes={f:sha(RECIPE/f) for f in files}, tokenizer_sha256=sha(RECIPE/'static/tokenizers/v41/tokenizer.json'),
                checkpoint_config_sha256=sha('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/config.json'))
save('runtime-identities',identity)
save('canonical-parity',dict(equal=True, count=len(base['records']), case_count=22,
     comparisons=[dict(case=x['name'],mode=x['mode'],equal=x==y) for x,y in zip(base['records'],patched['records'])],
     scope='Real pristine and candidate native bindings: protocol events, response, finish and usage. Only created timestamps normalized.'))
cases=['one_tool_call','multiple_tool_calls','thinking_to_tool','ordinary_text_OFF_ON','one_token_stop','multi_token_stop','shared_prefix_stop','unicode_adjacent_stop','stop_across_verify_cycles','full_acceptance','partial_acceptance','immediate_rejection','terminal_inside_accepted_prefix','terminal_as_correction_bonus','rejection_before_terminal','DSML_across_verify_cycles','interrupt_before_DSML','interrupt_in_DSML','interrupt_committed_undelivered_prefix','interrupt_before_closure','interrupt_after_terminal','semantic_quiescence','P6_P5_tool_result_reentry']
save('qualification',dict(schema='ds41f.m32.qualification.v1',decision='MTP_SEMANTIC_CLAMP_BLOCKED',
     ds41f_base_commit=identity['ds41f_base_commit'], recipe_candidate=identity['recipe_candidate'], omlx_revision=OMLX_PIN,
     build_used='FULL_STANDARD_NATIVE_BUILD',protocol_only_feature_gate=False, native_binding_gate_passed=True,
     full_protocol_gate_passed=False, full_distribution_release_qualified=False,
     canonical_binding_parity=dict(passed=True,comparisons=64,cases=22), preview_source_agreement=dict(passed=True,rows=77),
     blocker='Native _post_init_mtp forwards first response token before chain verify/clamp. Actual model Hello stop completes at index 0, safe=0, yet all 40 targets and DSpark commit it before emission. Chain-only seam insufficient; safe init/queued-terminal ownership not implemented.',
     mtp_semantic_clamp_implemented=False, terminal_queue_metadata_implemented=False,
     terminal_prediction_emission=dict(native_corpus='PASS',actual_initialization_probe='canonical match, unsafe pre-emission target commit',guarded_queue='NOT IMPLEMENTED'),
     live_qualification={case:dict(status='BLOCKED_SEMANTIC_COMMIT_BOUNDARY',target_frontiers=None,dspark_frontiers=None,parser_state=None,acceptance_topology=None) for case in cases},
     native_preview_latency=preview['preview_latency'], live_preview_cost=None, live_clone_latency=None,
     guarded_mtp_throughput=None, guarded_mtp_acceptance=None, first_token_latency=None,
     model_replay_repack_proof=None, semantic_quiescence_counters=None,
     counter_scope='No qualified semantic clamp/quiescence/reentry run: do NOT interpret unmeasured counters as zero proof.',
     tool_result_reentry='NOT RUN', regression_results=dict(rust='10 passed',python_binding='30 passed',selected_ds41f='65 passed, 24 subtests passed; one Pydantic deprecation warning', m32_evidence='4 passed', final_combined='99 passed, 24 subtests passed; one Pydantic deprecation warning'),
     production_mtp='OFF',public_mtp_option='disabled',mtp_persistence='fail closed',token_exact_immediate_abort='unsupported',
     operational_soak='NOT RUN',m33_authorized=False))
print(json.dumps(dict(candidate=identity['recipe_candidate'],native=identity['native_sha256'],libraries=len(images),decision='MTP_SEMANTIC_CLAMP_BLOCKED'),indent=2))
