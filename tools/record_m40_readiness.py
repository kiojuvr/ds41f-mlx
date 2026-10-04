"""Record a source/evidence evaluation, never a runtime release qualification.

Uses only stdlib; candidate paths are optional observations, not install defaults.
Run from repository root after the M40 checks. --check verifies recorded repository
identities without requiring the original machine or dependency checkouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/m40/readiness.json'
BASE = 'ebccd8877d22e00ecbd765b67a665de2c04571de'
DECISION = 'NOT_READY_PUBLIC_ADMISSION_AND_REPRODUCIBLE_DELIVERY'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *args):
    p = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def identities():
    paths = set()
    for directory in ('ds41f_mlx', 'native', 'rust/ds41f_api'):
        for p in (ROOT / directory).rglob('*'):
            if (p.is_file() and p.suffix in {'.py', '.rs', '.toml', '.cpp', '.h', '.hpp', '.metal', '.txt'}
                    and not any(x.startswith('build') or x in {'__pycache__', 'target'} for x in p.relative_to(ROOT).parts)):
                paths.add(p)
    for name in ('release/ds41f-release.json', 'pyproject.toml', 'Cargo.toml', 'Cargo.lock',
                 'docs/milestone-40-release-readiness.md', 'docs/api.md', 'docs/operations.md',
                 'tools/record_m40_readiness.py',
                 'tests/test_m40_readiness.py', 'tools/run_m11_tool_boundary_qualification.py'):
        paths.add(ROOT / name)
    for m in ('33', '34', '35', '36', '36r', '37', '38', '39'):
        paths.add(ROOT / f'artifacts/m{m}/qualification.json')
    for name in ('artifacts/m33/runtime-identities.json', 'artifacts/m32/runtime-identities.json',
                 'artifacts/m39/identities.log', 'artifacts/m39/integration.json',
                 'artifacts/m34/baseline-authority-findings.json',
                 'artifacts/m33/omlx-semantic-horizon.patch'):
        paths.add(ROOT / name)
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def observe(path, files):
    return {'location': str(path), 'revision': git(path, 'rev-parse', 'HEAD'),
            'git_status_porcelain': git(path, 'status', '--porcelain'),
            'files': {f: sha(path / f) if (path / f).is_file() else None for f in files}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        recorded = json.loads(OUT.read_text())
        current = identities()
        assert recorded['repository_hashes'] == current, 'M40 evaluation source/evidence is stale'
        for rel, digest in recorded['evaluation_logs'].items():
            assert sha(ROOT / rel) == digest, rel
        assert recorded['decision'] == DECISION and not recorded['release_qualified']
        print('M40 evaluation identities match; not MTP release qualification')
        return
    native = Path('/tmp/ds41f-m32-qual/lib/python3.13/site-packages/deepseek_recipe/_native.abi3.so')
    prior = json.loads((ROOT / 'artifacts/m33/runtime-identities.json').read_text())
    evidence = {}
    for m in ('33', '34', '35', '36', '36r', '37', '38', '39'):
        rel = f'artifacts/m{m}/qualification.json'
        data = json.loads((ROOT / rel).read_text())
        evidence[m] = {'path': rel, 'decision': data.get('decision'), 'sha256': sha(ROOT / rel),
                       'scope': 'historical internal qualification; not installed MTP release acceptance'}
    logs = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(OUT.parent.glob('*.log'))}
    record = {
        'schema': 'ds41f.m40.readiness.v1', 'evaluation_base': BASE,
        'final_commit_resolver': 'git log -1 --format=%H -- artifacts/m40/readiness.json',
        'decision': DECISION, 'architectural_blocker_found': False, 'release_qualified': False,
        'production_default_mtp': 'OFF',
        'document': 'docs/milestone-40-release-readiness.md',
        'evaluation_checks': [
            {'command': 'DS41F_OMLX_PATH=/tmp/ds41f-m33-omlx DS41F_RECIPE_PATH=/tmp/ds41f-m32-recipe /tmp/ds41f-m32-qual/bin/python -m pytest -q tests/test_m39_evidence.py tests/test_m39_lifetime_admission.py tests/test_m37_local_client.py tests/test_m38_client_faults.py tests/test_runtime_config.py tests/test_m22_release_metadata.py tests/test_stateful_request_policy.py tests/test_m25_mtp_boundary.py',
             'result': '65 passed, 8 subtests passed', 'log': 'artifacts/m40/regressions.log'},
            {'command': 'cargo test', 'result': '7 boundary tests passed', 'log': 'artifacts/m40/rust-tests.log'},
            {'command': '/Users/kioju/.venvs/omlx-0.7.0.release/bin/python -m pytest -q tests/test_m20_generation_dependency.py',
             'result': '5 passed, 1 deprecation warning', 'log': 'artifacts/m40/release-off-regressions.log'},
            {'command': '/tmp/ds41f-m32-qual/bin/python -m pytest -q tests/test_m40_readiness.py',
             'result': '5 passed', 'log': 'artifacts/m40/evaluation-tests.log'},
            {'command': '/tmp/ds41f-m32-qual/bin/python tools/check_authority_labels.py',
             'result': 'FAIL preserved historical findings', 'log': 'artifacts/m40/authority-labels.log'},
            {'command': '/tmp/ds41f-m32-qual/bin/python tools/check_source_identity_hashes.py',
             'result': 'FAIL preserved historical findings', 'log': 'artifacts/m40/source-identity-hashes.log'},
            {'command': 'otool -L /tmp/ds41f-m32-qual/lib/python3.13/site-packages/deepseek_recipe/_native.abi3.so',
             'result': 'host-linked library closure observed', 'log': 'artifacts/m40/native-linkage.log'}],
        'method': 'source/evidence audit plus cheap checks; no new model campaign or clean-install claim',
        'profiles': {
            'standard-off': {'status': 'existing release unchanged', 'default': True, 'mtp': 'OFF'},
            'mtp-singleton-v1': {'status': 'proposed only; not selectable', 'default': False,
                'process_wide': True, 'restart_to_switch': True, 'max_live_sessions': 1,
                'max_active_responses': 1, 'mtp': 'guarded ON', 'dspark': 'ON', 'depth': 5,
                'max_body_bytes': 1048576, 'max_prompt_plus_response_tokens': 8192,
                'max_output_tokens': 768, 'max_sequence': str((1 << 64) - 1),
                'max_lifetimes': str((1 << 128) - 1), 'bind': '127.0.0.1 only',
                'authentication': 'none; trusted single-operator local processes only',
                'required_fence': 'X-DS41F-Request-Sequence + SHA256 exact body bytes',
                'supported': ['health/model/profile inspection', 'server-issued create/GET/DELETE',
                              'bounded stateful text Chat Completions', 'canonical recipe SSE',
                              'certified outcome observation', 'stable Python living-client helper'],
                'narrow_tool_subset': 'exact lookup_weather declaration in tools/run_m11_tool_boundary_qualification.py:tool_def',
                'envelope': {'allowlisted_fields': ['model', 'messages', 'temperature', 'reasoning_effort',
                           'tools', 'tool_choice', 'stream', 'stream_options', 'max_tokens'],
                    'temperature': 'omitted or numeric zero', 'reasoning_effort': 'omitted or none',
                    'max_tokens': 'required integer 1..768', 'stop': 'field unavailable',
                    'unknown_fields': 'reject before reservation/mutation',
                    'messages': 'ordinary text recipe messages; exact-prefix retained continuation',
                    'tool_choice': 'auto or named lookup_weather with pinned declaration'},
                'unavailable': ['stateless Chat Completions', 'Responses', 'Messages', 'requested IDs',
                    'persistence/restore', 'browser client', 'general tool schemas', 'nonzero sampling',
                    'arbitrary reasoning', 'request path parameters', 'dynamic profile switching'],
                'new_ingress_policy_candidates': {'body_preparation_slots': 1, 'connections': 8,
                    'body_timeout_s': 30, 'send_stall_timeout_s': 30,
                    'status': 'must implement and qualify, not inherited'},
                'application_terminal_states': ['unrecoverable', 'poisoned', 'stopped lifecycle uncertainty',
                                                'tool_ambiguous', 'expired'],
                'diagnostics': 'disabled default; sanitized bounded metadata only',
                'rust': 'raw fencing header support required before claiming MTP request support'}},
        'dependency_delta': {
            'off_manifest': json.loads((ROOT / 'release/ds41f-release.json').read_text()),
            'mtp_historical_required_identity': {k: prior[k] for k in (
                'omlx_base', 'omlx_candidate', 'omlx_patch_sha256', 'omlx_source_hashes',
                'recipe_base', 'recipe_candidate', 'recipe_patch_sha256', 'recipe_lock_sha256',
                'recipe_native_sha256', 'model_kernel_scope', 'packages', 'python', 'platform',
                'tokenizer_sha256', 'checkpoint_config_sha256')},
            'delivery_policy': 'vendored exact source exports plus base/patch identities; target-native host-linked build',
            'portable_wheels_required': False,
            'checkpoint_metadata_not_full_weight_hash': True},
        'actual_import_observation': json.loads((OUT.parent / 'actual-imports.log').read_text()),
        'observed_candidates': {
            'omlx': observe(Path('/tmp/ds41f-m33-omlx'), list(prior['omlx_source_hashes'])),
            'recipe': observe(Path('/tmp/ds41f-m32-recipe'), ['Cargo.lock', 'static/tokenizers/v41/tokenizer.json']),
            'recipe_native': {'location': str(native), 'sha256': sha(native) if native.is_file() else None}},
        'historical_evidence': evidence, 'repository_hashes': identities(), 'evaluation_logs': logs,
        'ranked_required_blockers': [
            {'rank': 1, 'id': 'PUBLIC_ADMISSION_LOCAL_SECURITY', 'work': 'strict route/envelope/fence, bounded ingress/SSE, sanitized outcomes, loopback/Host/Origin'},
            {'rank': 2, 'id': 'PROVENANCE_REPRODUCIBLE_DELIVERY', 'work': 'export dependencies, native build/link identity, profile manifest, verifiable bundle and fresh provisioning'},
            {'rank': 3, 'id': 'SUPPORTED_CLIENT_OPERATOR_BOUNDARY', 'work': 'explicit process selector, stable helper/manual workflow, raw Rust header support if advertised'},
            {'rank': 4, 'id': 'INSTALLED_COMPOSED_ACCEPTANCE', 'work': 'fresh installed candidate matrix with OFF regression and matched benchmark'}],
        'acceptance_matrix': ['provenance/profile', 'clean startup/readiness', 'strict admission/rejection atomicity',
            'loopback/Host/Origin/ingress', 'fresh and retained text/SSE', 'completed tool/result loop',
            'certified disconnect recovery', 'partial DSML unrecoverable DELETE/fresh', 'controlled poison',
            'lifecycle/effect uncertainty', 'stale identity/exhaustion', 'cancellation/socket pressure/resources',
            'matched performance', 'installed Python helper and advertised Rust boundary', 'OFF regression'],
        'inheritance': 'exact owning source/dependency match only; cheap fixtures rerun; native rebuild reruns parity/representation; final package gates fresh',
        'performance_claim': 'workload-dependent accelerated opt-in; no universal speedup; M33 matched evidence is historical',
        'audit': {'global_checks_pass': False, 'historical_artifacts_rewritten': False,
            'nonblockers': ['historical HEAD hashes/named-ID harnesses', 'old authority labels',
                            'legacy source-hash metadata and mismatch requiring scoped interpretation'],
            'blockers': ['OFF manifest conflates candidate versions', 'native/helper identity coverage',
                         'dependency delivery', 'bundle self-digest', 'OFF-only installed acceptance']},
        'optional_parity': ['general tools', 'browser', 'stateless protocols', 'sampling/reasoning expansion', 'longer context'],
        'deferred': ['default/implicit MTP', 'persistence/restart recovery', 'distributed identity/recovery',
            'crash-safe effects', 'concurrent/shared MTP', 'batching', 'immediate token-exact abort',
            'unbounded backpressure', 'portable native wheels', 'universal partial DSML', '200K HTTP MTP',
            'remote unauthenticated multi-user serving'],
        'next_milestone': 'M41: implement and qualify complete explicit bounded local MTP release candidate',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
