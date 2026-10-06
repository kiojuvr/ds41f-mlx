#!/usr/bin/env python3
"""CPU-only partial tool audit receipt. Never closes Tool Capability Completion."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    'tests/test_web_acquisition.py',
    'tests/test_m19_web_client.py',
    'tests/test_web_application_boundary.py',
    'tests/test_web_budget.py',
    'tests/test_runtime_capacity.py',
    'tests/test_multimodal_contract.py',
    'tests/test_multimodal_restore_ownership.py',
    'tests/test_web_search_unusable.py',
    'tests/test_web_private_lan.py',
]
SOURCES = [
    'ds41f_mlx/web_tools.py', 'ds41f_mlx/web_budget.py',
    'ds41f_mlx/web.py', 'ds41f_mlx/web_client.py',
    'ds41f_mlx/runtime/multimodal.py',
    'ds41f_mlx/serving/capacity.py',
    'ds41f_mlx/web_static/app.js', 'ds41f_mlx/web_static/store.js',
    'ds41f_mlx/web_static/render.js',
    'docs/tool-capability-completion.md', 'docs/dynamic-capability-budget.md',
    'docs/doc-classification.json', 'docs/README.md',
    'tools/qualify_tool_acquisition_audit.py',
    *TESTS, 'tests/web_render.test.cjs', 'tests/web_platform.test.cjs',
]


def hashes():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwrite receipts')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    results = []
    commands = [
        [sys.executable, '-m', 'pytest', '-q', *TESTS],
        ['node', '--test', 'tests/web_render.test.cjs', 'tests/web_platform.test.cjs'],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log = f'checks-{index}.txt'
        (args.output / log).write_text(result.stdout + result.stderr)
        results.append(dict(command=command, returncode=result.returncode, log=log))
    after = hashes()
    passed = before == after and all(item['returncode'] == 0 for item in results)
    receipt = dict(
        schema='ds41f.tool_capability_audit.partial.v1',
        captured_at=datetime.now(timezone.utc).isoformat(),
        baseline=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=after, source_stable=before == after,
        status='CPU_REGRESSIONS_PASS_NOT_ACCEPTANCE' if passed else 'CPU_REGRESSIONS_FAIL_NOT_ACCEPTANCE',
        milestone_complete=False, results=results,
        scope='CPU transport/security/effect bookkeeping and admission regression tests; renderer tests',
        not_run=['real browser/runtime research', 'GPU Vision', 'PDF interpretation',
                 'binary reload/save/fresh restore', '1M', 'full Vision', 'R1', 'release', 'promotion'],
        declared_tools=['web_search', 'fetch_url'], new_dependencies=[],
    )
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], output=str(args.output))))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
