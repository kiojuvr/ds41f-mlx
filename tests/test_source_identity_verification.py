"""Executed source assertions: explicit methods, whole-file pins and fail-closed inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tools import check_source_identity_hashes as check
from tools.source_identity import AST_HASH_METHOD, SOURCE_HASH_METHOD, source_sha256


def fixture(tmp_path, method=SOURCE_HASH_METHOD):
    source = tmp_path/'official.py'
    source.write_bytes(b'class Reference:\n    def operation(self):\n        return 1\n')
    record = dict(file=str(source), file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_lines=[2, 3], source_hash_method=method,
                  source_sha256=source_sha256(source, 2, 3, method=method))
    path = tmp_path/'receipt.json'
    path.write_text(json.dumps(record))
    return source, path, record


@pytest.mark.parametrize('method', [SOURCE_HASH_METHOD, AST_HASH_METHOD])
def test_reproducible_declared_method(tmp_path, method):
    source, path, record = fixture(tmp_path, method)
    assert check.verify_artifact(path, tmp_path) == 1
    assert source_sha256(source, 2, 3, method=AST_HASH_METHOD) != source_sha256(source, 2, 3)


@pytest.mark.parametrize('fault', ['source', 'digest', 'file_digest', 'method', 'missing_method', 'span', 'missing_source'])
def test_source_fault_is_not_a_pass(tmp_path, fault):
    source, path, record = fixture(tmp_path)
    if fault == 'source': source.write_text(source.read_text().replace('return 1', 'return 2'))
    elif fault == 'digest': record['source_sha256'] = '0'*64
    elif fault == 'file_digest': record['file_sha256'] = '0'*64
    elif fault == 'method': record['source_hash_method'] = 'guess_any_matching_method'
    elif fault == 'missing_method': del record['source_hash_method']
    elif fault == 'span': record['source_lines'] = [2, 300]
    else: source.unlink()
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, OSError)):
        check.verify_artifact(path, tmp_path)


def test_historical_method_requires_original_artifact_bytes(tmp_path, monkeypatch):
    source, path, record = fixture(tmp_path, AST_HASH_METHOD)
    del record['source_hash_method']
    path.write_text(json.dumps(record))
    original = path.read_bytes()
    historical = {path.as_posix(): dict(revision='recorded-revision', methods={
        f'{source}:2:3': AST_HASH_METHOD})}
    calls = []
    def blob(args, **kwargs):
        calls.append(args)
        return original
    monkeypatch.setattr(check.subprocess, 'check_output', blob)
    assert check.verify_artifact(path, tmp_path, historical) == 1
    assert calls[0] == ['git', 'show', f'recorded-revision:{path.as_posix()}']
    path.write_bytes(original+b' ')
    with pytest.raises(ValueError, match='differs from original'):
        check.verify_artifact(path, tmp_path, historical)


def test_no_source_assertions_is_failure(tmp_path):
    path = tmp_path/'empty.json'; path.write_text('{}')
    assert check.main([str(path)]) == 1


def test_wording_is_independent_and_missing_inputs_fail(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path/'wording.md'; path.write_text('Implementation compatibility is not official correctness.\n')
    command = [sys.executable, str(root/'tools/check_authority_labels.py')]
    result = subprocess.run(command+[str(path)], capture_output=True, text=True, check=True)
    assert '1 files examined' in result.stdout and 'wording only' in result.stdout
    path.write_text('oMLX remains authoritative\n')
    assert subprocess.run(command+[str(path)], capture_output=True).returncode == 1
    assert subprocess.run(command+[str(tmp_path/'absent')], capture_output=True).returncode == 1
