"""Retired OFF production boundaries must fail before allocation or delivery."""
import json
import subprocess
import sys

import pytest

from ds41f_mlx import execution_identity as identity
from tools import build_execution_package as packaging


@pytest.mark.parametrize('module,args', [
    ('ds41f_mlx.serve', ['--profile', 'standard-off', '--print-config']),
    ('ds41f_mlx.ops', ['start', '--profile', 'standard-off']),
    ('ds41f_mlx.mtp_setup', ['--profile', 'standard-off', '--venv', '/unused', '--build-dir', '/unused']),
    ('tools.build_execution_package', ['--profile', 'standard-off', '--output', '/unused']),
])
def test_retired_off_cli_rejected(module, args):
    proc = subprocess.run([sys.executable, '-m', module, *args], text=True, capture_output=True)
    assert proc.returncode != 0
    assert 'invalid choice' in proc.stderr and 'standard-off' in proc.stderr


def test_retired_off_admission_and_app_rejected(tmp_path, monkeypatch):
    from ds41f_mlx.delivery import inspect
    with pytest.raises(ValueError, match='retired'):
        inspect(None)
    authority = tmp_path/'manifest.json'
    authority.write_text(json.dumps(dict(profile='standard-off')))
    monkeypatch.setattr(identity, 'MANIFEST', authority)
    with pytest.raises(ValueError, match='retired'):
        identity.inspect_payload()
    with pytest.raises(ValueError, match='retired'):
        packaging.build_wheel(dict(profile='standard-off'), tmp_path)
    from ds41f_mlx.serving.server import create_app
    with pytest.raises(ValueError, match='retired'):
        create_app(profile='standard-off')
    from ds41f_mlx.release_acceptance import main
    with pytest.raises(SystemExit, match='retired'):
        main([])
