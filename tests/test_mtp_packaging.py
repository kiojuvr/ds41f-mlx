"""Ordinary development delivery, independent of an installed oMLX or GPU."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from ds41f_mlx import execution_identity as identity
from ds41f_mlx.mtp_setup import SOURCES, apply_mtp_patches
from tools import build_execution_package as packaging


def export_omlx(destination):
    with tarfile.open(SOURCES/'omlx-source.tar.gz') as archive:
        archive.extractall(destination, filter='data')


def test_strict_overlay_chain_and_reapplication(tmp_path):
    export_omlx(tmp_path)
    apply_mtp_patches(tmp_path)
    model = (tmp_path/'omlx/patches/deepseek_v41/language.py').read_text()
    assert 'grouped = pre_woa(out.reshape(' in model
    with pytest.raises(subprocess.CalledProcessError):
        apply_mtp_patches(tmp_path)


def test_incompatible_patch_context_fails(tmp_path):
    export_omlx(tmp_path)
    path = tmp_path/'omlx/patches/deepseek_v41/language.py'
    path.write_text('incompatible source\n')
    with pytest.raises(subprocess.CalledProcessError):
        apply_mtp_patches(tmp_path)


def fixture_authority(tmp_path, monkeypatch):
    package = tmp_path/'example'
    package.mkdir()
    (package/'__init__.py').write_text('value = 1\n')
    monkeypatch.setattr(identity.importlib.util, 'find_spec', lambda name:
                        SimpleNamespace(submodule_search_locations=[str(package)]))
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '1.0')
    return dict(schema='ds41f.installed-admission.v3', profile='mtp', files={},
                dependencies={'example': '1.0'}, payloads={'example': {}})


def test_mtp_python_changes_and_bytecode_are_not_sealed(tmp_path, monkeypatch):
    authority = fixture_authority(tmp_path, monkeypatch)
    monkeypatch.setattr(identity, 'verify_tree', lambda *args: pytest.fail('OFF-only verifier'))
    identity.check_installed_packages(authority, tmp_path)
    (tmp_path/'example/__init__.py').write_text('value = 2\n')
    (tmp_path/'example/added.py').write_text('value = 3\n')
    cache = tmp_path/'example/__pycache__'
    cache.mkdir()
    (cache/'__init__.cpython-313.pyc').write_bytes(b'not a fingerprint authority')
    identity.check_installed_packages(authority, tmp_path)


def test_mtp_versions_origins_and_native_integrity_remain(tmp_path, monkeypatch):
    authority = fixture_authority(tmp_path, monkeypatch)
    authority['dependencies']['example'] = '2.0'
    with pytest.raises(ValueError, match='version drift'):
        identity.check_installed_packages(authority, tmp_path)
    authority['dependencies']['example'] = '1.0'
    with pytest.raises(ValueError, match='origin'):
        identity.check_installed_packages(authority, tmp_path/'other')
    binary = tmp_path/'example/native.so'
    binary.write_bytes(b'native')
    authority['payloads']['example']['native.so'] = packaging.sha(binary)
    identity.check_installed_packages(authority, tmp_path)
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError, match='native payload drift'):
        identity.check_installed_packages(authority, tmp_path)


def test_mtp_refuses_legacy_or_source_sealed_manifests(tmp_path, monkeypatch):
    authority = fixture_authority(tmp_path, monkeypatch)
    authority['schema'] = 'ds41f.installed-admission.v2'
    with pytest.raises(ValueError, match='rebuild MTP'):
        identity.check_installed_packages(authority, tmp_path)
    authority['schema'] = 'ds41f.installed-admission.v3'
    authority['payloads']['example']['__init__.py'] = 'obsolete seal'
    with pytest.raises(ValueError, match='must not seal Python'):
        identity.check_installed_packages(authority, tmp_path)


def test_off_still_checks_installed_python_payload(tmp_path, monkeypatch):
    authority = fixture_authority(tmp_path, monkeypatch)
    authority.update(profile='standard-off', schema='ds41f.installed-admission.v2')
    authority['payloads']['example'] = identity.inventory(tmp_path/'example')
    identity.check_installed_packages(authority, tmp_path)
    (tmp_path/'example/__init__.py').write_text('value = 2\n')
    with pytest.raises(ValueError, match='payload drift'):
        identity.check_installed_packages(authority, tmp_path)


def test_mtp_manifest_has_no_python_seals(monkeypatch):
    old = json.loads((SOURCES/'normal-local.json').read_text())
    # Recipe native payloads come from the retained repaired wheel, not Homebrew.
    manifest = packaging.mtp_manifest(old, 'portable')
    assert manifest['payloads']['ds41f_mlx'] == {}
    assert manifest['payloads']['omlx'] == {}
    assert all(path.endswith(('.so', '.dylib', '.metallib'))
               for files in manifest['payloads'].values() for path in files)
    assert all(value is None for value in manifest['files'].values())
    assert manifest['dependencies']['mlx-lm'] == old['packages']['mlx-lm']
    assert manifest['checkpoint']['shards'][0]['lfs_sha256'] != old['checkpoint']['shards'][0]['lfs_sha256']
    assert not any('sha256' in key for key in manifest['provenance'])


def test_staged_build_does_not_rewrite_checkout(tmp_path, monkeypatch):
    before = identity.MANIFEST.read_bytes()
    old = json.loads((SOURCES/'normal-local.json').read_text())
    manifest = packaging.mtp_manifest(old, 'portable')
    def fake_build(command, check):
        stage = Path(command[-1])
        assert json.loads((stage/'ds41f_mlx/execution_manifest.json').read_text()) == manifest
        assert not list(stage.rglob('*.pyc'))
        (tmp_path/'ds41f_mlx-test.whl').touch()
    monkeypatch.setattr(packaging.subprocess, 'run', fake_build)
    packaging.build_wheel(manifest, tmp_path)
    assert identity.MANIFEST.read_bytes() == before
    with pytest.raises(ValueError, match='prior Runtime wheels'):
        packaging.build_wheel(manifest, tmp_path)


@pytest.mark.parametrize('name', ['flash', 'flash-1017'])
def test_package_fixed_variant(name):
    from ds41f_mlx.mtp_identity import VARIANTS, checkpoint_variant
    old = json.loads((SOURCES/'normal-local.json').read_text())
    manifest = packaging.mtp_manifest(old, 'portable', name)
    assert checkpoint_variant(manifest) == VARIANTS[name]
    assert manifest['dependencies']['deepseek-recipe'] == '0.2.0'
    assert {s['source_revision'] for s in manifest['checkpoint']['shards']} == {VARIANTS[name].revision}
    assert all(not files for name, files in manifest['payloads'].items() if name in ('ds41f_mlx', 'omlx'))
