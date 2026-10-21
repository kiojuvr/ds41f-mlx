"""Profile-specific publication and fail-closed GLM selection boundaries."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ds41f_mlx import glm_profile as distribution
from ds41f_mlx.runtime import resource_admission as admission


def test_m3_ultra_local_default_is_explicit_native_with_portable_alternative():
    assert distribution.default_profile('mtp') == 'native'
    with pytest.raises(ValueError, match='retired'):
        distribution.default_profile('standard-off')
    authority = json.loads((distribution.SOURCES/'execution-profiles.json').read_text())
    assert 'standard-off' not in authority['defaults']
    for serving in ('mtp',):
        assert authority['status'][serving]['portable'] == 'explicit_alternative'
        assert authority['status'][serving]['native'] == 'production_default'


def test_repository_bundle_provisioning_is_exact_and_separate(tmp_path):
    root = tmp_path/'omlx'/distribution.PREFIX
    root.mkdir(parents=True)
    distribution.provision(tmp_path, 'portable')
    assert not list(root.iterdir())
    receipt = distribution.bundle_authority()
    distribution.provision(tmp_path, 'native')
    assert {p.name for p in root.iterdir()} == set(receipt['files'])
    for name, expected in receipt['files'].items():
        admission.check_file(root/name, expected, name)
    with pytest.raises(ValueError, match='extension-absent'):
        distribution.provision(tmp_path, 'portable')


def test_selected_native_cannot_use_portable_even_with_no_files(tmp_path):
    fast = SimpleNamespace(__file__=str(tmp_path/'fast.py'), _ext=None)
    pin = dict(glm_profiles={'native': {}, 'portable': {}})
    with pytest.raises(admission.ResourceAdmissionError, match='fallback refused'):
        admission.check_glm_bundle(fast, pin, selected='native')
    assert admission.check_glm_bundle(fast, pin, selected='portable') == 'portable'


def test_present_but_abi_disabled_native_cannot_fall_back(tmp_path):
    (tmp_path/'_ext.so').write_bytes(b'present but unloadable')
    fast = SimpleNamespace(__file__=str(tmp_path/'fast.py'), _ext=None)
    pin = dict(glm_profiles={'native': {}, 'portable': {}})
    with pytest.raises(admission.ResourceAdmissionError, match='fallback refused'):
        admission.check_glm_bundle(fast, pin, selected='native')
    with pytest.raises(admission.ResourceAdmissionError, match='mixed GLM'):
        admission.check_glm_bundle(fast, pin, selected='portable')


def test_native_partial_bundle_rejected_before_load(tmp_path):
    fast = SimpleNamespace(__file__=str(tmp_path/'fast.py'), _ext=object())
    pin = dict(glm_profiles={'native': {'missing.metallib': {}}, 'portable': {}})
    with pytest.raises(admission.ResourceAdmissionError, match='mixed GLM'):
        admission.check_glm_bundle(fast, pin, selected='native')


def test_native_requires_extension_symbols_not_mlx_capability(tmp_path):
    fast = SimpleNamespace(__file__=str(tmp_path/'fast.py'), _ext=SimpleNamespace(),
                           NATIVE_SYMBOLS=('required_native_entry',))
    pin = dict(glm_profiles={'native': {}, 'portable': {}})
    with pytest.raises(admission.ResourceAdmissionError, match='symbols missing'):
        admission.check_glm_bundle(fast, pin, selected='native')
