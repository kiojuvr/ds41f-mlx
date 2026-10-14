"""Exact OFF recipe delivery must agree with the existing qualification pin."""
import json
from pathlib import Path

import pytest

from ds41f_mlx import off_recipe, glm_profile


def test_retained_qualified_wheel_matches_existing_pin_and_source():
    path, receipt = off_recipe.artifact_authority()
    assert path.is_file()
    assert receipt['payload']['_native.abi3.so'] == 'c4b17870812f0f03f1d026b102e21e870335a7717768f3847256de8a6f7ee5ca'
    assert receipt['source']['upstream_base'] == '8cadfede7063c896b944e7bae05daa3549ae97ea'
    assert off_recipe.artifact_authority(path)[1] == receipt


def test_same_version_altered_wheel_is_not_authorized(tmp_path):
    path, _ = off_recipe.artifact_authority()
    substitute = tmp_path/path.name
    substitute.write_bytes(path.read_bytes()+b'new build metadata')
    with pytest.raises(ValueError, match='wheel identity drift'):
        off_recipe.artifact_authority(substitute)


def test_artifact_receipt_cannot_override_resource_pin(tmp_path, monkeypatch):
    pin = json.loads((off_recipe.ROOT/'ds41f_mlx/runtime/admitted_resources.json').read_text())
    pin['modules']['deepseek_recipe._native']['sha256'] = '0'*64
    p = tmp_path/'ds41f_mlx/runtime/admitted_resources.json'
    p.parent.mkdir(parents=True); p.write_text(json.dumps(pin))
    monkeypatch.setattr(off_recipe, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='qualified resource pin'):
        off_recipe.artifact_authority()


def test_pending_acceptance_is_not_a_production_default(tmp_path, monkeypatch):
    a = json.loads((glm_profile.SOURCES/'execution-profiles.json').read_text())
    a['status']['standard-off']['native'] = 'temporarily_deferred_pending_acceptance'
    (tmp_path/'execution-profiles.json').write_text(json.dumps(a))
    monkeypatch.setattr(glm_profile, 'SOURCES', tmp_path)
    with pytest.raises(ValueError, match='production GLM default'):
        glm_profile.default_profile('standard-off')
    assert glm_profile.default_profile('mtp') == 'native'
