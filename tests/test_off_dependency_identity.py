"""OFF must not inherit MTP's different numerical dependency identity."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.build_execution_package import off_dependency_payloads

ROOT = Path(__file__).resolve().parents[1]


def authorities():
    return (json.loads((ROOT/'third_party/mtp/standard-off-dependencies.json').read_text()),
            json.loads((ROOT/'ds41f_mlx/runtime/admitted_resources.json').read_text()))


def test_off_payload_matches_existing_numerical_authority_not_mtp():
    off, pin = authorities()
    payloads = off_dependency_payloads(off, pin)
    mtp = json.loads((ROOT/'third_party/mtp/normal-local.json').read_text())
    assert payloads['numpy']['__init__.py'] == pin['modules']['numpy']['sha256']
    assert payloads['numpy']['__init__.py'] != mtp['dependency_payload']['numpy/__init__.py']
    assert off['versions'] == {'numpy': '2.5.3', 'transformers': '5.18.0'}


@pytest.mark.parametrize('kind', ['version', 'module', 'native'])
def test_off_authority_drift_fails_closed(kind):
    off, pin = authorities()
    off = deepcopy(off)
    if kind == 'version':
        off['versions']['numpy'] = '2.3.5'
    elif kind == 'module':
        off['payloads']['numpy']['__init__.py'] = '0'*64
    else:
        path = next(iter(pin['package_resources']['numpy']))
        off['payloads']['numpy'][path] = '0'*64
    with pytest.raises(ValueError, match='OFF dependency authority'):
        off_dependency_payloads(off, pin)
