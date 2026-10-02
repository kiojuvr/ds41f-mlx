"""M20 dependency pin and path-independent native identity regressions."""
from pathlib import Path
import tempfile

from ds41f_mlx.config import DEFAULT_OMLX
from ds41f_mlx.provenance import PINNED_OMLX_REVISION, omlx_decode_native_identity
from ds41f_mlx.qualify import compare_identity_projection, identity_projection


def test_promoted_default_and_revision():
    assert DEFAULT_OMLX == Path.home() / 'omlx-0.7.0.release'
    assert PINNED_OMLX_REVISION == '4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40'


def test_native_identity_ignores_checkout_name_but_detects_binary_change():
    with tempfile.TemporaryDirectory() as tmp:
        roots = [Path(tmp) / name for name in ('versioned', 'operational')]
        for root in roots:
            native = root / 'omlx/custom_kernels/glm_moe_dsa'
            native.mkdir(parents=True)
            (native / '_ext.cpython-313-darwin.so').write_bytes(b'qualified')
        before = omlx_decode_native_identity(roots[0])
        assert before == omlx_decode_native_identity(roots[1])
        (roots[1] / 'omlx/custom_kernels/glm_moe_dsa/_ext.cpython-313-darwin.so').write_bytes(b'changed')
        assert before != omlx_decode_native_identity(roots[1])


def test_existing_artifact_check_detects_dependency_and_native_drift():
    current = {'omlx': {'revision': PINNED_OMLX_REVISION, 'local_identity_sha256': 'clean',
                        'decode_native_identity': {'sha256': 'qualified'}},
               'packages': {'mlx': '0.32.2', 'mlx-lm': '0.31.4.dev132+g94cdcae13'}}
    expected = identity_projection(current)
    artifact = Path('promotion.json')
    assert compare_identity_projection(expected, current, artifact=artifact)['status'] == 'CURRENT_RUNTIME_MATCHES_ARTIFACT'
    for altered in (
        {**current, 'omlx': {**current['omlx'], 'revision': '0.7.1'}},
        {**current, 'omlx': {**current['omlx'], 'decode_native_identity': {'sha256': 'changed'}}},
        {**current, 'packages': {'mlx': 'new-version'}},
    ):
        assert compare_identity_projection(expected, altered, artifact=artifact)['status'] == 'EXPENSIVE_QUALIFICATION_STALE'
