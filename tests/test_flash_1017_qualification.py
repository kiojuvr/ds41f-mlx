"""Transferred Flash-1017 primitive/admission regressions; no model weights."""
import json
from pathlib import Path

import pytest


def test_source_emulation_ties_scale_floor_and_chunking():
    np = pytest.importorskip('numpy')
    from tools.probe_flash_1017_pre_woa import bf16, e4m3, reference
    np.testing.assert_array_equal(e4m3(np.array([1.0625, 1.1875, -1.0625, 500])),
                                  [1, 1.25, -1, 448])
    x = np.full((1, 3, 8, 64), 2**-40, np.float32)
    y, scales = reference(x)
    assert np.all(y == 0)
    assert np.all(scales == 2**-22)
    joined = np.concatenate([reference(x[:, i:i+1])[0] for i in range(3)], 1)
    np.testing.assert_array_equal(y, joined)
    # Input BF16 cast precedes FP8 rounding (midpoint perturbation erased).
    assert bf16(np.array([1.0625 + 2**-12]))[0] == 1.0625


def test_selected_inventory_replaces_original_without_changing_layout():
    from ds41f_mlx.mtp_identity import CHECKPOINT_REVISION, flash_1017_inventory
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root/'third_party/mtp/normal-local.json').read_text())['checkpoint']
    actual = flash_1017_inventory(original)
    assert len(actual['shards']) == 48
    assert CHECKPOINT_REVISION == '1bae0e1088993b9b82de08f59de5be5a34e42170'
    assert all(s['source_revision'] == CHECKPOINT_REVISION for s in actual['shards'])
    assert [(s['name'], s['bytes']) for s in actual['shards']] == [
        (s['name'], s['bytes']) for s in original['shards']]
    assert all(a['lfs_sha256'] != b['lfs_sha256'] for a,b in zip(actual['shards'], original['shards']))
    assert flash_1017_inventory(actual) == actual
    with pytest.raises(ValueError, match='48-shard'):
        flash_1017_inventory(dict(original, shards=[]))


def test_pinned_target_and_dspark_use_the_shared_boundary(tmp_path):
    """Inspect development source in scratch, not installed admission or sealing."""
    import subprocess
    import tarfile
    root = Path(__file__).resolve().parents[1]
    with tarfile.open(root/'third_party/mtp/omlx-source.tar.gz') as archive:
        archive.extractall(tmp_path, filter='data')
    for label in ('compressed-growth', 'bit-exact-kernels', 'template-reuse', 'context-copy'):
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i',
                        str(root/f'third_party/mtp/{label}.patch')], cwd=tmp_path, check=True,
                       capture_output=True)
    for name in ('language', 'dspark'):
        source = (tmp_path/f'omlx/patches/deepseek_v41/{name}.py').read_text()
        boundary = source.index('grouped = pre_woa(out.reshape(')
        assert source.rindex('inverse=True)', 0, boundary) < boundary < source.index('mx.einsum(', boundary)
        assert 'from ds41f_mlx.model_execution.activation import pre_woa' in source
    for path in ('model_execution/language.py', 'runtime/state_production.py',
                 'prefill_fp8_mlx/omlx_suffix_math.py'):
        source = (root/'ds41f_mlx'/path).read_text()
        boundary = source.index('grouped = pre_woa(out.reshape(')
        assert source.rindex('inverse=True)', 0, boundary) < boundary < source.index('mx.einsum(', boundary)
    # No checkpoint-toggle/configuration shim is needed for the selected baseline.
    assert 'pre_woa_fp8' not in (tmp_path/'omlx/patches/deepseek_v41/config.py').read_text()


def test_checkpoint_byte_cache_rejects_replacement(tmp_path, monkeypatch):
    import hashlib
    from ds41f_mlx import mtp_identity as identity
    monkeypatch.setattr(identity, 'CHECKPOINT_CACHE', tmp_path/'cache.json')
    shard = tmp_path/'weights'
    shard.write_bytes(b'expected')
    assets = {'shards': [dict(name='weights', lfs_sha256=hashlib.sha256(b'expected').hexdigest())]}
    identity.verify_checkpoint_bytes(tmp_path, assets)
    identity.verify_checkpoint_bytes(tmp_path, assets)
    shard.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='checkpoint bytes'):
        identity.verify_checkpoint_bytes(tmp_path, assets)


@pytest.mark.parametrize('variant', ['flash', 'flash-1017'])
@pytest.mark.parametrize('receipts', [False, True])
def test_checkpoint_selection_requires_exact_assets(tmp_path, monkeypatch, receipts, variant):
    import hashlib
    from dataclasses import replace
    from ds41f_mlx import mtp_identity as identity
    digest = hashlib.sha256(b'{}').hexdigest()
    monkeypatch.setattr(identity, 'CHECKPOINT', {'config.json': digest})
    (tmp_path/'config.json').write_bytes(b'{}')
    selected = identity.VARIANTS[variant]
    expected = dict(metadata=identity.CHECKPOINT, shards=[
        dict(name=f'model-{i:05d}.safetensors', bytes=1, source_revision=selected.revision,
             lfs_sha256=hashlib.sha256(b'x').hexdigest()) for i in range(48)],
        verification='metadata/source inventory only; weights not freshly rehashed')
    # Tiny asset fixture; production fingerprints are tested separately below.
    selected = replace(selected, inventory_sha256=hashlib.sha256(
        json.dumps(expected, sort_keys=True).encode()).hexdigest())
    monkeypatch.setitem(identity.VARIANTS, variant, selected)
    metadata = {'tokenizer_config.json': '6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547'}
    (tmp_path/'tokenizer_config.json').write_bytes(b'{}')
    authority = dict(checkpoint=expected, checkpoint_metadata=metadata,
                     checkpoint_variant=variant, numerics=selected.numerics)
    real_sha = identity.sha
    monkeypatch.setattr(identity, 'sha', lambda path:
        metadata['tokenizer_config.json'] if path.name == 'tokenizer_config.json' else real_sha(path))
    for s in expected['shards']:
        (tmp_path/s['name']).write_bytes(b'x')
        if receipts:
            p = tmp_path/'.cache/huggingface/download'/(s['name']+'.metadata')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s['source_revision']+'\n'+s['lfs_sha256']+'\n')
    verified = []
    monkeypatch.setattr(identity, 'verify_checkpoint_bytes', lambda root, assets: verified.append(assets))
    assert identity.admit_checkpoint(tmp_path, authority) == expected
    assert verified == [expected]
    (tmp_path/'foreign.json').write_text('{}')
    with pytest.raises(ValueError, match='configuration files'):
        identity.admit_checkpoint(tmp_path, authority)
    (tmp_path/'foreign.json').unlink()
    (tmp_path/'configuration.json').write_text('{}')
    with pytest.raises(ValueError, match='checkpoint metadata|configuration files'):
        identity.admit_checkpoint(tmp_path, authority)
    (tmp_path/'configuration.json').unlink()
    with pytest.raises(ValueError, match='not admitted'):
        identity.admit_checkpoint(tmp_path, dict(authority, checkpoint=dict(expected, shards=[])))
    receipt = tmp_path/'.cache/huggingface/download'/(expected['shards'][0]['name']+'.metadata')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text('foreign-revision\n'+'0'*64+'\n')
    with pytest.raises(ValueError, match='source revision'):
        identity.admit_checkpoint(tmp_path, authority)
    receipt.unlink()
    (tmp_path/expected['shards'][0]['name']).write_bytes(b'xx')
    with pytest.raises(ValueError, match='checkpoint'):
        identity.admit_checkpoint(tmp_path, authority)


def test_pre_woa_metal_boundary_is_distinct_from_generic_quantization():
    import sys
    if sys.platform != 'darwin':
        pytest.skip('Metal requires macOS')
    import numpy as np
    import mlx.core as mx
    from ds41f_mlx.model_execution.activation import quantize_pre_woa, quantize_fp8_activation
    from tools.probe_flash_1017_pre_woa import bf16, reference
    rng = np.random.default_rng(1017)
    x = bf16(rng.normal(size=(1, 3, 8, 64)).astype(np.float32))
    x.reshape(-1,32)[0] = 2**-40
    x.reshape(-1,32)[1] = np.copysign(np.zeros(32, np.float32), np.arange(32) % 2 - .5)
    a = mx.array(x).astype(mx.bfloat16)
    legacy = np.asarray(quantize_fp8_activation(a).astype(mx.float32))
    assert np.all(legacy.reshape(-1,32)[0] == 2**-40)
    actual = quantize_pre_woa(a)
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)).view(np.uint32),
                                  reference(x)[0].view(np.uint32))
    chunked = mx.concatenate([quantize_pre_woa(a[:, i:i+1]) for i in range(3)], axis=1)
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), np.asarray(chunked.astype(mx.float32)))
    with pytest.raises(ValueError, match='BF16'):
        quantize_pre_woa(a.astype(mx.float32))


@pytest.mark.parametrize('name', ['flash', 'flash-1017'])
def test_approved_manifest_tuple_rejects_mixed_or_foreign_inventory(name):
    from copy import deepcopy
    from ds41f_mlx.mtp_identity import VARIANTS, checkpoint_variant, flash_1017_inventory
    old = json.loads((Path(__file__).resolve().parents[1]/'third_party/mtp/normal-local.json').read_text())
    assets = old['checkpoint'] if name == 'flash' else flash_1017_inventory(old['checkpoint'])
    authority = dict(checkpoint_variant=name, numerics=VARIANTS[name].numerics,
                     checkpoint=assets, checkpoint_metadata=old['checkpoint_metadata'])
    assert checkpoint_variant(authority) is VARIANTS[name]
    for field, value in [('checkpoint_variant', 'unknown'), ('numerics', 'foreign'),
                         ('checkpoint_variant', 'flash-1017' if name == 'flash' else 'flash')]:
        with pytest.raises(ValueError):
            checkpoint_variant(dict(authority, **{field: value}))
    for key in ('lfs_sha256', 'source_revision', 'name', 'bytes'):
        foreign = deepcopy(authority)
        foreign['checkpoint']['shards'][0][key] = 0 if key == 'bytes' else 'foreign'
        with pytest.raises(ValueError, match='not admitted'):
            checkpoint_variant(foreign)
    with pytest.raises(ValueError):
        checkpoint_variant(dict(authority, checkpoint_metadata={}))


def test_shared_boundary_is_fail_closed_immutable_and_io_free(monkeypatch):
    from dataclasses import FrozenInstanceError
    from ds41f_mlx import execution_identity as identity
    from ds41f_mlx.mtp_identity import VARIANTS
    from ds41f_mlx.model_execution import activation
    monkeypatch.setattr(identity, '_admitted_variant', None)
    with pytest.raises(RuntimeError, match='admission'):
        activation.pre_woa(object())
    identity._bind_variant(VARIANTS['flash'])
    with pytest.raises(FrozenInstanceError):
        identity._admitted_variant.numerics = 'foreign'
    with pytest.raises(ValueError, match='changed'):
        identity._bind_variant(VARIANTS['flash-1017'])
    monkeypatch.setattr(Path, 'read_text', lambda *a, **k: pytest.fail('hot-path I/O'))
    x = object()
    monkeypatch.setattr(activation, 'quantize_pre_woa', lambda x: pytest.fail('original quantization'))
    assert activation.pre_woa(x) is x
    monkeypatch.setattr(identity, '_admitted_variant', VARIANTS['flash-1017'])
    calls = []
    result = object()
    monkeypatch.setattr(activation, 'quantize_pre_woa', lambda value: calls.append(value) or result)
    assert activation.pre_woa(x) is result
    assert calls == [x]
