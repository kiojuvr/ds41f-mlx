"""Small boundary tests; real installed/native/model acceptance is separate."""
import json
from pathlib import Path
import pytest
from ds41f_mlx.execution_identity import inventory, verify_tree, links, host_closure


def test_payload_add_change_remove_are_rejected(tmp_path):
    file = tmp_path/'execution.py'
    file.write_text('value = 1\n')
    expected = inventory(tmp_path)
    verify_tree(tmp_path, expected, 'fixture')
    for action in ('change', 'add', 'remove'):
        file.write_text('value = 1\n')
        extra = tmp_path/'extra.py'
        extra.unlink(missing_ok=True)
        if action == 'change': file.write_text('value = 2\n')
        if action == 'add': extra.write_text('value = 1\n')
        if action == 'remove': file.unlink()
        with pytest.raises(ValueError, match='payload drift'):
            verify_tree(tmp_path, expected, 'fixture')


def test_native_semantic_graph_is_not_observed_load_closure():
    manifest = json.loads(Path('ds41f_mlx/execution_manifest.json').read_text())
    names = set(manifest['semantic_native'])
    assert any('opencv_imgproc' in p for p in names)
    assert any('opencv_imgcodecs' in p for p in names)
    assert any('webp.7' in p for p in names)
    assert not any(any(v in p for v in ('harfbuzz', 'glib', 'pcre', 'freetype', 'avcodec')) for p in names)
    assert 'runtime' not in manifest and 'native_links' not in manifest


def test_host_abi_regression_fails_without_hash_authority(tmp_path, monkeypatch):
    import ds41f_mlx.execution_identity as identity
    consumer, provider = tmp_path/'consumer', tmp_path/'provider'
    consumer.touch(); provider.touch()
    monkeypatch.setattr(identity, 'links', lambda path: [(provider, '2.0.0')] if path == consumer else [])
    monkeypatch.setattr(identity.subprocess, 'check_output', lambda *a, **k:
                        'provider:\n provider (compatibility version 1.0.0, current version 10.0.0)\n')
    with pytest.raises(ValueError, match='incompatible native ABI'):
        host_closure(consumer)
    monkeypatch.setattr(identity.subprocess, 'check_output', lambda *a, **k:
                        'provider:\n provider (compatibility version 2.0.0, current version 10.49.0)\n')
    assert host_closure(consumer) == sorted([str(consumer), str(provider)])
