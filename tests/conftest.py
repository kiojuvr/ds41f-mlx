"""Retain lifecycle-only TinyBlock doubles below the newly owned boundary.

Real state-producer tests use actual loaded Block/Attention/Compressor/Indexer
modules and do not take this branch. There is no production fallback.
"""
import pytest


@pytest.fixture(scope='session')
def installed_test_variant():
    """Authenticate the disposable installed dependency closure, without model I/O.

    Source-tree numerical unit tests then use that package's immutable selection;
    this is not admission of a checkpoint or a seal of mutable Python sources.
    """
    import json
    from pathlib import Path
    import subprocess
    import sys
    import sysconfig
    import ds41f_mlx
    from ds41f_mlx.mtp_identity import checkpoint_variant
    assert Path(ds41f_mlx.__file__).resolve().parent == Path(__file__).resolve().parents[1]/'ds41f_mlx'
    subprocess.run([sys.executable, '-I', '-c',
                    'from ds41f_mlx.execution_identity import inspect_payload; inspect_payload()'],
                   check=True)
    manifest = Path(sysconfig.get_paths()['purelib']) / 'ds41f_mlx/execution_manifest.json'
    return checkpoint_variant(json.loads(manifest.read_text()))


@pytest.fixture(autouse=True)
def preserve_process_environment():
    # App construction publishes configuration; one test must not configure the next.
    import os
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


@pytest.fixture(autouse=True)
def admitted_test_numerics(monkeypatch, installed_test_variant):
    from ds41f_mlx import execution_identity
    monkeypatch.setattr(execution_identity, '_admitted_variant', installed_test_variant)


@pytest.fixture(autouse=True)
def tiny_lifecycle_block_double(monkeypatch):
    from ds41f_mlx.runtime.state_production import DecodeStateProducer
    from ds41f_mlx.runtime import target_forward, dwarfstar_prefill
    import importlib
    from types import SimpleNamespace
    admitted = target_forward.resources_for
    def resources(model):
        if (getattr(getattr(model, '_config', None), '_state_production_test_double', False)
                or type(model).__module__ == 'test_m7_production_selector'):
            from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
            return SimpleNamespace(math=importlib.import_module('omlx.patches.deepseek_v41.language'),
                                   cache_type=DeepseekV41Cache, assert_active=lambda: None,
                                   validate_binding=lambda model: None, require_backend=lambda mx: None)
        return admitted(model)
    monkeypatch.setattr(target_forward, 'resources_for', resources)
    monkeypatch.setattr(dwarfstar_prefill, 'resources_for', resources)
    original = DecodeStateProducer.block

    def block(self, module, h, pre, cache, shared, start):
        if (type(module).__name__ == 'TinyBlock'
                and type(module).__module__ in ('test_m44_target_generation', 'test_target_generation')):
            return module(h, pre, cache, shared, start, None)
        return original(self, module, h, pre, cache, shared, start)

    monkeypatch.setattr(DecodeStateProducer, 'block', block)
    completion = DecodeStateProducer.validate_completion
    def validate(cache, config, frontier):
        if getattr(config, '_state_production_test_double', False):
            return  # Lifecycle doubles intentionally produce no attention state.
        return completion(cache, config, frontier)
    monkeypatch.setattr(DecodeStateProducer, 'validate_completion', staticmethod(validate))
