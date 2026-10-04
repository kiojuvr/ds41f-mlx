"""Retain lifecycle-only TinyBlock doubles below the newly owned boundary.

Real state-producer tests use actual loaded Block/Attention/Compressor/Indexer
modules and do not take this branch. There is no production fallback.
"""
import pytest


@pytest.fixture(autouse=True)
def tiny_lifecycle_block_double(monkeypatch):
    from ds41f_mlx.runtime.state_production import DecodeStateProducer
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
