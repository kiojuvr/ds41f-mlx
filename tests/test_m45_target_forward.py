"""All-layer transaction failures use real MLX and the retained cache classes."""
import pytest
try:
    from test_m44_target_generation import mx, session
except ModuleNotFoundError as exc:
    if exc.name != 'test_m44_target_generation':
        raise
    from test_target_generation import mx, session  # explicit future delivery name
from ds41f_mlx.runtime.target_generation import TargetGenerationSession


@pytest.mark.parametrize('layer', [0, 1, 14, 20, 39])
def test_partial_layer_failure_burns_entire_lease(layer):
    model, gen = session()
    gen.start(3)
    cache = gen._cache
    frontier = gen.token_frontier
    model.fail, model.fail_layer = True, layer
    try:
        with pytest.raises(RuntimeError, match='injected'):
            gen.next_token()
        assert gen.token_frontier == frontier
        assert gen.current_token_history() == [0, 1, 3]
        assert all(c._p6_append_invalid and c._p6_append_failed for c in cache)
        # Earlier layers really mutated; nothing publishes this as continuation.
        assert tuple(c.size() for c in cache) == (frontier+1,)*layer + (frontier,)*(40-layer)
        with pytest.raises(RuntimeError, match='failed generation'):
            gen.extract_final_state()
        with pytest.raises(RuntimeError, match='not sealed/admissible'):
            TargetGenerationSession.from_prefilled_cache(model, cache, gen.current_token_history())
    finally:
        gen.close()


@pytest.mark.parametrize('boundary', ['sampler', 'synchronize', 'frontier'])
def test_failure_after_all_layer_mutation_does_not_publish(boundary, monkeypatch):
    _, gen = session()
    gen.start(3)
    cache = gen._cache
    history = gen.current_token_history()
    if boundary == 'sampler':
        def fail(p):
            raise RuntimeError('injected sampler')
        gen.sampler = fail
    elif boundary == 'synchronize':
        original = mx.synchronize
        def fail(stream):
            original(stream)
            raise RuntimeError('injected synchronize')
        monkeypatch.setattr(mx, 'synchronize', fail)
    else:
        original = gen.target_forward.forward
        def fail(*args):
            logits = original(*args)
            cache[39][0] = mx.array([900], mx.int32)
            return logits
        monkeypatch.setattr(gen.target_forward, 'forward', fail)
    try:
        with pytest.raises(RuntimeError):
            gen.next_token()
        assert gen.current_token_history() == history
        assert all(c._p6_append_invalid for c in cache)
        with pytest.raises(RuntimeError, match='failed generation'):
            gen.extract_final_state()
    finally:
        monkeypatch.undo()
        gen.close()


def test_preflight_failure_before_mutation_burns_lease():
    _, gen = session()
    gen.initial_cache[39].compress_ratio = 8
    cache = gen.initial_cache
    try:
        with pytest.raises(ValueError, match='layout mismatch'):
            gen.start(3)
        assert all(c.size() == 2 for c in cache)
        assert all(c._p6_append_invalid for c in cache)
        assert gen.current_token_history() == [0, 1]
    finally:
        gen.close()


def test_exact_objects_cancel_commit_and_resume():
    model, gen = session()
    cache = gen.initial_cache
    objects = tuple(map(id, cache))
    gen.start(3)
    gen.next_token()
    final, history = gen.extract_final_state('cancelled')
    assert final is cache and tuple(map(id, final)) == objects
    assert all(c.size() == len(history) == 4 for c in final)
    assert not any(getattr(c, '_p6_append_pending', False) for c in final)
    gen.close()
    resumed = TargetGenerationSession.from_prefilled_cache(model, final, history, max_tokens=1)
    try:
        resumed.start(3)
        resumed.next_token()
        final2, history2 = resumed.extract_final_state()
        assert final2 is cache and tuple(map(id, final2)) == objects
        assert all(c.size() == len(history2) == 6 for c in final2)
    finally:
        resumed.close()


def test_engram_prefetch_failure_retires_scope_and_burns_lease():
    from contextlib import contextmanager
    model, gen = session()
    model._config.engram_layer_ids = (14,)
    original_hash = model._hasher
    def hashes(ids, history, mask):
        _, new_history = original_hash(ids, history, mask)
        return mx.zeros((1, 1, 1), mx.int64), new_history
    model._hasher = hashes
    events = []
    class Prefetch:
        @contextmanager
        def forward(self):
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')
        def submit(self, embed, ids):
            events.append('submit')
    import mlx.nn as nn
    class Engram(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(4, 4)
        def __call__(self, h, ids, mask):
            raise RuntimeError('injected SSD Engram')
    model._engram_prefetch = Prefetch()
    model.layers[14].engram = Engram()
    cache = gen.initial_cache
    try:
        with pytest.raises(RuntimeError, match='SSD Engram'):
            gen.start(3)
        assert events == ['enter', 'submit', 'exit']
        assert tuple(c.size() for c in cache) == (3,)*14 + (2,)*26
        assert all(c._p6_append_invalid for c in cache)
        assert gen.current_token_history() == [0, 1]
    finally:
        gen.close()
