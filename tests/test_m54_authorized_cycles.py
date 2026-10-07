"""M54 primitives on reduced real MLX; proposal arithmetic is M53's stub."""
from types import SimpleNamespace

import pytest
from test_m53_proposal_producer import producer_session
from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession, M8TurnRecord
from ds41f_mlx.runtime.semantic_cycle import SemanticCycleAdapter


class Processor:
    def __init__(self, terminal=14):
        self.tokens = []
        self.terminal = terminal
        self.semantic_terminal = None
        self.finished = False
    def semantic_snapshot(self):
        return tuple(self.tokens)
    def preview_tokens(self, ids):
        at = next((i for i, token in enumerate(ids) if token == self.terminal), None)
        return SimpleNamespace(mapping_exact=True, safe_token_count=len(ids) if at is None else at,
            completing_token_index=at, terminal_kind=None if at is None else 'STOP_SEQUENCE',
            source_start=0, source_end=1)
    def observe(self, report):
        assert self.semantic_terminal is None
        self.tokens.append(report.token)
        if report.token == self.terminal:
            self.semantic_terminal = SimpleNamespace(kind='STOP_SEQUENCE', source_start=0, source_end=1)


def setup(monkeypatch, terminal=14):
    gen, child, producer, cache = producer_session(monkeypatch)
    m8 = M8LiveContinuationSession(gen.model, [], gen.current_token_history(), config=gen.config, generation=gen)
    m8.turn_records.append(M8TurnRecord(0, 5, 1, 0, 10, (), gen.token_frontier, 0, 0, 0., 0.))
    processor = Processor(terminal)
    adapter = SemanticCycleAdapter(m8, processor, producer=producer)
    return gen, child, producer, cache, m8, processor, adapter


@pytest.mark.parametrize('terminal', [12, 13, 14, 15, 16])
def test_high_acceptance_tail_cannot_cross_recipe_permission(monkeypatch, terminal):
    gen, child, producer, exact, m8, p, adapter = setup(monkeypatch, terminal)
    try:
        while p.semantic_terminal is None:
            adapter.advance(p.observe)
            assert m8.token_history == gen.current_token_history()
            assert set(gen.active_cache_offsets()) == {m8.frontier}
            assert all(token <= terminal for token in gen.generated_tokens)
        assert p.tokens[-1] == terminal
        assert gen.generated_tokens[-1] == terminal
        assert producer.frontier == gen.token_frontier
        with pytest.raises(RuntimeError, match='already terminal'):
            adapter.advance(p.observe)
        m8.ensure_idle('recipe_stop')
        assert m8.live_cache is exact and m8.token_history[-1] == terminal
        assert not child.producers and not child.receipts
    finally:
        if m8.generation is not None:
            gen.close()
        m8.close()


@pytest.mark.parametrize('phase', ['settled', 'adopted', 'observed', 'reported'])
def test_postmutation_report_failure_burns_and_cannot_reenter(monkeypatch, phase):
    gen, child, producer, cache, m8, p, adapter = setup(monkeypatch, terminal=99)
    def fault(where):
        if where == phase:
            raise RuntimeError('report fault')
    try:
        with pytest.raises(RuntimeError, match='report fault'):
            adapter.advance(p.observe, _fault=fault)
        assert m8.closed and gen._failed and not producer.active
        assert all(c._p6_append_invalid for c in cache)
        with pytest.raises(RuntimeError):
            gen.extract_final_state()
        with pytest.raises(RuntimeError):
            adapter.advance(p.observe)
    finally:
        gen.close()
    assert not child.receipts and not child.producers


def test_cancel_after_authorization_before_mutation(monkeypatch):
    gen, child, producer, cache, m8, p, adapter = setup(monkeypatch)
    before = gen.current_token_history()
    cancelled = [False]
    try:
        assert adapter.advance(p.observe, cancelled=lambda: cancelled[0],
            _fault=lambda phase: cancelled.__setitem__(0, True)) == []
        assert gen.current_token_history() == m8.token_history == before
        assert producer.frontier == gen.token_frontier and not p.tokens
    finally:
        gen.close()


def test_duplicate_or_incomplete_batch_rejected(monkeypatch):
    gen, _, _, _, m8, p, adapter = setup(monkeypatch, terminal=99)
    try:
        reports = adapter.advance(p.observe)
        with pytest.raises(RuntimeError, match='frontier mismatch'):
            m8.adopt_cycle_reports(reports)
    finally:
        gen.close()


def test_tool_eof_preview_fails_before_numerical_entry(monkeypatch):
    gen, _, producer, _, m8, p, _ = setup(monkeypatch)
    try:
        before = gen.current_token_history()
        with pytest.raises(RuntimeError, match='EOF-sensitive'):
            SemanticCycleAdapter(m8, p, parse_tool_calls=True, producer=producer)
        assert gen.current_token_history() == before
    finally:
        gen.close()
