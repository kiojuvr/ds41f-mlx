"""Evaluation-only contracts for the retained #4363 suffix proposer."""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'artifacts/context-copy/upstream-context-copy.py'
spec = importlib.util.spec_from_file_location('context_copy_evaluation', SOURCE)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def proposer(history):
    c = cc.ContextCopy()
    c.extend(history, [])
    return c


def test_minimum_match_and_full_dspark_window():
    span = list(range(100, 112))
    continuation = list(range(200, 205))
    assert proposer(span + continuation + [999] + span).propose(5) == continuation
    assert proposer(span + continuation + [999] + span[1:]).propose(5) == []
    copied = proposer(span + continuation[:4] + span).propose(5)
    # A full window is possible even when it overlaps the later known context;
    # no token beyond the frontier is fabricated.
    assert copied == continuation[:4] + span[:1]


def test_frontier_clips_source_and_room():
    span = list(range(100, 112))
    history = span + [999] + span
    c = proposer(history)
    assert c.propose(1) == []
    assert c.propose(5) == [999] + span[:4]
    # The tail itself must never supply an uncommitted continuation.
    assert proposer(span).propose(5) == []


def test_longest_match_then_most_recent_tie():
    span = list(range(100, 120))
    history = span + [300, 301, 302] + span + [400, 401, 402] + span
    assert proposer(history).propose(3) == [400, 401, 402]
    shorter = [999] + span[-12:]
    history = span + [300, 301, 302] + shorter + [400, 401, 402] + span
    assert proposer(history).propose(3) == [300, 301, 302]


def test_misses_pause_but_never_commit():
    span = list(range(100, 112))
    c = proposer(span + [300, 301, 302] + span)
    before = list(c._ids)
    for _ in range(4):
        c.observe(0, 3)
    for _ in range(8):
        assert c.propose(3) == []
    assert c.propose(3) == [300, 301, 302]
    c.observe(3, 3)
    assert c._ids == before


def test_upstream_source_can_include_unconsumed_pending_anchor():
    # Native settlement discards the final queue token on cancellation when
    # target is ahead of delivered history. Upstream extends the lookup with
    # that token too: known-to-the-proposer is not consumed/canonical context.
    consumed = [100, 101, 102] * 4 + [100, 101]
    c = proposer(consumed)
    c.extend(consumed, [102])
    assert c.propose(3) == [100, 101, 102]
    # Longest eligible earlier suffix ends at 11 (12-token match); selected
    # continuation is positions 12..14. Position 14 is the pending anchor.
    assert c._ids[12:15] == c.propose(3)
    assert 15 > len(consumed)


def test_copied_cycle_skip_changes_acceptance_shape_policy():
    from omlx.patches.deepseek_v41.mtp import AcceptanceDepthController
    ordinary = AcceptanceDepthController(3)
    copied = AcceptanceDepthController(3)
    ordinary.observe(3, 0, 0.)
    # #4363 deliberately omits observe for copies. Here the next native
    # verification widths differ; this is a policy fact, NOT proof of an
    # observed token divergence in the real workloads.
    assert ordinary.cur == 1
    assert copied.cur == 3


def test_appended_committed_history_is_indexed():
    span = list(range(100, 112))
    c = proposer(span + [300, 301, 302])
    history = list(c._ids)
    assert not c.extend(history, span)
    assert c.propose(3) == [300, 301, 302]
    assert c._ids == history + span
