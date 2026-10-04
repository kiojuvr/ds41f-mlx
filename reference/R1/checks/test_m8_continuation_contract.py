import pytest

from ds41f_mlx.runtime.continuation_session import M8ContinuationError, M8LiveContinuationSession
from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession


def test_m8_rejects_non_extension_before_cache_mutation():
    sess = M8LiveContinuationSession(model=object(), live_cache=[object()], token_history=[10, 20, 30])
    before_cache = sess.live_cache
    before_history = list(sess.token_history)

    with pytest.raises(M8ContinuationError, match="exact extension"):
        sess.begin_turn_from_recipe_tokens([10, 99, 30, 40])

    assert sess.live_cache is before_cache
    assert sess.token_history == before_history
    assert sess.state == "idle"
    assert sess.turn_records == []


def test_m8_rejects_empty_next_suffix_before_cache_mutation():
    sess = M8LiveContinuationSession(model=object(), live_cache=[object()], token_history=[1, 2, 3])

    with pytest.raises(M8ContinuationError, match="no new terminal"):
        sess.begin_turn_from_recipe_tokens([1, 2, 3])

    assert sess.token_history == [1, 2, 3]
    assert sess.turn_records == []


def test_generation_current_token_history_includes_terminal_once():
    gen = OMLXGenerationSession.__new__(OMLXGenerationSession)
    gen.prefix_tokens = [7, 8]
    gen.first_input_token = 9
    gen.token_frontier = 5
    gen.admitted_frontier = 2
    gen.generated_tokens = [10, 11]

    assert gen.current_token_history() == [7, 8, 9, 10, 11]


def test_m8_diagnostics_are_bounded_to_recent_turns():
    sess = M8LiveContinuationSession(model=object(), live_cache=[], token_history=[1])
    for i in range(20):
        # record shape is exercised indirectly through diagnostics without
        # invoking model/cache code.
        from ds41f_mlx.runtime.continuation_session import M8TurnRecord
        sess.turn_records.append(M8TurnRecord(i, i, 1, 0, 100 + i, (), i + 1, 0, 0, 0.0, 0.0))

    diag = sess.diagnostics()
    assert diag["turn_count"] == 20
    assert len(diag["turns"]) == 16
    assert diag["turns"][0]["turn_index"] == 4
