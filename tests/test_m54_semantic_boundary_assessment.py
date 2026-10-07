"""M54 negative integration probes, NOT application/checkpoint qualification.

Real MLX and reduced M51 packed-state target from M52. Token 12 represents
an application-owned semantic boundary; no DSML grammar or producer math is
simulated as authoritative. These passing tests demonstrate why a naked cycle
or a report queue is insufficient to integrate the current runtime.
"""
from collections import deque

import pytest

from test_m52_generation import session
from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession, M8TurnRecord
from ds41f_mlx.serving.server import create_app


@pytest.mark.parametrize('boundary_kind', ['tool_call', 'recipe_stop', 'response_completion'])
def test_unrestricted_cycle_commits_past_application_boundary(boundary_kind):
    gen, target, exact = session()
    control, _, control_cache = session()
    try:
        # The ordinary runtime can stop after canonical observation of token 12.
        assert [control.next_token().token for _ in range(2)] == [11, 12]
        control.extract_final_state(boundary_kind)
        result = gen.speculative_cycle([12, 13, 14])
        assert [r.token for r in result['reports']] == [11, 12, 13, 14]
        cache, history = gen.extract_final_state(boundary_kind)
        assert cache is exact and control._final_cache is control_cache
        assert history[-4:] == [11, 12, 13, 14]
        assert gen.token_frontier == control.token_frontier + 2
        assert target.children[-1].phase == 'retired'
        # Stopping *after* the cycle cannot remove an already committed tail.
        assert set(gen.cache_offsets(cache)) == {len(history)}
    finally:
        gen.close()
        control.close()


def test_report_queue_hides_committed_tail_until_idle_reentry():
    gen, _, exact = session()
    m8 = M8LiveContinuationSession(gen.model, [], gen.current_token_history(),
                                  config=gen.config, generation=gen)
    m8.turn_records.append(M8TurnRecord(0, 5, 1, 0, 10, (), 6, 0, 0, 0., 0.))
    try:
        result = gen.speculative_cycle([12, 13, 14])
        queue = deque(result['reports'])
        # Counterexample to replacing next_token with buffered cycle reports.
        gen.next_token = queue.popleft
        assert [m8.next_token().token for _ in range(2)] == [11, 12]
        assert m8.token_history[-2:] == [11, 12]
        assert m8.frontier + 2 == gen.token_frontier
        assert set(gen.active_cache_offsets()) == {gen.token_frontier}
        m8.ensure_idle('tool_call')
        assert m8.live_cache is exact
        # Idle extraction adopts tokens never observed by the recipe/application.
        assert m8.token_history[-4:] == [11, 12, 13, 14]
        assert m8.turn_records[-1].generated_tokens == (11, 12)
    finally:
        if m8.generation is not None:
            gen.close()
        m8.close()


def test_predeclared_backend_stop_is_not_arbitrary_recipe_horizon():
    gen, _, exact = session(stop=(12,))
    try:
        result = gen.speculative_cycle([12, 13, 14])
        assert [r.token for r in result['reports']] == [11, 12]
        assert gen.stop_reason == 'stop'
        cache, history = gen.extract_final_state()
        assert cache is exact and history[-2:] == [11, 12]
        assert set(gen.cache_offsets(cache)) == {len(history)}
        # This control only proves the existing static EOS/stop-ID contract.
        # Multi-token stop strings / parsed tool boundaries are not static IDs.
    finally:
        gen.close()


def test_existing_native_recipe_preview_is_available_and_nonmutating():
    # Actual installed recipe + checkpoint tokenizer, no target/checkpoint math.
    # The missing primitive is its canonical runtime integration, NOT a parser.
    from ds41f_mlx.config import load_runtime_config
    from ds41f_mlx.serving.server import load_v41_tokenizer
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import deepseek_recipe as d
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    options = d.ParsingOptions()
    options.stop_sequences = ['HALT NOW']
    processor = d.StreamProcessor(
        d.ChatCompletionChunkGenerator('m54', cfg.model_id, True, False),
        options, tokenizer)
    try:
        ids = list(tokenizer.encode('Hello HALT NOW after'))
        before = processor.semantic_snapshot()
        preview = processor.preview_tokens(ids)
        assert processor.semantic_snapshot() == before
        assert preview.mapping_exact and preview.terminal_kind == 'STOP_SEQUENCE'
        assert preview.completing_token_index < len(ids) - 1
        for token in ids[:preview.completing_token_index + 1]:
            processor.push(d.InferenceChunk.token(token))
        assert processor.semantic_terminal.kind == 'STOP_SEQUENCE'
    finally:
        processor.close()


def test_unqualified_first_party_profile_rejected_before_backend_allocation():
    # No pretend operational selector and no silent request-time OFF fallback.
    with pytest.raises(ValueError, match='unknown capability profile'):
        create_app(profile='first-party-mtp-development')
