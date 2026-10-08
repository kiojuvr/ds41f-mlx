"""Actual recipe/tokenizer, consuming protocol projection vs historical M11 EOF.

No target doubles establish the semantic ordinal in these tests.
"""
import json
from types import SimpleNamespace
import pytest
from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
from ds41f_mlx.runtime.tool_boundary_session import (
    _make_processor_and_response, _record_protocol_output,
    _recent_tool_arguments_look_complete, _probe_tool_calls_complete,
)
from ds41f_mlx.runtime.tool_eof_preview import ToolEOFPreview

TEXT = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
        '</｜DSML｜ invoke>\n</｜DSML｜ calls>')


def fixture():
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import deepseek_recipe as d
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    body = json.dumps(dict(model=cfg.model_id,
        messages=[dict(role='user', content='Weather?')], max_tokens=64,
        reasoning_effort='none', tools=[dict(type='function', function=dict(
            name='lookup_weather', parameters=dict(type='object', properties=dict(city=dict(type='string')))))])).encode()
    prepared = prepare_request('chat_completions', body, tokenizer=tokenizer, recipe_path=cfg.recipe_path)
    p, response = _make_processor_and_response(d, prepared, 'm54-native-eof', cfg.model_id, tokenizer)
    turn = SimpleNamespace(prepared=prepared, processor=p, response=response, events=[])
    for out in p.push(d.InferenceChunk.ready()):
        _record_protocol_output(out, response, turn.events)
    return cfg, d, tokenizer, turn


@pytest.mark.parametrize('text', [TEXT, TEXT.replace('Paris', 'café🙂'),
    TEXT.replace('Paris', 'brace } in string'),
    '<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup_weather">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>',
    TEXT[:TEXT.index('Paris')], 'Ordinary café🙂 text.'])
def test_every_prefix_matches_consuming_m11(text):
    cfg, d, tokenizer, turn = fixture()
    ids = list(tokenizer.encode(text))
    oracle = ToolEOFPreview(turn)
    earliest = None
    try:
        for i, token in enumerate(ids):
            before = (turn.processor.preview_revision, turn.processor.semantic_snapshot(), turn.response.to_json(), list(turn.events))
            proof = oracle.preview([token], i)
            assert proof == oracle.preview([token], i)
            oracle.validate(proof, [token], i)
            assert before == (turn.processor.preview_revision, turn.processor.semantic_snapshot(), turn.response.to_json(), turn.events)
            for out in turn.processor.push(d.InferenceChunk.token(token)):
                _record_protocol_output(out, turn.response, turn.events)
            old = (_recent_tool_arguments_look_complete(turn.prepared.protocol, turn.events)
                   and _probe_tool_calls_complete(d, turn.prepared, 'm54-native-eof', cfg.model_id, tokenizer, ids[:i+1]))
            assert (proof.boundary == 'M11_TOOL_EOF') == old
            assert oracle.completed(i+1) == old
            with pytest.raises(RuntimeError, match='stale'):
                oracle.validate(proof, [token], i)
            if old:
                earliest = i
                break
        if text == TEXT:
            assert earliest == 31
    finally:
        turn.processor.close()


def test_native_authorization_clamps_proposal_tail_and_protects_eof_anchor():
    from ds41f_mlx.runtime.semantic_cycle import SemanticCycleAdapter
    _, d, tokenizer, turn = fixture()
    ids = list(tokenizer.encode(TEXT))
    try:
        for token in ids[:30]:
            for out in turn.processor.push(d.InferenceChunk.token(token)):
                _record_protocol_output(out, turn.response, turn.events)
        gen = SimpleNamespace(stop_token_ids=(), max_tokens=64, generated_tokens=[],
            token_frontier=30, current_token_history=lambda: [1],
            _pending=SimpleNamespace(item=lambda: ids[30]))
        m8 = SimpleNamespace(generation=gen, token_history=[1])
        offered = []
        def propose():
            offered.append(True)
            return tuple(ids[31:])  # producer may look beyond both boundaries
        producer = SimpleNamespace(active=True, child=SimpleNamespace(active=True,
            config=SimpleNamespace(window_size=4)), rings=[SimpleNamespace(keys=SimpleNamespace(shape=(1,1,4)))], propose=propose)
        oracle = ToolEOFPreview(turn)
        adapter = SemanticCycleAdapter(m8, turn.processor, parse_tool_calls=True, producer=producer, eof_preview=oracle)
        adapter.ordinal = 30
        permission = adapter.authorize()
        assert offered and permission.drafts == ()
        assert permission.eof_proofs[-1].observed_count == 2
        assert permission.eof_proofs[-1].boundary == 'M11_TOOL_EOF'
        for out in turn.processor.push(d.InferenceChunk.token(ids[30])):
            _record_protocol_output(out, turn.response, turn.events)
        gen._pending = SimpleNamespace(item=lambda: ids[31])
        gen.token_frontier += 1
        adapter.ordinal += 1
        offered.clear()
        permission = adapter.authorize()
        assert not offered and not permission.drafts
        assert permission.terminal_identity == ('M11_TOOL_EOF', 31)
        with pytest.raises(RuntimeError, match='stale'):
            oracle.validate(permission.eof_proofs[0], [ids[30]], 31)
        # Mutating only the protocol projection must also invalidate permission.
        turn.events.append({'other':'projection'})
        with pytest.raises(RuntimeError, match='stale'):
            oracle.validate(permission.eof_proofs[0], [ids[31]], 31)
    finally:
        turn.processor.close()


def test_native_projection_matches_exact_consuming_json_and_sse_at_eof():
    _, d, tokenizer, turn = fixture()
    p = turn.processor
    ids = list(tokenizer.encode(TEXT))[:32]
    predicted = turn.response.fork()
    try:
        rev, observed, rows = p.preview_eof_tokens(ids)
        assert rev == p.preview_revision and observed == ids and len(rows) == len(ids)
        predicted_events = []
        for token_events, _ in rows:
            for event in token_events:
                predicted_events.append(event.to_json())
                predicted.append(event)
        eof_events = rows[-1][1]
        for event in eof_events:
            predicted_events.append(event.to_json())
            predicted.append(event)
        actual_events = []
        for token in ids:
            for event in p.push(d.InferenceChunk.token(token)):
                actual_events.append(event.to_json())
                turn.response.append(event)
        for event in p.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
            actual_events.append(event.to_json())
            turn.response.append(event)
        for event in p.finish():
            actual_events.append(event.to_json())
            turn.response.append(event)
        assert predicted_events == actual_events
        assert predicted.to_json() == turn.response.to_json()
    finally:
        p.close()


def test_wide_prefix_provenance_and_foreign_processor_rejected():
    _, d, tokenizer, turn = fixture()
    _, _, _, other = fixture()
    try:
        oracle = ToolEOFPreview(turn)
        ids = list(tokenizer.encode(TEXT))
        # Include the complete invocation, but not the later native block terminal.
        proof = oracle.preview(ids[:36], 0)
        assert proof.observed_count == 32 and proof.candidate_ids == tuple(ids[:36])
        assert proof.includes_eof and proof.boundary == 'M11_TOOL_EOF'
        with pytest.raises(RuntimeError, match='stale'):
            oracle.validate(proof, ids[:35], 0)
        with pytest.raises(RuntimeError, match='stale'):
            ToolEOFPreview(other).validate(proof, ids[:36], 0)
    finally:
        turn.processor.close()
        other.processor.close()
