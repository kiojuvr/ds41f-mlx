"""Actual recipe/tokenizer proof of the NEXT runtime boundary, no target math."""
import json
from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
from ds41f_mlx.runtime.tool_boundary_session import (
    _make_processor_and_response, _record_protocol_output,
    _recent_tool_arguments_look_complete, _probe_tool_calls_complete,
)


def test_native_token_preview_does_not_authorize_standard_m11_eof_tool_horizon():
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    import deepseek_recipe as d
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    body = json.dumps(dict(model=cfg.model_id,
        messages=[dict(role='user', content='Weather?')], max_tokens=64,
        reasoning_effort='none', tools=[dict(type='function', function=dict(
            name='lookup_weather', parameters=dict(type='object', properties=dict(city=dict(type='string')))))])).encode()
    prepared = prepare_request('chat_completions', body, tokenizer=tokenizer, recipe_path=cfg.recipe_path)
    processor, response = _make_processor_and_response(d, prepared, 'm54-eof', cfg.model_id, tokenizer)
    events = []
    text = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup_weather">\n'
            '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
            '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    ids = list(tokenizer.encode(text))
    earliest_eof = native_terminal = None
    try:
        processor.push(d.InferenceChunk.ready())
        for i, token in enumerate(ids):
            before = processor.semantic_snapshot()
            preview = processor.preview_tokens([token])
            assert preview.mapping_exact and processor.semantic_snapshot() == before
            for out in processor.push(d.InferenceChunk.token(token)):
                _record_protocol_output(out, response, events)
            eof = (_recent_tool_arguments_look_complete(prepared.protocol, events)
                and _probe_tool_calls_complete(d, prepared, 'm54-eof', cfg.model_id, tokenizer, ids[:i+1]))
            if eof and earliest_eof is None:
                earliest_eof = i
                assert preview.terminal_kind is None and processor.semantic_terminal is None
            if preview.terminal_kind is not None:
                native_terminal = i
        assert earliest_eof is not None and native_terminal is not None
        assert earliest_eof < native_terminal
        # Pinned current recipe evidence: EOF-sensitive completion is six token
        # inputs earlier than native DSML block-end. They are not interchangeable.
        assert (earliest_eof, native_terminal) == (31, 37)
    finally:
        processor.close()
