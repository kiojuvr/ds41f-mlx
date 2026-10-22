"""Tool identity extraction against the actual supported Recipe 0.2 parser."""
import json
import deepseek_recipe as d

from ds41f_mlx.runtime.tool_boundary_session import _extract_tool_calls, _protocol_finish_reason
from verification_support import RECIPE


def test_m11_extracts_chat_tool_identity_from_recipe_response_shape():
    response = {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "lookup_weather", "arguments": '{"city":"Paris"}'}}
    ]}}]}
    calls = _extract_tool_calls('chat_completions', response, [])
    assert len(calls) == 1
    assert calls[0].id == 'call_1' and calls[0].name == 'lookup_weather'
    assert json.loads(calls[0].arguments) == {'city': 'Paris'}
    assert _protocol_finish_reason('chat_completions', response, []) == 'tool_calls'


def test_official_recipe02_stream_processor_parses_dsml_tool_call():
    tokenizer = d.Tokenizer.from_file(str(RECIPE/'static/tokenizers/v41/tokenizer.json'))
    converted = d.ChatCompletionRequest(dict(model='DeepSeek-V4.1-Flash',
        messages=[dict(role='user', content='Weather?')], reasoning_effort='none',
        tools=[dict(type='function', function=dict(name='lookup_weather'))])).convert(d.ConversionOptions())
    generator = d.ChatCompletionRequest.chunk_generator(converted, 'resp', 'm')
    processor = d.StreamProcessor(generator, converted.parsing_options, tokenizer)
    response = d.ChatCompletionResponse('resp', 'm', 0, 0, 0)
    text = '<｜DSML｜tool_calls><｜DSML｜invoke name="lookup_weather"><｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>'
    try:
        for chunk in (d.InferenceChunk.ready(), d.InferenceChunk.text(text, 10)):
            for out in processor.push(chunk):
                response.accumulate(out)
        for out in processor.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
            response.accumulate(out)
        for out in processor.finish():
            response.accumulate(out)
        assert processor.finished
        data = json.loads(response.to_json())
        calls = _extract_tool_calls('chat_completions', data, [])
        assert data['choices'][0]['finish_reason'] == 'tool_calls'
        assert calls[0].name == 'lookup_weather'
        assert json.loads(calls[0].arguments) == {'city': 'Paris'}
        assert calls[0].id
    finally:
        processor.close()
