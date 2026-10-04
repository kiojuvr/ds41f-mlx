import json
import pytest

from ds41f_mlx.runtime.tool_boundary_session import _extract_tool_calls, _protocol_finish_reason


def test_m11_extracts_chat_tool_identity_from_recipe_response_shape():
    response = {
        "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup_weather", "arguments": "{\"city\": \"Paris\"}"}}
        ]}}]
    }
    calls = _extract_tool_calls("chat_completions", response, [])
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "lookup_weather"
    assert json.loads(calls[0].arguments) == {"city": "Paris"}
    assert _protocol_finish_reason("chat_completions", response, []) == "tool_calls"


def test_official_recipe_stream_processor_parses_dsml_tool_call_when_available():
    d = __import__("deepseek_recipe")
    tokenizer = d.Tokenizer.from_file(str(__import__("pathlib").Path(__import__("sys").prefix)/"share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json"))
    text = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup_weather">\n<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    gen = d.ChatCompletionChunkGenerator("resp", "m", False, True)
    processor = d.StreamProcessor(gen, d.ParsingOptions(), tokenizer)
    response = d.ChatCompletionResponse("resp", "m", 0, 0, 0)
    for out in processor.push(d.InferenceChunk.ready()):
        response.append(out)
    for out in processor.push(d.InferenceChunk.text(text, 10)):
        response.append(out)
    for out in processor.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
        response.append(out)
    for out in processor.finish():
        response.append(out)
    data = json.loads(response.to_json())
    calls = _extract_tool_calls("chat_completions", data, [])
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert calls[0].name == "lookup_weather"
    assert json.loads(calls[0].arguments) == {"city": "Paris"}
