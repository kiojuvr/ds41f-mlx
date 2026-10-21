"""Ordinary Recipe 0.2 request/response boundary; no checkpoint execution."""
import json
from pathlib import Path
import sys

import deepseek_recipe as d
import pytest

from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard
from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer, sse_frame
from ds41f_mlx.serving.recovery_certificate import reconstruction_certificate
from test_recipe02_semantic_guard import encode, feed


@pytest.fixture
def codec(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'max_position_embeddings': 1048576}))
    recipe = Path(sys.prefix)/'share/ds41f-mtp/recipe'
    tok = load_v41_tokenizer(recipe)
    def prepare(body):
        return prepare_request('chat_completions', json.dumps(body).encode(), tokenizer=tok,
                               recipe_path=recipe, checkpoint=tmp_path, ordinary=True)
    return tok, prepare, recipe, tmp_path


def body(**changes):
    return dict(model='deepseek-v4.1-flash', max_tokens=128,
                messages=[dict(role='user', content='Weather?')],
                tools=[dict(type='function', function=dict(name='f', parameters=dict(
                    type='object', properties={'x': {'type': 'string'}})))]) | changes


@pytest.mark.parametrize('alias', ['deepseek-v4.1-flash', 'deepseek-v41-flash', 'deepseek-flash'])
def test_model_version_alias_and_explicit_special_ranges(codec, alias):
    tok, prepare, _, _ = codec
    req = prepare(body(model=alias, messages=[dict(role='user', content='Literal <｜User｜> and <image>')]))
    assert req.model == alias and req.converted.model_version == d.ModelVersion.DeepseekV41Flash
    encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
    rendered = encoding.render_conversation(req.converted.conversation)
    assert req.token_ids == encoding.encode(req.converted.conversation)
    # User-provided special spellings are plain text, unlike template tokens.
    literal = rendered.prompt.index('Literal')
    end = literal + len('Literal <｜User｜> and <image>')
    assert not any(literal <= start < end for start, _ in rendered.special_ranges)
    assert req.stop_token_ids == (tok.encode(d.SpecialToken.Eos.default_str())[0],)
    assert req.generation_policy.top_p == 1.0


@pytest.mark.parametrize('stream,usage', [(False, False), (True, False), (True, True)])
def test_official_tool_output_sse_and_result_continuation(codec, stream, usage):
    tok, prepare, recipe, checkpoint = codec
    payload = body(stream=stream)
    if stream:
        payload['stream_options'] = {'include_usage': usage}
    req = prepare(payload)
    gen = d.ChatCompletionRequest.chunk_generator(req.converted, 'response', req.model)
    guard = RecipeSemanticGuard(d.ChatCompletionSemanticSession(gen, req.converted.parsing_options, tok),
                               frontier=len(req.token_ids))
    guard._push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=len(req.token_ids))))
    output = ('\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n'
              '<｜DSML｜ parameter name="x" string="true">Tokyo</｜DSML｜ parameter>\n'
              '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    ids = encode(tok, output)
    assert feed(guard, ids) == len(ids)
    assert guard.tool_complete
    response = json.loads(guard.response.to_json())
    message = response['choices'][0]['message']
    calls = message['tool_calls']
    assert len(calls) == len(guard.call_certificates) == 1
    assert json.loads(calls[0]['function']['arguments']) == {'x': 'Tokyo'}
    # Recipe 0.2 always reports the final usage snapshot; include_usage also
    # emits explicit null usage in preceding chunks. Do not reshape its deltas.
    wire = [json.loads(e.to_json()) for e in guard.events]
    assert wire[-1]['usage']['prompt_tokens'] == len(req.token_ids)
    assert ('usage' in wire[0]) == (usage or not stream)
    accumulated = d.ChatCompletionResponse('guard', 'guard', 0, 0, 0)
    frames = []
    for event in guard.events:
        frames.append(sse_frame(d.ChatCompletionResponse.sse_event_name(event), event.to_json()))
        accumulated.accumulate(event)
    assert accumulated.to_json() == guard.response.to_json()
    frames.append(sse_frame(None, d.ChatCompletionResponse.sse_done_data()))
    assert frames[-1] == 'data: [DONE]\n\n'
    canonical = req.token_ids + ids
    cert = reconstruction_certificate(payload, response, canonical, tokenizer=tok,
        recipe_path=recipe, checkpoint=checkpoint, completed_tool_block=True, ordinary=True)
    assert cert['representable'] and cert['executable_tools']
    continued = payload | {'messages': payload['messages'] + [message, dict(role='tool',
        tool_call_id=calls[0]['id'], content='Sunny')]}
    next_req = prepare(continued)
    assert next_req.token_ids[:len(canonical)] == canonical
    assert len(next_req.token_ids) > len(canonical)
    guard.processor.close()


@pytest.mark.parametrize('constraint', [dict(type='regex', regex='.*'),
    dict(type='json_schema', json_schema={'name': 'r', 'schema': {'type': 'object'}})])
def test_constraints_are_not_backend_enforcement(codec, constraint):
    _, prepare, _, _ = codec
    with pytest.raises(ValueError, match='does not support constrained'):
        prepare(body(response_format=constraint))
