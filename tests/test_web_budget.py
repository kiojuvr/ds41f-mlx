import json
from pathlib import Path

import pytest

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer, RequestError
from ds41f_mlx.web_client import RuntimeHTTPError
from ds41f_mlx.web_budget import fit_tool_results


def test_fetch_context_fit_uses_actual_recipe_for_every_candidate(tmp_path):
    cfg = load_runtime_config(); cfg.apply_import_paths()
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    (tmp_path/'config.json').write_text(json.dumps({'max_position_embeddings': 2048}))
    class Runtime:
        calls = []
        def request(self, method, path, value):
            self.calls.append(value)
            try: prepared = prepare_request('chat_completions', json.dumps(value).encode(), tokenizer=tokenizer, checkpoint=tmp_path)
            except RequestError as error:
                raise RuntimeHTTPError(400, json.dumps({'error': {'code': error.code}})) from error
            return {**prepared.capacity, 'request_count': 1}
    runtime = Runtime()
    base = {'model': cfg.model_id, 'max_tokens': 'auto', 'reasoning_effort': 'none', 'messages': [
        {'role': 'user', 'content': 'Read the URL.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function', 'function': {'name': 'fetch_url', 'arguments': '{"url":"https://example.com"}'}}]},
    ]}
    payload = {'url': 'https://example.com', 'excerpt': 'many distinct words, unicode あいうえお. ' * 1000, 'offset': 0}
    original = [{'role': 'tool', 'tool_call_id': 'call1', 'content': json.dumps(payload)}]
    messages, displays, budget = fit_tool_results(runtime, 's', 1, base, original, [{'tool': 'fetch_url', **payload}])
    output = json.loads(messages[0]['content'])
    assert output['context_truncated'] and output['termination_reason'] == 'context_budget'
    assert output['next_offset'] == len(output['excerpt'])
    assert len(output['excerpt']) < len(payload['excerpt'])
    final = prepare_request('chat_completions', json.dumps({**base, 'messages': [*base['messages'], *messages]}).encode(), tokenizer=tokenizer, checkpoint=tmp_path)
    assert final.capacity == {key: value for key, value in budget.items() if key != 'request_count'}
    assert json.loads(original[0]['content']) == payload  # original effects preserved
    assert len(runtime.calls) > 2


def test_non_capacity_error_does_not_become_truncation_or_effect_retry():
    class Runtime:
        def request(self, *args): raise RuntimeHTTPError(409, '{"error":{"code":"session_conflict"}}')
    with pytest.raises(RuntimeHTTPError):
        fit_tool_results(Runtime(), 's', 1, {'messages': []}, [{'role': 'tool', 'content': '{"excerpt":"body"}'}], [{'tool': 'fetch_url'}])
