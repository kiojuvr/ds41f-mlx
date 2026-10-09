"""Exact recipe preparation and admission regressions, not inference evidence."""
import base64
import hashlib
import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.serving.capacity import resolve_capacity
from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer, RequestError
from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
from ds41f_mlx.web_tools import readable_text


@pytest.fixture(autouse=True)
def preserve_runtime_environment():
    # RuntimeConfig.apply_environment in existing preparation/backend tests must
    # not leak standard-off defaults into later singleton admission tests.
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def test_auto_is_actual_recipe_tokens_not_frontier_subtraction():
    cfg = load_runtime_config(); cfg.apply_import_paths()
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    value = {'model': cfg.model_id, 'messages': [{'role': 'user', 'content': 'A new input with structural tokens.'}], 'reasoning_effort': 'none', 'max_tokens': 'auto'}
    automatic = prepare_request('chat_completions', json.dumps(value).encode(), tokenizer=tokenizer)
    value['max_tokens'] = 1
    custom = prepare_request('chat_completions', json.dumps(value).encode(), tokenizer=tokenizer)
    assert automatic.token_ids == custom.token_ids
    assert automatic.capacity['prompt_tokens'] == len(custom.token_ids)
    assert automatic.resolved_max_tokens == 1048576 - len(custom.token_ids)
    assert custom.resolved_max_tokens == 1
    value['max_tokens'] = 1048576
    with pytest.raises(RequestError, match='total envelope'):
        prepare_request('chat_completions', json.dumps(value).encode(), tokenizer=tokenizer)


def test_source_review_preparation_preserves_recipe_encoding():
    from deepseek_recipe import ChatCompletionRequest, ConversionOptions, DeepseekV41Encoding
    cfg = load_runtime_config(); cfg.apply_import_paths()
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    source = (Path(__file__).resolve().parents[1] / 'ds41f_mlx/mtp_profile.py').read_text()
    value = {'model': cfg.model_id, 'max_tokens': 16, 'messages': [
        {'role': 'user', 'content': 'Review source'},
        {'role': 'assistant', 'tool_calls': [{'id': 'read-1', 'function': {'name': 'read', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'read-1', 'content': source}]}
    raw = json.dumps(value).encode()
    converted = ChatCompletionRequest(raw).convert(ConversionOptions(default_thinking_mode=False))
    authoritative_ids = DeepseekV41Encoding().with_tokenizer(tokenizer).encode(converted.conversation)
    prepared = prepare_request('chat_completions', raw, tokenizer=tokenizer,
                               checkpoint=cfg.checkpoint_path, ordinary=True)
    assert prepared.token_ids == authoritative_ids
    assert prepared.converted.conversation.messages[-1].content == source


def test_unbacked_image_placeholder_in_escaped_arguments_is_rejected():
    from deepseek_recipe import IMAGE_SPECIAL_TOKEN
    cfg = load_runtime_config(); cfg.apply_import_paths()
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    value = {'model': cfg.model_id, 'max_tokens': 16, 'messages': [
        {'role': 'user', 'content': 'Read a file'},
        {'role': 'assistant', 'tool_calls': [{'id': 'read-1', 'function': {
            'name': 'read', 'arguments': json.dumps({'path': IMAGE_SPECIAL_TOKEN})}}]},
        {'role': 'tool', 'tool_call_id': 'read-1', 'content': 'OK'}]}
    # Recipe can preserve escaped JSON arguments, then unescape them on render.
    # Existing encoded-placeholder admission still protects the text-only runtime.
    with pytest.raises(RequestError, match='missing image source'):
        prepare_request('chat_completions', json.dumps(value).encode(), tokenizer=tokenizer,
                        checkpoint=cfg.checkpoint_path, ordinary=True)


def test_ordinary_auto_uses_resolved_budget_for_execution():
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    cfg = load_runtime_config(); cfg.apply_import_paths()
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    value = {'model': cfg.model_id, 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 'auto'}
    prepared = prepare_request('chat_completions', json.dumps(value).encode(),
                               tokenizer=tokenizer, checkpoint=cfg.checkpoint_path, ordinary=True)
    assert prepared.inference_options.max_tokens == 1  # recipe placeholder, not execution budget
    assert prepared.resolved_max_tokens == 393216
    backend = object.__new__(ProductionMTPBackend)
    assert backend.request_max_tokens(prepared) == 393216


def test_ordinary_rejects_images_before_render_or_expansion(monkeypatch):
    from ds41f_mlx.serving import server
    def forbidden():
        pytest.fail('rejected image request reached rendering/expansion')
    monkeypatch.setattr(server, 'DeepseekV41Encoding', forbidden)
    value = {'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/a.png'}}]}]}
    with pytest.raises(ValueError, match='text Chat'):
        prepare_request('chat_completions', json.dumps(value).encode(), tokenizer=None, ordinary=True)


def test_qualified_boundary_and_model_boundary_are_total_and_separate(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'text_config': {'max_position_embeddings': 1048576}}))
    request = SimpleNamespace(token_ids=list(range(8000)), multimodal=object(), inference_options=SimpleNamespace(max_tokens=1))
    budget = resolve_capacity(request, checkpoint=tmp_path, automatic=True)
    assert budget['qualified_total_tokens'] == 8192
    assert budget['max_tokens'] == 192
    request.token_ids = list(range(8192))
    with pytest.raises(ValueError, match='capacity exhausted'): resolve_capacity(request, checkpoint=tmp_path, automatic=True)
    (tmp_path/'config.json').write_text(json.dumps({'max_position_embeddings': 8192}))
    request.multimodal = None; request.token_ids = list(range(8191))
    assert resolve_capacity(request, checkpoint=tmp_path, automatic=True)['max_tokens'] == 1


@pytest.mark.parametrize('prompt,maximum,accepted', [
    (600000, 393216, True), (900000, 393216, False),
    (655360, 393216, True), (655361, 393216, False),
    (32, 32768, True), (32, 393217, False),
])
def test_ordinary_output_and_total_capacity(tmp_path, prompt, maximum, accepted):
    (tmp_path / 'config.json').write_text(json.dumps({'max_position_embeddings': 1048576}))
    request = SimpleNamespace(token_ids=range(prompt), multimodal=None,
                              inference_options=SimpleNamespace(max_tokens=maximum))
    if accepted:
        budget = resolve_capacity(request, checkpoint=tmp_path, ordinary=True)
        assert budget['max_tokens'] == maximum
        assert budget['model_output_ceiling'] == 393216
    else:
        reason = 'output capability ceiling' if maximum > 393216 else r'prompt \+ requested output exceeds 1,048,576'
        with pytest.raises(ValueError, match=reason):
            resolve_capacity(request, checkpoint=tmp_path, ordinary=True)


def test_budget_prefix_guard_never_mutates_count_or_authority(monkeypatch):
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=load_runtime_config())
    rec = SimpleNamespace(busy=False, recovery_state='ready', request_count=7, m11=SimpleNamespace(m8=SimpleNamespace(token_history=[1,2,3]), image_identities=[]))
    monkeypatch.setattr(backend, 'get_stateful_session', lambda _: rec)
    request = SimpleNamespace(model=None, multimodal=None, image_sources=[], token_ids=[1,2,3,4])
    assert backend.validate_stateful_admission('s', request) is rec
    assert rec.request_count == 7 and not rec.busy
    request.token_ids = [1,2,5,4]
    with pytest.raises(RuntimeError, match='exact extension'): backend.validate_stateful_admission('s', request)
    assert rec.request_count == 7 and rec.m11.m8.token_history == [1,2,3]


def test_auto_vision_uses_expanded_original_bytes_and_separate_envelope():
    cfg = load_runtime_config(); cfg.apply_import_paths()
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    image = (cfg.checkpoint_path / 'inference/examples/images/corn.jpeg').read_bytes()
    value = {'model': cfg.model_id, 'reasoning_effort': 'none', 'max_tokens': 'auto', 'messages': [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Name this vegetable.'},
         {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(image).decode()}}]}]}
    automatic = prepare_request('chat_completions', json.dumps(value).encode(), tokenizer=tokenizer)
    value['max_tokens'] = 1
    custom = prepare_request('chat_completions', json.dumps(value).encode(), tokenizer=tokenizer)
    assert automatic.token_ids == custom.token_ids
    assert automatic.multimodal.identities() == custom.multimodal.identities()
    assert automatic.multimodal.identities()[0]['sha256'] == hashlib.sha256(image).hexdigest()
    assert automatic.capacity['envelope_kind'] == 'multimodal'
    assert automatic.resolved_max_tokens == 8192 - len(custom.token_ids)
    assert len(custom.token_ids) > 100  # actual expanded image, not one placeholder


def test_budget_http_is_observation_only_and_count_fenced():
    import asyncio
    import httpx
    from ds41f_mlx.serving.server import create_app
    cfg = load_runtime_config()
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    app = create_app(backend=backend, runtime_config=cfg)
    async def run():
        rec = await backend.create_stateful_session()
        before = rec.to_json()
        value = {'model': cfg.model_id, 'messages': [{'role': 'user', 'content': 'Budget only.'}],
                 'reasoning_effort': 'none', 'max_tokens': 'auto'}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            endpoint = f'/v1/sessions/{rec.session_id}/budget'
            budget = await client.post(endpoint, json=value, headers={'X-DS41F-Expected-Request-Count': '0'})
            assert budget.status_code == 200 and not budget.json()['binding']
            assert budget.json()['max_tokens'] + budget.json()['prompt_tokens'] == 1048576
            changed = await client.post(endpoint, json=value, headers={'X-DS41F-Expected-Request-Count': '1'})
            assert changed.status_code == 409
            value['max_tokens'] = 1048576
            exhausted = await client.post(endpoint, json=value)
            assert exhausted.status_code == 400
            assert exhausted.json()['error']['code'] == 'context_capacity_exhausted'
        assert backend._model is None and rec.m11 is None and rec.to_json() == before
    asyncio.run(run())


def test_readable_fetch_text_does_not_feed_scripts_or_html_markup():
    assert readable_text('<h1>Title</h1><script>attack()</script><style>hidden</style><p>Hello &amp; world</p>', 'text/html') == 'Title Hello & world'
