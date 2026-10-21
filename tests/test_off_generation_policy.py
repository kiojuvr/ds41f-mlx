"""Unchanged Standard-OFF Recipe 0.1 lane; run in its own pinned environment."""
import json
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('changes,temperature,top_p', [({}, 1, .95),
    ({'temperature': None, 'top_p': None}, 1, .95),
    ({'temperature': 0, 'top_p': 1}, 0, 1)])
def test_off_defaults_remain_recipe01_owned(monkeypatch, changes, temperature, top_p):
    from ds41f_mlx.serving import server, capacity
    class Encoding:
        def with_tokenizer(self, tokenizer): return self
        def render_conversation(self, conversation): return NS(image_sources=[])
        def encode(self, conversation): return [1, 2]
    monkeypatch.setattr(server, 'DeepseekV41Encoding', Encoding)
    monkeypatch.setattr(capacity, 'resolve_capacity', lambda *a, **k: None)
    body = dict(model='deepseek-v4.1-flash', messages=[{'role': 'user', 'content': 'Hi'}]) | changes
    request = server.prepare_request('chat_completions', json.dumps(body).encode(),
                                    tokenizer=NS(encode=lambda text: [1]), ordinary=False)
    assert request.generation_policy.temperature == request.inference_options.temperature == temperature
    assert request.generation_policy.top_p == request.inference_options.top_p == pytest.approx(top_p)
