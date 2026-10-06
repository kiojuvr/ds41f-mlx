"""Transport/control regressions. Real checkpoint qualification is separate."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from ds41f_mlx.config import load_runtime_config
from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request
from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend
from ds41f_mlx.runtime.live_turn import LiveRecipeTurn
from ds41f_mlx.runtime.tool_boundary_session import M11RecipeToolSession


class M8Double:
    def __init__(self, tokens, output, delay=0):
        self.token_history = list(tokens)
        self.output = list(output)
        self.total_prompt_replay_count = self.total_full_cache_repack_count = 0
        self.generation = object()
        self.closed = False
        self.delay = delay
        self.consumed = 0
    @property
    def state(self): return 'closed' if self.closed else ('generating' if self.generation is not None else 'idle')
    @property
    def frontier(self): return len(self.token_history)
    def next_token(self):
        time.sleep(self.delay)
        token = self.output.pop(0)
        self.token_history.append(token)  # test's committed transaction
        self.consumed += 1
        return SimpleNamespace(token=token, finish_reason='stop' if not self.output else None)
    def ensure_idle(self, reason): self.generation = None
    def cancel_turn(self, reason): self.generation = None
    def close(self): self.generation = None; self.closed = True
    def diagnostics(self):
        return dict(frontier=self.frontier, all_cache_offsets_equal_frontier=True,
                    cache_offsets_all=[self.frontier]*40, total_prompt_replay_count=0,
                    total_full_cache_repack_count=0)


def setup(monkeypatch, delay=0):
    cfg = load_runtime_config()
    cfg.apply_import_paths()
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    body = json.dumps(dict(model=cfg.model_id, messages=[dict(role='user', content='Count.')],
                           stream=True, reasoning_effort='none', max_tokens=512)).encode()
    prepared = prepare_request('chat_completions', body, tokenizer=tokenizer, recipe_path=cfg.recipe_path)
    from deepseek_recipe import EOS_TOKEN
    output = list(tokenizer.encode(''.join(f'{i}\n' for i in range(1, 150)))) + list(tokenizer.encode(EOS_TOKEN))
    backend = DeepSeekRecipeRuntimeBackend(runtime_config=cfg)
    monkeypatch.setattr(backend, 'load', lambda: None)
    monkeypatch.setattr(backend, 'make_sampler', lambda _: None)
    def start(**kwargs):
        return M11RecipeToolSession(model=None, tokenizer=tokenizer, checkpoint=cfg.checkpoint_path,
            omlx_path=cfg.omlx_path, recipe_path=cfg.recipe_path, protocol='chat_completions', model_id=cfg.model_id,
            m8=M8Double(prepared.token_ids, output, delay))
    monkeypatch.setattr(M11RecipeToolSession, 'start_from_prepared', staticmethod(start))
    return backend, tokenizer, body, prepared


def test_delivery_is_complete_independent_of_64_event_diagnostics(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        nonce = 'c7f29ea7-51b5-43e1-b8b4-4458c779fd55'
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body, application_id=nonce)
        events = []
        async for frame in stream:
            if '[DONE]' not in frame:
                events.append(json.loads(frame[6:]))
        content = ''.join(e.get('choices', [{}])[0].get('delta', {}).get('content') or '' for e in events)
        assert len(events) > 64
        assert content == rec.last_turn['response_json']['choices'][0]['message']['content']
        assert len(rec.last_turn['stream_events']) == 64
        assert rec.last_turn['application_request_id'] == nonce
        assert rec.busy is False and not backend._lock.locked()
    asyncio.run(run())
    backend.close()


def test_preiteration_close_and_stale_cancel_do_not_leak_or_release_new_reservation(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        first = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        assert rec.busy
        await first.aclose()
        second = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        await first.aclose()
        assert rec.active_stream is second and rec.busy
        with pytest.raises(RuntimeError, match='stale'):
            await backend.cancel_stateful_stream(rec.session_id, first.request_id)
        await second.aclose()
        assert not rec.busy and rec.m11 is None
    asyncio.run(run())
    backend.close()


def test_cancel_drains_only_one_target_transaction_and_reconstructs(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch, delay=.05)
    async def run():
        rec = await backend.create_stateful_session()
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        await anext(stream)  # committed prompt, role/ready
        job = asyncio.create_task(anext(stream))
        await asyncio.sleep(.01)
        job.cancel()
        with pytest.raises(asyncio.CancelledError): await job
        assert rec.m11.m8.consumed == 1
        assert not rec.busy and not backend._lock.locked()
        assert rec.last_turn['cancelled']
        assert rec.last_turn['reconstruction']['representable']
        assert rec.m11.m8.generation is None
    asyncio.run(run())
    backend.close()


def test_slow_consumer_does_not_advance_generation_between_demands(monkeypatch):
    backend, tokenizer, body, prepared = setup(monkeypatch)
    async def run():
        rec = await backend.create_stateful_session()
        stream = await backend.open_stateful_stream(rec.session_id, prepared, tokenizer=tokenizer, body=body)
        await anext(stream)
        count = rec.m11.m8.consumed
        await asyncio.sleep(.05)
        assert rec.m11.m8.consumed == count
        await backend.cancel_stateful_stream(rec.session_id, stream.request_id)
        assert not rec.busy and not backend._lock.locked()
    asyncio.run(run())
    backend.close()
