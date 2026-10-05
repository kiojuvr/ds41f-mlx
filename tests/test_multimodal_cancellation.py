"""Protected worker cancellation retains one ownership/lifecycle authority."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend, RecipePreparedRequest


def request():
    return RecipePreparedRequest('chat_completions',SimpleNamespace(model=None,stream=False,inference_options=SimpleNamespace(max_tokens=8)),None,[1,2],[])


async def entered(event):
    for _ in range(200):
        if event.is_set(): return
        await asyncio.sleep(.005)
    raise AssertionError('worker never entered')


def test_discard_ready_prefill_burns_aliases_and_releases_certificate(monkeypatch):
    import gc,weakref
    from test_prefill_fp8_mlx_p1_p2 import FakeLanguageModel,full_ready_cache
    from test_prefill_fp8_mlx_p6 import fake_p6_app
    from ds41f_mlx.prefill_fp8_mlx import LivePrefillResult,LivePrefillContinuation
    monkeypatch.setattr('ds41f_mlx.prefill_fp8_mlx.handoff._validate_live_cache_structure',lambda *a:None)
    old=gc.isenabled();gc.disable()
    try:
        app=fake_p6_app(FakeLanguageModel(),full_ready_cache(),list(range(16)))
        app.execute_all();certificate=app.commit_certificate
        live=LivePrefillResult.from_committed(certificate,prefix_token_ids=list(range(16)))
        cache=live.live_cache;ref=weakref.ref(certificate);del app,certificate
        live.discard()
        assert ref() is None and live.cache_authority_owner=='invalidated'
        assert all(c._p6_append_failed and c._p6_append_invalid for c in cache)
        with pytest.raises(Exception):live.live_cache
        with pytest.raises(Exception):LivePrefillContinuation.from_cache(cache)
    finally:
        if old:gc.enable()


def test_cancel_drains_worker_and_runs_cleanup_on_same_thread():
    backend=DeepSeekRecipeRuntimeBackend()
    begin,release=threading.Event(),threading.Event()
    calls=[]
    def work():
        calls.append(('work',threading.get_ident()));begin.set();assert release.wait(5)
        return object()
    def cleanup(result): calls.append(('cleanup',threading.get_ident()))
    async def run():
        task=asyncio.create_task(backend._call(work,on_cancel=cleanup))
        await entered(begin)
        task.cancel();await asyncio.sleep(.02)
        assert not task.done()
        task.cancel();release.set()
        with pytest.raises(asyncio.CancelledError): await task
        assert calls[0][1]==calls[1][1] and calls[1][0]=='cleanup'
    try: asyncio.run(run())
    finally: release.set();backend.close()


def test_cancelled_queued_turn_clears_busy_without_releasing_other_lock():
    backend=DeepSeekRecipeRuntimeBackend()
    async def run():
        rec=await backend.create_stateful_session()
        await backend._lock.acquire()
        task=asyncio.create_task(backend.run_stateful_chat_turn(rec.session_id,request(),tokenizer=None))
        await asyncio.sleep(.01);assert rec.busy
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert not rec.busy and backend._lock.locked()
        backend._lock.release()
    try: asyncio.run(run())
    finally: backend.close()


def test_cancelled_initial_turn_publishes_completed_protocol_state():
    backend=DeepSeekRecipeRuntimeBackend()
    begin,release=threading.Event(),threading.Event()
    turn=SimpleNamespace(to_json=lambda:{'completed':True},finish_reason='stop',tool_calls=(),generated_tokens=(7,))
    m8=SimpleNamespace(generation=None,close=lambda:None)
    sess=SimpleNamespace(m8=m8,diagnostics=lambda:{'m8':{'frontier':3,'turns':[]}})
    def work(*_): begin.set();assert release.wait(5);return sess,turn
    backend.load=lambda:None;backend.make_sampler=lambda _:None;backend._start_and_run_m11=work
    async def run():
        rec=await backend.create_stateful_session()
        task=asyncio.create_task(backend.run_stateful_chat_turn(rec.session_id,request(),tokenizer=None))
        await entered(begin);task.cancel();await asyncio.sleep(.02)
        assert rec.busy and backend._lock.locked() and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError): await task
        assert rec.m11 is sess and rec.last_turn=={'completed':True}
        assert rec.request_count==1 and not rec.busy and not backend._lock.locked()
        assert backend.session_traces[-1]['ok'] and backend.session_traces[-1]['cancelled']
    try: asyncio.run(run())
    finally: release.set();backend.close()
