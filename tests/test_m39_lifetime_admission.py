"""Actual internal admission methods; native execution intentionally absent."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend as Backend
from ds41f_mlx.serving.request_fence import reserve, finish


def request():
    return SimpleNamespace(protocol='chat_completions', image_sources=[], model=None,
                           token_ids=[1,2], inference_options=SimpleNamespace(max_tokens=3), stream=True)


def backend():
    b = Backend()
    b.make_sampler = lambda _: None
    b._retire = lambda rec: None  # no model/native authority allocated
    return b


def test_identity_eviction_and_admission_invariant():
    async def run():
        b = backend(); old = None
        try:
            for _ in range(160):
                rec = await b.create_stateful_session(); sid = rec.session_id
                old = old or sid
                before = (b._issued, deepcopy(rec.to_json()))
                with pytest.raises(RuntimeError): await b.create_stateful_session()
                assert before == (b._issued, rec.to_json())
                assert b.identity_state(sid) == 'live'
                rec.canonical = [1,2]; rec.last_turn = {'response': {}}
                rec.certificate = {'representable': True}; rec.guard = object()
                rec.reconstruction_body = {}; rec.reconstruction_tokenizer = object()
                await b.close_stateful_session(sid)
                assert rec.closed and not rec.canonical
                assert all(getattr(rec,k) is None for k in ('owner','cache','rings','processor','guard',
                           'last_turn','certificate','fence','reconstruction_body','reconstruction_tokenizer'))
            assert len(b.retired_diagnostics) == 16 and not b.sessions
            assert old not in [d['id'] for d in b.retired_diagnostics]
            rec = await b.create_stateful_session()
            before = (b._issued, deepcopy(rec.to_json()), deepcopy(list(b.retired_diagnostics)))
            for sid in (old, 'arbitrary', f'mtp_{b._namespace}_{0:032x}',
                        f'mtp_{b._namespace}_{b._issued+1:032x}', 'mtp_'+'0'*32+'_'+f'{1:032x}'):
                with pytest.raises(KeyError): b.get_stateful_session(sid)
                with pytest.raises(KeyError): await b.close_stateful_session(sid)
                with pytest.raises(ValueError): await b.create_stateful_session(session_id=sid)
                with pytest.raises(KeyError):
                    await b.qualification_response(sid,request(),tokenizer=None,body=b'{}',sequence=1)
                assert before == (b._issued,rec.to_json(),list(b.retired_diagnostics))
            assert b.identity_state(old) == 'retired'
            assert b.identity_state('mtp_'+'0'*32+'_'+f'{1:032x}') == 'stale_namespace'
            assert b.identity_state(f'mtp_{b._namespace}_{b._issued+1:032x}') == 'never_valid'
            await b.close_stateful_session(rec.session_id)
        finally: b.close()
    asyncio.run(run())


@pytest.mark.parametrize('kwargs', [dict(body=b'x'*1048577,sequence=1),dict(body=b'{}',sequence=2**64),
                                    dict(body=b'{}',sequence=True),dict(body='{}',sequence=1)])
def test_capacity_before_reservation_or_native_mutation(kwargs):
    async def run():
        b = backend()
        try:
            rec = await b.create_stateful_session(); before = deepcopy(rec.to_json())
            b._start = lambda *_: pytest.fail('native authority created')
            with pytest.raises(ValueError): await b.qualification_response(rec.session_id,request(),tokenizer=None,**kwargs)
            assert rec.to_json() == before and not b._lock.locked() and not b.session_traces
            await b.close_stateful_session(rec.session_id)
        finally: b.close()
    asyncio.run(run())


def test_fixed_counter_exhaustion_and_last_settled_retry():
    async def run():
        b = backend()
        try:
            b._issued = b.MAX_LIFETIMES-1
            rec = await b.create_stateful_session()
            assert rec.session_id.endswith('f'*32)
            reserve(rec,b.MAX_SEQUENCE,b'{}'); rec.last_turn = {'response': {'choices': []}}
            rec.certificate = {'representable': True}; finish(rec)
            before = deepcopy(rec.to_json())
            for _ in range(3):
                out = await b.qualification_response(rec.session_id,request(),tokenizer=None,body=b'{}',sequence=b.MAX_SEQUENCE,outcome_projection=True)
                assert json.loads(out.body)['outcome_state'] == 'recoverable'
            assert before == rec.to_json()
            with pytest.raises(ValueError):
                await b.qualification_response(rec.session_id,request(),tokenizer=None,body=b'{}',sequence=b.MAX_SEQUENCE+1)
            rec.fence = None  # synthetic legacy ceiling, independently bounded
            with pytest.raises(RuntimeError,match='exhausted'):
                await b.qualification_response(rec.session_id,request(),tokenizer=None,body=b'{}')
            await b.close_stateful_session(rec.session_id)
            with pytest.raises(RuntimeError,match='exhausted'): await b.create_stateful_session()
            assert not b.sessions and b._issued == b.MAX_LIFETIMES
        finally: b.close()
    asyncio.run(run())


def test_failed_retirement_keeps_singleton_authority_blocked():
    async def run():
        b = backend()
        try:
            rec = await b.create_stateful_session()
            def fail(_): raise RuntimeError('native close failed')
            b._retire = fail
            with pytest.raises(RuntimeError): await b.close_stateful_session(rec.session_id)
            assert b.identity_state(rec.session_id) == 'live' and rec.poisoned
            assert not b.retired_diagnostics and not b._lock.locked()
            with pytest.raises(RuntimeError): await b.create_stateful_session()
            b._retire = lambda _: None
            await b.close_stateful_session(rec.session_id)
        finally: b.close()
    asyncio.run(run())
