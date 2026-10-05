"""The free allocation budget is lifetime-scoped, never a live-KV eviction."""
from types import SimpleNamespace
import pytest
from ds41f_mlx.runtime import resource_admission as a
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig


class Allocator:
    def __init__(self, limit):
        self.limit=limit; self.calls=[]; self.wired=0; self.wired_calls=[]; self.syncs=0
    def device_info(self): return {'max_recommended_working_set_size': 498216206336}
    def synchronize(self): self.syncs+=1
    def set_wired_limit(self, value):
        self.wired_calls.append(value); old=self.wired; self.wired=value; return old
    def set_cache_limit(self, value):
        self.calls.append(value); old=self.limit; self.limit=value; return old


def resources(mx):
    return a.AdmittedResources('test', 'native', {'mlx.core':mx}, {}, {})


@pytest.mark.parametrize('old', [0, 8*1024**3, 522268023193])
def test_never_loosen_and_restore_exactly_once(old):
    mx=Allocator(old); r=resources(mx)
    try:
        r.acquire_allocator_policy()
        assert mx.limit == min(old,32*1024**3)
        assert mx.wired == 498216206336
        calls=list(mx.calls); wired_calls=list(mx.wired_calls)
        r.acquire_allocator_policy(); assert mx.calls==calls and mx.wired_calls==wired_calls
        r.retire(); assert mx.limit==old and mx.wired==0 and not r.active
        calls=list(mx.calls); r.retire(); assert mx.calls==calls
        with pytest.raises(a.ResourceAdmissionError,match='retired'): r.acquire_allocator_policy()
    finally: r.retire()


def test_overlapping_owner_fails_without_changing_global_limit():
    mx=Allocator(100*1024**3); first=resources(mx); second=resources(mx)
    try:
        first.acquire_allocator_policy(); calls=list(mx.calls)
        with pytest.raises(a.ResourceAdmissionError,match='already owned'): second.acquire_allocator_policy()
        second.retire(); assert mx.calls==calls
        first.retire(); assert mx.limit==100*1024**3
    finally: first.retire(); second.retire()


def test_concurrent_acquisition_has_exactly_one_owner():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    mx=Allocator(100*1024**3); owners=[resources(mx),resources(mx)]; barrier=Barrier(2)
    def acquire(r):
        barrier.wait()
        try: r.acquire_allocator_policy(); return True
        except a.ResourceAdmissionError: return False
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(acquire,owners))==[False,True]
    finally:
        for r in owners: r.retire()
    assert mx.limit==100*1024**3 and a._allocator_owner is None


def test_setter_failure_does_not_claim_owner():
    mx=Allocator(0)
    def fail(value): raise RuntimeError('allocator unavailable')
    mx.set_cache_limit=fail; r=resources(mx)
    with pytest.raises(RuntimeError): r.acquire_allocator_policy()
    assert r._old_cache_limit is None and a._allocator_owner is None
    r.retire()


def test_smaller_limit_set_failure_rolls_back_without_owner():
    mx=Allocator(8*1024**3); setter=mx.set_cache_limit; calls=0
    def fail_once(value):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('set failed')
        return setter(value)
    mx.set_cache_limit=fail_once; r=resources(mx)
    with pytest.raises(RuntimeError): r.acquire_allocator_policy()
    assert mx.limit==8*1024**3 and a._allocator_owner is None
    r.retire()


def test_retire_failure_revokes_execution_keeps_owner_until_retry():
    mx=Allocator(100*1024**3); r=resources(mx); other=resources(mx)
    setter=mx.set_cache_limit
    try:
        r.acquire_allocator_policy()
        mx.set_cache_limit=lambda value: (_ for _ in ()).throw(RuntimeError('restore failed'))
        with pytest.raises(RuntimeError): r.retire()
        assert not r.active
        with pytest.raises(a.ResourceAdmissionError,match='already owned'): other.acquire_allocator_policy()
        mx.set_cache_limit=setter; r.retire()
        assert mx.limit==100*1024**3 and a._allocator_owner is None
    finally:
        mx.set_cache_limit=setter; r.retire(); other.retire()


def test_runtime_close_releases_model_even_if_policy_restore_fails():
    closed=[]; rt=OmlxRuntime()
    def fail(): raise RuntimeError('restore failed')
    rt.admission=SimpleNamespace(retire=fail)
    rt.model=SimpleNamespace(close=lambda:closed.append(True))
    with pytest.raises(RuntimeError): rt.close()
    assert closed==[True] and rt.model is None and rt.processor is None


def test_load_failure_restores_policy_and_does_not_fall_back(monkeypatch):
    mx=Allocator(100*1024**3); r=resources(mx); r._checkpoint='/verified'
    def fail(*args,**kwargs):
        assert mx.limit==32*1024**3
        raise RuntimeError('load failed')
    r.modules['ds41f_mlx.model_execution.loading']=SimpleNamespace(load=fail)
    monkeypatch.setattr(a,'prepare_resources',lambda *args:r)
    rt=OmlxRuntime(OmlxRuntimeConfig(preserve_mtp=False))
    with pytest.raises(RuntimeError,match='load failed'): rt.load_model()
    assert mx.limit==100*1024**3 and not r.active and rt.model is None
    rt.close()
