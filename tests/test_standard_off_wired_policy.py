"""Resident OFF weights must survive P6/idle, not only a decode lease."""
import pytest
from tests.test_standard_off_allocator_policy import Allocator, resources
from ds41f_mlx.runtime import resource_admission as a


def test_restores_nondefault_caller_wired_budget():
    mx = Allocator(100*1024**3); mx.wired = 123
    r = resources(mx)
    try:
        r.acquire_allocator_policy()
        assert mx.wired == mx.device_info()['max_recommended_working_set_size']
        r.retire()
        assert mx.wired == 123 and mx.syncs == 2
    finally: r.retire()


def test_wired_acquisition_failure_restores_cache():
    mx = Allocator(100*1024**3); r = resources(mx)
    mx.set_wired_limit = lambda value: (_ for _ in ()).throw(RuntimeError('wire failed'))
    with pytest.raises(RuntimeError, match='wire failed'): r.acquire_allocator_policy()
    assert mx.limit == 100*1024**3 and a._allocator_owner is None
    r.retire()


def test_wired_restore_failure_retains_exclusive_retired_owner():
    mx = Allocator(100*1024**3); r = resources(mx); other = resources(mx)
    setter = mx.set_wired_limit
    try:
        r.acquire_allocator_policy()
        mx.set_wired_limit = lambda value: (_ for _ in ()).throw(RuntimeError('wire restore failed'))
        with pytest.raises(RuntimeError): r.retire()
        assert not r.active and mx.limit == 32*1024**3
        with pytest.raises(a.ResourceAdmissionError, match='already owned'): other.acquire_allocator_policy()
        mx.set_wired_limit = setter
        r.retire()
        assert mx.wired == 0 and mx.limit == 100*1024**3 and a._allocator_owner is None
    finally:
        mx.set_wired_limit = setter; r.retire(); other.retire()


def test_synchronize_failure_cannot_retire_or_admit_another_owner():
    mx = Allocator(100*1024**3); r = resources(mx); sync = mx.synchronize
    try:
        r.acquire_allocator_policy()
        mx.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('sync failed'))
        with pytest.raises(RuntimeError): r.retire()
        assert not r.active and a._allocator_owner is r
        mx.synchronize = sync; r.retire()
        assert mx.wired == 0 and a._allocator_owner is None
    finally: mx.synchronize = sync; r.retire()
