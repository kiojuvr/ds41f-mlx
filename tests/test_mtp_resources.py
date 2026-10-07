from types import SimpleNamespace

import pytest

from ds41f_mlx.runtime.mtp_resources import MTPWiredLimitLease


class MX:
    def __init__(self, recommended=100, initial=7):
        self.recommended, self.limit, self.calls = recommended, initial, []
    def device_info(self):
        return {'max_recommended_working_set_size': self.recommended}
    def set_wired_limit(self, value):
        previous = self.limit
        self.calls.append(('set', value))
        self.limit = value
        return previous
    def synchronize(self, stream):
        self.calls.append(('sync', stream))


@pytest.mark.parametrize('initial', [0, 7])
@pytest.mark.parametrize('failure', [ValueError, BaseException])
def test_pre_transfer_failure_restores_after_sync(initial, failure):
    mx = MX(initial=initial)
    with pytest.raises(failure), MTPWiredLimitLease(mx, 'generation'):
        assert mx.limit == 100
        raise failure('prefill failed or cancelled')
    assert mx.limit == initial
    assert mx.calls == [('set', 100), ('sync', 'generation'), ('set', initial)]


def test_native_generator_owns_original_restore_after_transfer():
    mx = MX()
    with MTPWiredLimitLease(mx, 'generation') as lease:
        generator = SimpleNamespace(_old_wired_limit=mx.set_wired_limit(100))
        lease.transfer_to(generator)
        assert generator._old_wired_limit == 7
        with pytest.raises(RuntimeError, match='already transferred'):
            lease.transfer_to(generator)
    assert mx.limit == 100  # protected decode remains wired until native close
    mx.synchronize('generation')
    mx.set_wired_limit(generator._old_wired_limit)
    assert mx.limit == 7


def test_mismatch_fails_closed_and_restores():
    mx = MX()
    generator = SimpleNamespace(_old_wired_limit=9)
    def close():
        mx.synchronize('generation')
        mx.set_wired_limit(generator._old_wired_limit)
        generator._old_wired_limit = None
    generator.close = close
    with pytest.raises(RuntimeError, match='mismatch'), MTPWiredLimitLease(mx, 'generation') as lease:
        lease.transfer_to(generator)
    assert mx.limit == 7
    assert generator._old_wired_limit is None


def test_no_recommended_limit_is_native_noop():
    mx = MX(recommended=None)
    with MTPWiredLimitLease(mx, 'generation') as lease:
        lease.transfer_to(SimpleNamespace(_old_wired_limit=None))
    assert not mx.calls
