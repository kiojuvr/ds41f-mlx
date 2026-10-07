"""Request-scoped MTP wiring lease, acquired before target prefix allocation.

This owns only an MLX resource limit, never target or speculative state. The
native BatchGenerator retains its ordinary close/restore contract after transfer.
"""
from __future__ import annotations


class MTPWiredLimitLease:
    def __init__(self, mx, stream):
        self.mx, self.stream = mx, stream
        self.previous = None
        self.limit = None
        self.transferred = False

    def __enter__(self):
        self.limit = self.mx.device_info().get('max_recommended_working_set_size')
        if self.limit is not None:
            self.previous = self.mx.set_wired_limit(self.limit)
        return self

    def transfer_to(self, generator):
        if self.transferred:
            raise RuntimeError('MTP wired-limit lease already transferred')
        if self.limit is not None:
            # The ordinary native constructor re-acquires the same limit.
            # Its close must restore our original limit, not the nested value.
            if generator._old_wired_limit != self.limit:
                # Dispose the newly constructed empty generator before the
                # lease restores its original limit. A deferred finalizer must
                # not later restore the mismatched nested limit over that value.
                generator.close()
                raise RuntimeError('MTP generator wired-limit acquisition mismatch')
            generator._old_wired_limit = self.previous
        self.transferred = True

    def __exit__(self, *_):
        if not self.transferred and self.previous is not None:
            self.mx.synchronize(self.stream)
            self.mx.set_wired_limit(self.previous)
            self.previous = None
