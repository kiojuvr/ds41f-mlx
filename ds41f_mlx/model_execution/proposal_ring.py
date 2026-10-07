# SPDX-License-Identifier: MIT
# Physical ring algorithm derived from oMLX deepseek_v4_dspark (MIT), modified.
"""Discardable bounded proposal context, never executable target state."""
import mlx.core as mx


class DSparkContextCache:
    def __init__(self, max_size):
        self.max_size = int(max_size)
        if self.max_size < 1:
            raise ValueError('positive proposal window required')
        self.offset = 0
        self.keys = None

    def append(self, keys, *, start_offset=None):
        start = self.offset if start_offset is None else int(start_offset)
        if self.keys is not None and start != self.offset:
            raise ValueError('noncontiguous proposal context')
        old = self.keys
        if old is not None and old.shape[2] == self.max_size:
            cutoff = self.offset % self.max_size
            if cutoff:
                old = mx.concatenate([old[:, :, cutoff:], old[:, :, :cutoff]], 2)
        joined = keys if old is None else mx.concatenate([old, keys], 2)
        end = start + keys.shape[2]
        joined = joined[:, :, -self.max_size:]
        if joined.shape[2] == self.max_size and end % self.max_size:
            split = self.max_size - end % self.max_size
            joined = mx.concatenate([joined[:, :, split:], joined[:, :, :split]], 2)
        # Explicit output allocation, not a slice retaining prefill parents.
        self.keys = mx.take(joined, mx.arange(joined.shape[2]), axis=2)
        mx.eval(self.keys)
        self.offset = end

    def retire(self):
        self.keys = None
