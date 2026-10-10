"""Immutable paired state authority, indexed/evicted by pinned oMLX PagedCache.

There is no independent lookup, LRU, or conversation rewind here. Within the
qualified context envelope every extendable checkpoint is an upstream root tail.
The payload is memory-only: losing either owner is a miss, never unprimed MTP.
"""
from dataclasses import dataclass
from typing import Any
from time import perf_counter

from .capacity import TEXT_QUALIFIED_ENVELOPE


def _copy(value, mx):
    if value is None:
        return None
    copier = getattr(value, 'copy', None)
    # Pinned MLX arrays are functional values (setitem replaces the array
    # graph); mx.array is the upstream snapshot primitive on MLX 0.32.2.
    return copier() if copier is not None else mx.array(value)


def _arrays(target, rings):
    return [v for layer in target for v in layer.cache if v is not None] + [r.keys for r in rings if r.keys is not None]


def validate_pair(target, rings, tokens):
    C = len(tokens)
    if not 0 < C <= TEXT_QUALIFIED_ENVELOPE - 3 or len(target) != 40 or len(rings) != 3:
        raise ValueError('incomplete paired checkpoint')
    for layer in target:
        if len(layer.cache) != 7 or layer.size() != C or layer.cache[0] is None or layer.cache[1] is None:
            raise ValueError('target checkpoint frontier/slots mismatch')
        if getattr(layer, '_p6_append_failed', False) or getattr(layer, '_p6_append_pending', False):
            raise ValueError('uncommitted target checkpoint')
        ratio = layer.compress_ratio
        if ratio is None or ratio < 0:
            raise ValueError('unknown target checkpoint layout')
        if ratio and C >= ratio and any(layer.cache[i] is None for i in (2, 3)):
            raise ValueError('missing compressed target checkpoint state')
        if ratio > 1 and C % ratio and any(layer.cache[i] is None for i in (4, 5)):
            raise ValueError('missing compressor tail')
        if any(v.shape[0] != 1 for v in layer.cache if v is not None):
            raise ValueError('paired checkpoint is singleton only')
    for ring in rings:
        if ring.max_size != 128 or ring.offset != C or ring.keys is None or ring.keys.shape[0] != 1 or ring.keys.shape[2] != min(C, 128):
            raise ValueError('incomplete DSpark physical ring')


def clone_pair(target, rings, mx):
    # Array-value snapshots, not pack/unpack, storage reconstruction, or model
    # forwards. Independent array/cache handles on capture AND acquisition;
    # immutable MLX backing may be shared until a functional update. No active
    # mutable cache container or ring object is shared with retained state.
    new_target = []
    for layer in target:
        cache_type = getattr(layer, 'cache_type', type(layer))
        item = cache_type.from_state([_copy(v, mx) for v in layer.cache], layer.meta_state)
        item.left_padding = _copy(getattr(layer, 'left_padding', None), mx)
        item.lengths = _copy(getattr(layer, 'lengths', None), mx)
        new_target.append(item)
    new_rings = []
    for ring in rings:
        cache_type = getattr(ring, 'cache_type', type(ring))
        item = cache_type(ring.max_size)
        item.offset, item.keys = ring.offset, _copy(ring.keys, mx)
        new_rings.append(item)
    mx.eval(*_arrays(new_target, new_rings))
    return new_target, new_rings


@dataclass(frozen=True)
class TargetSnapshot:
    cache_type: type
    cache: tuple[Any, ...]
    meta_state: tuple
    compress_ratio: int
    left_padding: Any
    lengths: Any

    def size(self):
        return int(self.cache[0].item())


@dataclass(frozen=True)
class RingSnapshot:
    cache_type: type
    max_size: int
    offset: int
    keys: Any


@dataclass(frozen=True)
class PairedCheckpoint:
    identity: str
    tokens: tuple[int, ...]
    target: tuple[Any, ...]
    rings: tuple[Any, ...]

    @classmethod
    def capture(cls, identity, target, rings, tokens, mx):
        tokens = tuple(map(int, tokens))
        validate_pair(target, rings, tokens)
        target, rings = clone_pair(target, rings, mx)
        validate_pair(target, rings, tokens)
        return cls(identity, tokens,
                   tuple(TargetSnapshot(type(c), tuple(c.cache), tuple(c.meta_state), c.compress_ratio,
                                        c.left_padding, c.lengths) for c in target),
                   tuple(RingSnapshot(type(c), c.max_size, c.offset, c.keys) for c in rings))

    def restore(self, identity, prompt, mx):
        if self.identity != identity or tuple(prompt[:len(self.tokens)]) != self.tokens:
            raise ValueError('paired checkpoint identity mismatch')
        validate_pair(self.target, self.rings, self.tokens)
        target, rings = clone_pair(self.target, self.rings, mx)
        validate_pair(target, rings, self.tokens)
        return target, rings, list(self.tokens)


class PairedCheckpointAuthority:
    def __init__(self, identity, mx, *, retained_checkpoints=4, context_tokens=TEXT_QUALIFIED_ENVELOPE):
        from omlx.cache.paged_cache import PagedCacheManager
        if not 3 <= context_tokens <= TEXT_QUALIFIED_ENVELOPE:
            raise ValueError('invalid paired context envelope')
        self.identity, self.mx = identity, mx
        self.context_tokens = context_tokens
        # Complete pairs are indivisible root tails, not token-grid KV blocks.
        # The null block consumes one slot. Four immutable pairs remain bounded;
        # live/restored handles are outside this reusable-state capacity.
        self.paged = PagedCacheManager(block_size=context_tokens, max_blocks=retained_checkpoints+1,
                                      initial_blocks=retained_checkpoints+1, model_name=identity)
        self._payloads = {}
        self.paged.on_block_hash_dropped = self._drop_payload
        self.paged.on_hash_map_cleared = self._payloads.clear

    def _drop_payload(self, key):
        checkpoint = self._payloads.pop(key, None)
        if checkpoint is not None:
            self.paged.stats.total_tokens_cached -= len(checkpoint.tokens)

    def capture(self, target, rings, tokens):
        if len(tokens) + 3 > self.context_tokens:
            raise ValueError('checkpoint cannot be extended within context envelope')
        return PairedCheckpoint.capture(self.identity, target, rings, tokens, self.mx)

    def publish(self, checkpoint):
        if checkpoint.identity != self.identity or len(checkpoint.tokens) + 3 > self.context_tokens:
            raise ValueError('foreign or unextendable checkpoint publication')
        validate_pair(checkpoint.target, checkpoint.rings, checkpoint.tokens)
        tokens = list(checkpoint.tokens)
        if self.paged.find_cached_block(tokens) is not None:
            return
        if not self.paged.get_stats().free_blocks:
            # Use the upstream LRU candidate/refcount policy, never a second
            # retention index. Active acquisition leases are not evictable.
            for candidate in self.paged.get_evictable_blocks(1):
                self.paged.evict_block_permanently(candidate.block_id)
        block = self.paged.allocate_block()
        if block is None:
            return
        try:
            self.paged.register_block_hash(block, tokens)
            block.token_count = len(tokens)
            self._payloads[block.block_hash] = checkpoint
            self.paged.stats.total_tokens_cached += len(tokens)
            self.paged.register_tail_block(None, block.block_hash, len(tokens))
        finally:
            self.paged.release_for_eviction([block.block_id])

    def acquire(self, prompt, *, timings=None):
        # Leave a genuine suffix append before P5's terminal holdout. In
        # particular, NEVER turn an exact N hit into N-1 by recurrent trimming.
        lookup_t0 = perf_counter()
        restore_s = 0.
        if timings is not None:
            timings.update(cache_lookup_s=0., paired_restore_s=0.)
        eligible = list(prompt[:-2])
        while eligible:
            blocks, C = self.paged.get_computed_blocks(eligible)
            if not blocks:
                break
            block = blocks[-1]
            key = block.block_hash
            checkpoint = self._payloads.get(key)
            leased = self.paged.acquire_cached_block(block.block_id, key)
            if leased is None:
                break
            try:
                if checkpoint is not None and len(checkpoint.tokens) == C:
                    try:
                        restore_t0 = perf_counter()
                        try:
                            target, rings, tokens = checkpoint.restore(self.identity, prompt, self.mx)
                        finally:
                            restore_s += perf_counter()-restore_t0
                            if timings is not None:
                                timings.update(paired_restore_s=restore_s,
                                               cache_lookup_s=perf_counter()-lookup_t0-restore_s)
                        return target, rings, tokens
                    except (ValueError, TypeError, AttributeError, IndexError):
                        pass
            finally:
                self.paged.release_for_eviction([block.block_id])
            # Missing/invalid pair cannot restore target alone. Invalidate the
            # indexed candidate, then let upstream select a shorter root tail.
            self.paged.evict(key)
        if timings is not None:
            timings.update(paired_restore_s=restore_s,
                           cache_lookup_s=perf_counter()-lookup_t0-restore_s)
        return None

    def clear(self):
        self.paged.clear()
