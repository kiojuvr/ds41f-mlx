"""Internal DeepSeek-V4.1 DSpark/MTP stateful lifecycle primitives.

This module is deliberately *not* wired into public serving defaults.  It owns
only the M29 internal lifecycle boundary around pinned upstream oMLX DSpark/MTP:

* canonical server history and transport-delivered history are distinct;
* the 40-layer target cache is the only executable target authority;
* DSpark rings are bounded committed context derived from target hidden taps;
* cancellation means bounded canonical quiescence, not token-exact abort.

The existing :mod:`ds41f_mlx.runtime.omlx_generation` MTP-OFF session remains the
qualified production implementation.  Classes here are opt-in/internal and fail
closed when the native singleton topology cannot be proven.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from threading import RLock
from functools import wraps
from typing import Any, Callable, Sequence
import importlib
import sys

import numpy as np

from ds41f_mlx.runtime.omlx_core import DEFAULT_OMLX
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig


class MTPLifecycleError(RuntimeError):
    """Unsupported or incoherent MTP lifecycle state."""


@dataclass(frozen=True)
class DSparkCommittedContext:
    """Bounded native DSpark committed-context owner.

    ``caches`` are the upstream DSpark ring cache objects.  They are not target
    truth and must never be used as a second executable target cache.
    """

    caches: tuple[Any, ...]
    frontier: int
    target_layer_ids: tuple[int, ...]
    stage_count: int
    window_size: int | None = None

    @classmethod
    def from_native(cls, caches: Sequence[Any], *, frontier: int, target_layer_ids: Sequence[int], window_size: int | None = None) -> "DSparkCommittedContext":
        offsets = tuple(int(getattr(c, "offset")) for c in caches)
        if not offsets or any(o != int(frontier) for o in offsets):
            raise MTPLifecycleError(f"DSpark offsets {offsets} do not match frontier {frontier}")
        return cls(tuple(caches), int(frontier), tuple(int(i) for i in target_layer_ids), len(offsets), window_size)

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(int(getattr(c, "offset")) for c in self.caches)

    def validate_idle(self, canonical_frontier: int) -> None:
        if self.offsets != tuple([int(canonical_frontier)] * len(self.caches)):
            raise MTPLifecycleError("DSpark committed context is not aligned to canonical frontier")

    def to_json(self) -> dict[str, Any]:
        return {
            "frontier": self.frontier,
            "offsets": list(self.offsets),
            "target_layer_ids": list(self.target_layer_ids),
            "stage_count": self.stage_count,
            "window_size": self.window_size,
            "authority": "bounded committed DSpark context; not executable target truth",
        }


@dataclass
class CanonicalTransportHistory:
    """Separate canonical server output from interrupted transport delivery."""

    prompt_tokens: tuple[int, ...]
    canonical_generated_tokens: list[int] = field(default_factory=list)
    transport_delivered_tokens: list[int] = field(default_factory=list)

    @property
    def canonical_tokens(self) -> list[int]:
        return [*self.prompt_tokens, *self.canonical_generated_tokens]

    @property
    def canonical_frontier(self) -> int:
        return len(self.canonical_tokens)

    @property
    def delivered_frontier(self) -> int:
        return len(self.prompt_tokens) + len(self.transport_delivered_tokens)

    @property
    def recovery_suffix_tokens(self) -> list[int]:
        delivered = len(self.transport_delivered_tokens)
        return [int(t) for t in self.canonical_generated_tokens[delivered:]]

    def record_delivered(self, token: int) -> None:
        t = int(token)
        self.transport_delivered_tokens.append(t)
        self.canonical_generated_tokens.append(t)

    def commit_undelivered(self, tokens: Sequence[int]) -> None:
        self.canonical_generated_tokens.extend(int(t) for t in tokens)

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt_len": len(self.prompt_tokens),
            "canonical_frontier": self.canonical_frontier,
            "delivered_frontier": self.delivered_frontier,
            "transport_delivered_tokens": list(self.transport_delivered_tokens),
            "canonical_generated_tokens": list(self.canonical_generated_tokens),
            "recovery_suffix_tokens": self.recovery_suffix_tokens,
        }


@dataclass(frozen=True)
class QuiescenceCounters:
    new_verify_cycles: int = 0
    new_proposals: int = 0
    target_forwards: int = 0
    history_replay: int = 0
    full_cache_repack: int = 0
    dspark_appends: int = 0

    def require_m28_zeroes(self) -> None:
        if self.new_verify_cycles or self.new_proposals or self.history_replay or self.full_cache_repack:
            raise MTPLifecycleError(f"forbidden quiescence work: {self}")

    def to_json(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CanonicalQuiescenceResult:
    target_cache: list[Any]
    canonical_tokens: tuple[int, ...]
    dspark_context: DSparkCommittedContext
    recovery_suffix_tokens: tuple[int, ...]
    discarded_future_token: int | None
    counters: QuiescenceCounters
    latency_s: float
    queue_drained_tokens: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "canonical_frontier": len(self.canonical_tokens),
            "recovery_suffix_tokens": list(self.recovery_suffix_tokens),
            "discarded_future_token": self.discarded_future_token,
            "counters": self.counters.to_json(),
            "latency_s": self.latency_s,
            "queue_drained_tokens": list(self.queue_drained_tokens),
            "dspark": self.dspark_context.to_json(),
        }


class NativeDSparkPriming:
    """Narrow sidecar seam for upstream-owned DSpark projection/append math."""

    def __init__(self, language_model: Any):
        self.language_model = language_model
        self.config = getattr(language_model, "_config", None)
        ids = getattr(self.config, "dspark_target_layer_ids", None)
        if ids is None:
            raise MTPLifecycleError("loaded language model does not expose dspark_target_layer_ids")
        self.target_layer_ids = tuple(int(x) for x in ids)

    def make_context_from_native(self, caches: Sequence[Any], *, frontier: int) -> DSparkCommittedContext:
        window = getattr(self.config, "dspark_block_size", None)
        return DSparkCommittedContext.from_native(caches, frontier=frontier, target_layer_ids=self.target_layer_ids, window_size=None if window is None else int(window))

    def append_hidden(self, hidden: Any, caches: Sequence[Any], *, start_offset: int, end_frontier: int) -> DSparkCommittedContext:
        append = getattr(self.language_model, "dspark_append_context", None)
        if append is None:
            raise MTPLifecycleError("language model cannot append DSpark context")
        append(hidden, caches, start_offset=int(start_offset))
        return self.make_context_from_native(caches, frontier=int(end_frontier))

    def forward_and_append_once(self, token_ids: Sequence[int], *, target_cache: list[Any], dspark_caches: Sequence[Any], start_offset: int, mx: Any | None = None) -> DSparkCommittedContext:
        """Append suffix DSpark context from the same native target forward.

        This is the P6 continuity shape: the caller supplies the existing target
        cache and a suffix that is being committed exactly once.  We require the
        native ``return_dspark_hidden`` path so ds41f does not reimplement DSpark
        projection math.  The method is intentionally small and is not used by
        the MTP-OFF path.
        """
        if not token_ids:
            return self.make_context_from_native(dspark_caches, frontier=int(start_offset))
        mx = mx or importlib.import_module("mlx.core")
        arr = mx.array([list(map(int, token_ids))], dtype=mx.int64)
        out = self.language_model(arr, cache=target_cache, return_dspark_hidden=True)
        if not isinstance(out, tuple) or len(out) < 2:
            raise MTPLifecycleError("native forward did not return DSpark hidden taps")
        hidden = out[1]
        return self.append_hidden(hidden, dspark_caches, start_offset=int(start_offset), end_frontier=int(start_offset) + len(token_ids))


def _target_offsets(cache: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(c.size()) for c in cache)


def _dspark_offsets(caches: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(getattr(c, "offset")) for c in caches)


def canonical_quiesce_native_singleton(
    *,
    language_model: Any,
    target_cache: list[Any],
    mtp_state: Any,
    history: CanonicalTransportHistory,
    mx: Any | None = None,
    queue_pop: Callable[[Any], tuple[int, Any, str]] | None = None,
    observe_canonical: Callable[[int], Any] | None = None,
) -> CanonicalQuiescenceResult:
    """Promote the M28 diagnostic drain into a runtime primitive.

    The operation validates the frontier relation and mutates only the existing
    native singleton state.  It never calls ``BatchGenerator.next`` or any path
    that can start proposals/verification.  ``queue_pop`` exists for tests and
    alternate upstream queue containers; by default ``state.queue.pop(0)`` is
    used.
    """
    t0 = perf_counter()
    mx = mx or importlib.import_module("mlx.core")
    fs = _target_offsets(target_cache)
    if len(fs) != 40 or len(set(fs)) != 1:
        raise MTPLifecycleError(f"target cache must expose 40 aligned offsets, got {fs[:8]}")
    target_frontier = fs[0]
    ds_caches = getattr(mtp_state, "mtp_cache", None)
    if ds_caches is None:
        raise MTPLifecycleError("native MTP state has no DSpark cache")
    ds_offsets = _dspark_offsets(ds_caches)
    if len(set(ds_offsets)) != 1:
        raise MTPLifecycleError(f"DSpark offsets are not aligned: {ds_offsets}")
    queue = getattr(mtp_state, "queue", None)
    if queue is None:
        raise MTPLifecycleError("native MTP state has no response queue")
    h_frontier = history.canonical_frontier
    needed = target_frontier - h_frontier
    drained: list[int] = []
    discarded: int | None = None
    target_forwards = 0
    dspark_appends = 0

    if needed >= 0:
        if len(queue) != needed + 1:
            raise MTPLifecycleError(f"unknown queue topology: target-history={needed}, queue={len(queue)}")
        pop = queue_pop or (lambda q: q.pop(0))
        for _ in range(needed):
            entry = pop(queue)
            token = int(entry[0])
            if observe_canonical is not None:
                # The bounded committed safe suffix becomes canonical exactly
                # once. A semantic guard must reject any terminal in this drain.
                observe_canonical(token)
            drained.append(token)
        final = pop(queue)
        discarded = int(final[0])
        history.commit_undelivered(drained)
        # DSpark and target are already at the canonical target frontier.
        if _dspark_offsets(ds_caches) != tuple([target_frontier] * len(ds_caches)):
            raise MTPLifecycleError("DSpark ring was not aligned with target before drain")
    elif needed == -1 and len(queue) == 0:
        # One emitted/canonical token is ahead of target/DSpark. Materialize only
        # that bounded suffix; do not sample a successor.
        if not history.canonical_tokens:
            raise MTPLifecycleError("cannot materialize target-behind state with empty history")
        token = int(history.canonical_tokens[-1])
        priming = NativeDSparkPriming(language_model)
        priming.forward_and_append_once([token], target_cache=target_cache, dspark_caches=ds_caches, start_offset=target_frontier, mx=mx)
        target_forwards = 1
        dspark_appends = 1
    else:
        raise MTPLifecycleError(f"unsupported target/history relation target={target_frontier} history={h_frontier} queue={len(queue)}")

    # Clear transient singleton scheduler state without interpreting it as idle authority.
    for name in ("queue", "drafts", "next_main", "rollback_stash", "_rollback_stash", "anchor", "uid"):
        if hasattr(mtp_state, name):
            try:
                val = [] if name == "queue" else None
                setattr(mtp_state, name, val)
            except Exception:
                pass
    canonical_frontier = history.canonical_frontier
    if hasattr(mtp_state, 'hist_offset'):
        mtp_state.hist_offset = canonical_frontier
    final_offsets = _target_offsets(target_cache)
    final_ds = _dspark_offsets(ds_caches)
    if any(o != canonical_frontier for o in final_offsets) or any(o != canonical_frontier for o in final_ds):
        raise MTPLifecycleError(f"idle frontier mismatch target={final_offsets[:4]} dspark={final_ds} canonical={canonical_frontier}")
    if len(getattr(mtp_state, "queue", ())) != 0:
        raise MTPLifecycleError("queue not empty after quiescence")
    ctx = NativeDSparkPriming(language_model).make_context_from_native(ds_caches, frontier=canonical_frontier)
    counters = QuiescenceCounters(target_forwards=target_forwards, dspark_appends=dspark_appends)
    counters.require_m28_zeroes()
    return CanonicalQuiescenceResult(
        target_cache=target_cache,
        canonical_tokens=tuple(history.canonical_tokens),
        dspark_context=ctx,
        recovery_suffix_tokens=tuple(history.recovery_suffix_tokens),
        discarded_future_token=discarded,
        counters=counters,
        latency_s=perf_counter() - t0,
        queue_drained_tokens=tuple(drained),
    )


def _serialized_mtp_operation(method):
    @wraps(method)
    def operation(self, *args, **kwargs):
        with self._operation_lock:
            if self._operation_failed and method.__name__ != 'close':
                raise MTPLifecycleError('native MTP session failed closed')
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                self._operation_failed = True
                raise
    return operation


@dataclass
class OMLXMTPGenerationSession:
    """Internal opt-in native MTP session preserving MTP-OFF class identity."""

    model: Any
    initial_cache: list[Any]
    initial_token_ids: np.ndarray
    dspark_context: DSparkCommittedContext
    config: OMLXDecodeConfig = field(default_factory=lambda: OMLXDecodeConfig(preserve_mtp=True, speculation_enabled=True))
    sampler: Callable[[Any], Any] | None = None
    max_tokens: int = 128
    stream: Any | None = None
    semantic_guard: Any | None = None
    wired_limit_lease: Any | None = None

    def __post_init__(self) -> None:
        self._operation_lock = RLock()
        self._operation_failed = False
        if not self.config.speculation_enabled or self.config.preserve_mtp is not True:
            raise MTPLifecycleError("OMLXMTPGenerationSession is internal MTP-ON only")
        root = str(self.config.omlx_path or DEFAULT_OMLX)
        if root not in sys.path:
            sys.path.insert(0, root)
        self.mx = importlib.import_module("mlx.core")
        gen = importlib.import_module("mlx_lm.generate")
        try:
            importlib.import_module("omlx.scheduler")
            mtp = importlib.import_module("omlx.patches.mlx_lm_mtp.batch_generator")
            rb = importlib.import_module("omlx.patches.mlx_lm_mtp.cache_rollback")
            if self.semantic_guard is not None and getattr(mtp, 'SEMANTIC_HORIZON_VERSION', 0) != 1:
                raise MTPLifecycleError('guarded MTP requires the semantic-horizon candidate engine')
            if hasattr(mtp, "apply"):
                mtp.apply()
            if hasattr(rb, "apply"):
                rb.apply()
        except Exception as exc:
            raise MTPLifecycleError("pinned upstream oMLX MTP patches are unavailable") from exc
        self.BatchGenerator = gen.BatchGenerator
        self.generation_stream = gen.generation_stream
        self.stream = self.stream or self.generation_stream
        self.language_model = getattr(self.model, "language_model", self.model)
        if self.semantic_guard is not None and set(getattr(self.semantic_guard, 'control_token_ids', ())) - set(self.config.stop_token_ids or ()):
            raise MTPLifecycleError('suppressed control IDs require native backend stop matchers')
        if self.semantic_guard is not None and not all(callable(getattr(self.language_model, name, None)) for name in (
                'mtp_validate_committed_context', 'mtp_take_committed_context', 'mtp_install_committed_context')):
            raise MTPLifecycleError('guarded MTP requires native committed-context ownership hooks')
        if hasattr(self.language_model, "configure_mtp"):
            depth = int(getattr(getattr(self.language_model, "_config", object()), "n_mtp_layers", 5) or 5)
            self.language_model.configure_mtp(True, depth)
        self.sampler = self.sampler or (lambda logits: self.mx.argmax(logits, axis=-1))
        ids = np.asarray(self.initial_token_ids, dtype=np.int64).reshape(1, -1)
        self.prefix_tokens = [int(x) for x in ids[0].tolist()]
        if len(self.prefix_tokens) != self.dspark_context.frontier:
            raise MTPLifecycleError("DSpark context frontier does not match token history")
        if any(o != len(self.prefix_tokens) for o in _target_offsets(self.initial_cache)):
            raise MTPLifecycleError("target cache frontier does not match token history")
        self.history = CanonicalTransportHistory(prompt_tokens=tuple(self.prefix_tokens))
        guarded_stops = [[int(t)] for t in (self.config.stop_token_ids or ())] if self.semantic_guard is not None else None
        self._bg = self.BatchGenerator(self.language_model, max_tokens=self.max_tokens, sampler=self.sampler, stop_tokens=guarded_stops or None, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=2048, stream=self.stream)
        self.uid: int | None = None
        self._started = False
        self._closed = False
        self._quiesced = False
        if self.wired_limit_lease is not None:
            self.wired_limit_lease.transfer_to(self._bg)
            self.wired_limit_lease = None

    @_serialized_mtp_operation
    def start(self, terminal_prompt_token: int) -> None:
        if self._started:
            raise MTPLifecycleError("MTP session already started")
        # Upstream owns the transfer of primed DSpark context; do not rebuild it
        # from token history.  Different pinned revisions expose this either via
        # take/drop priming helpers or directly on the language model.
        install = getattr(self.language_model, 'mtp_install_committed_context', None)
        if callable(install):
            install(self.initial_cache, self.dspark_context.caches)
        elif hasattr(self.language_model, "_dspark_prime_context"):
            self.language_model._dspark_prime_context = self.dspark_context.caches
        uids = self._bg.insert([[int(terminal_prompt_token)]], max_tokens=[self.max_tokens], caches=[self.initial_cache], all_tokens=[list(self.prefix_tokens)], samplers=[self.sampler])
        self.uid = int(uids[0])
        pr, gr = self._bg.next(); self.mx.synchronize(self.stream)
        if gr:
            raise MTPLifecycleError("unexpected generation response during terminal bootstrap")
        self.history.prompt_tokens = tuple([*self.prefix_tokens, int(terminal_prompt_token)])
        if self.semantic_guard is not None:
            self._bg._generation_batch._omlx_semantic_guard = self.semantic_guard
        self.prompt_replay_count = 0
        self.last_response = None
        self._started = True
        self.initial_cache = []

    @_serialized_mtp_operation
    def next_token(self, *, transport_delivered: bool = True) -> int | None:
        if self._quiesced or (self.semantic_guard is not None and self.semantic_guard.finished):
            return None
        if not self._started:
            raise MTPLifecycleError("call start() before next_token")
        if transport_delivered and len(self.history.transport_delivered_tokens) != len(self.history.canonical_generated_tokens):
            raise MTPLifecycleError('transport delivery must remain a canonical prefix')
        _, gr = self._bg.next(); self.mx.synchronize(self.stream)
        if not gr:
            return None
        self.last_response = gr[0]
        token = int(gr[0].token)
        if transport_delivered:
            self.history.record_delivered(token)
        else:
            self.history.commit_undelivered([token])
        if self.semantic_guard is not None and gr[0].finish_reason is not None and not self.semantic_guard.finished:
            self.semantic_guard.finish_backend(gr[0].finish_reason)
        return token

    @_serialized_mtp_operation
    def confirm_delivery(self, token_ids: Sequence[int], *, start_ordinal: int) -> None:
        """Acknowledge the exact canonical response prefix, metadata only."""
        delivered = len(self.history.transport_delivered_tokens)
        ids = list(map(int, token_ids))
        if int(start_ordinal) != delivered or self.history.canonical_generated_tokens[delivered:delivered + len(ids)] != ids:
            raise MTPLifecycleError('transport acknowledgement is not the next canonical prefix')
        self.history.transport_delivered_tokens.extend(ids)

    def active_cache_offsets(self) -> tuple[int, ...] | None:
        gb = getattr(self._bg, '_generation_batch', None)
        cache = getattr(gb, 'prompt_cache', None)
        if not cache:
            return None
        return _target_offsets(cache)

    @_serialized_mtp_operation
    def quiesce(self) -> CanonicalQuiescenceResult:
        gb = getattr(self._bg, "_generation_batch", None)
        if gb is None:
            raise MTPLifecycleError("no active native generation batch to quiesce")
        horizon = getattr(gb, '_omlx_semantic_horizon', None)
        state = getattr(gb, "_omlx_mtp_state", None)
        cache = getattr(gb, "prompt_cache", None)
        # Backend finish filters the row. The optional horizon retains only
        # the exact idle-transition owners, never a reconstructed target cache.
        if horizon is not None and horizon.last_state is not None:
            state, cache = horizon.last_state, horizon.last_cache
        if state is None and self.semantic_guard is not None and cache:
            # External cancellation before lazy MTP activation: adopt only the
            # existing committed prompt rings and discard the standard pending
            # response. No forward, sample, proposal or verification is needed.
            from collections import deque
            from types import SimpleNamespace
            take = getattr(self.language_model, 'mtp_take_committed_context', None)
            pending = getattr(gb, '_next_tokens', None)
            if not callable(take) or pending is None or any(c.size() != self.history.canonical_frontier for c in cache):
                raise MTPLifecycleError('preactivation committed context is unavailable')
            rings, offset = take(cache)
            state = SimpleNamespace(mtp_cache=rings, queue=deque([(int(pending.tolist()[0]), None, 'pending')]))
        if state is None:
            raise MTPLifecycleError("native MTP state unavailable")
        if not cache:
            raise MTPLifecycleError("native target cache unavailable")
        observe = None if self.semantic_guard is None else lambda token: self.semantic_guard.observe_canonical_emit(token, None)
        with self.mx.stream(self.stream):
            # Match native completed-response ownership transfer for active
            # cancellation too. filter([]) clears the batch's cache container;
            # retaining its objects also retains the old P6 sealed capability.
            # Native singleton extraction transfers row views, not repacked
            # arrays or reconstructed history. The old prefill owner stays revoked.
            if cache is getattr(gb, 'prompt_cache', None):
                cache = gb.extract_cache(0)
            result = canonical_quiesce_native_singleton(language_model=self.language_model, target_cache=cache, mtp_state=state, history=self.history, mx=self.mx,
                                                       queue_pop=lambda q: q.popleft() if hasattr(q, 'popleft') else q.pop(0), observe_canonical=observe)
            self.mx.eval(*[c.keys for c in result.dspark_context.caches if c.keys is not None])
            self.mx.synchronize(self.stream)
        if horizon is not None:
            # Any un-emitted future terminal was discarded, not canonically
            # observed. Its ordinal ownership cannot leak into the next turn.
            horizon.pending = None
        if self.uid is not None:
            try:
                self._bg.remove([self.uid])
            except Exception:
                pass
            self.uid = None
        self._quiesced = True
        return result

    cancel = quiesce

    @_serialized_mtp_operation
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.uid is not None:
                self._bg.remove([self.uid])
        finally:
            self._bg.close()
