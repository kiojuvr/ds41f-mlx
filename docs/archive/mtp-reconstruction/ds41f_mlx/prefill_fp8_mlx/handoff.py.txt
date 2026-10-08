"""P5: validate and transfer the committed live cache once, without repacking.

Terminal-token contract (request ownership, not the last sweep's arena):
    full_prompt = complete_prefix_token_ids + [terminal_prompt_token]
Only the prefix was prefilled. Its length must equal all 40 cache frontiers.
The held-out terminal token is forwarded once by ds41f target-engine bootstrap;
it is not prompt replay. No final-prefix logits are needed. Never pass an
already-prefilled terminal token to start() again.

No cache creation, tensor copy/conversion, merge, continuation-state export,
or reconstruction from publication records occurs here. After transfer the
prefill runner is revoked; after bootstrap the ds41f engine is decode authority.
External diagnostic aliases are passive references, not executable authorities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING

from .executor import PrefillExecutionSetup

if TYPE_CHECKING:
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession


class LiveCacheHandoffError(RuntimeError):
    """Uncommitted, malformed, or already-consumed live-cache admission."""


def _shape(value: Any, *, layer: int, slot: int) -> tuple[int, ...]:
    if value is None or not hasattr(value, 'shape') or not hasattr(value, 'dtype'):
        raise LiveCacheHandoffError(f'layer {layer} slot {slot}: missing qualified array object')
    return tuple(int(x) for x in value.shape)


def _dtype(value: Any) -> str:
    return str(value.dtype).rsplit('.', 1)[-1]


def validate_committed_cache(setup: PrefillExecutionSetup, prefix_token_ids: Sequence[int]) -> int:
    """Scalar frontiers and array metadata only; never inspect tensor contents."""
    if hasattr(setup, "final_setup") and hasattr(setup, "sealed"):
        return _validate_p6_append_commit(setup, prefix_token_ids)
    if hasattr(setup, "p6_commit_authority"):
        return _validate_p6_append_commit(setup.p6_commit_authority, prefix_token_ids)
    arena, runner, manager = setup.arena, setup.block_runner, setup.publication_manager
    tx = arena.transaction
    if not tx.begun or not tx.committed or not tx.valid or tx.failed or manager.failed:
        raise LiveCacheHandoffError('prefill/publication transaction is not successfully committed')
    if manager.pending_cumulative_by_layer or manager.pending_spans_by_layer or arena.publications.pending_events:
        raise LiveCacheHandoffError('publication transaction still has pending sources')
    if runner.prefill_continuation_exported or arena.prefill_continuation_exported:
        raise LiveCacheHandoffError('PrefillContinuationState export is forbidden for live handoff')
    if runner.full_cache_repack_count != 0:
        raise LiveCacheHandoffError('cache repack occurred before live handoff')
    if runner.logits_policy.compute_final_prefix_logits or not runner.final_logits_suppressed or runner.final_logits is not None:
        raise LiveCacheHandoffError('serving prefix logits must be suppressed')
    cache = runner.working_cache
    if cache is None or not isinstance(cache, list) or len(cache) != 40:
        raise LiveCacheHandoffError('handoff requires the existing 40-layer live cache list')
    c = runner.language_model._config
    frontier = len(prefix_token_ids)
    if frontier <= 0 or frontier != int(arena.base_frontier) + int(arena.plan.count):
        raise LiveCacheHandoffError('complete prefix history length does not match committed frontier')
    _validate_live_cache_structure(cache, c, frontier)
    return frontier


def _validate_live_cache_structure(cache: list[Any], config: Any, frontier: int) -> None:
    if cache is None or not isinstance(cache, list) or len(cache) != 40:
        raise LiveCacheHandoffError('handoff requires the existing 40-layer live cache list')
    kv_sources = set(config.kv_source_layers)
    index_sources = set(config.index_source_layers)
    widths = {1: config.head_dim + config.head_dim // 32,
              2: config.head_dim // 2 + config.head_dim // 16,
              3: config.index_head_dim // 2 + config.index_head_dim // 32}
    for layer, item in enumerate(cache):
        if item is None:
            raise LiveCacheHandoffError(f'layer {layer}: missing cache')
        try:
            slots = [item[i] for i in range(7)]
            if _shape(slots[0], layer=layer, slot=0) != (1,) or _dtype(slots[0]) != 'int32':
                raise LiveCacheHandoffError(f'layer {layer}: invalid singleton frontier slot')
            offset = int(item.size())
        except (AttributeError, TypeError, IndexError) as exc:
            raise LiveCacheHandoffError(f'layer {layer}: unsupported cache representation') from exc
        if offset != frontier:
            raise LiveCacheHandoffError(f'layer {layer} frontier {offset} != complete prefix length {frontier}')
        ratio = int(config.compress_ratios[layer]) if layer in kv_sources else 0
        if getattr(item, 'compress_ratio', None) != ratio:
            raise LiveCacheHandoffError(f'layer {layer}: cache compression layout mismatch')
        for slot in (1, 2, 3):
            shape = _shape(slots[slot], layer=layer, slot=slot)
            if len(shape) != 3 or shape[0] != 1 or shape[2] != widths[slot] or _dtype(slots[slot]) != 'uint8':
                raise LiveCacheHandoffError(f'layer {layer} slot {slot}: invalid packed geometry/dtype')
            expected = None
            if slot == 1:
                expected = min(frontier, config.window_size)
            elif layer in kv_sources and (slot == 2 or layer in index_sources):
                expected = frontier // ratio
            if expected is not None and shape[1] != expected:
                raise LiveCacheHandoffError(f'layer {layer} slot {slot}: rows {shape[1]} != {expected}')
        pending = frontier % ratio if ratio > 1 else 0
        for slot in (4, 5):
            if _shape(slots[slot], layer=layer, slot=slot) != (1, pending, config.head_dim) or _dtype(slots[slot]) not in ('float32', 'bfloat16'):
                raise LiveCacheHandoffError(f'layer {layer} slot {slot}: invalid compressor pending state')
        if slots[4].dtype != slots[5].dtype:
            raise LiveCacheHandoffError(f'layer {layer}: pending KV/gate dtypes differ')
        if layer == 0 and config.engram_layer_ids:
            shape = _shape(slots[6], layer=layer, slot=6)
            if shape != (1, config.engram_max_ngram_size - 1) or _dtype(slots[6]) != 'int64':
                raise LiveCacheHandoffError('layer0 Engram history is absent or malformed')


def _validate_p6_append_commit(commit: Any, prefix_token_ids: Sequence[int]) -> int:
    if not getattr(commit, "sealed", False):
        raise LiveCacheHandoffError("P6 append commit is not sealed")
    T = int(commit.T)
    if tuple(int(t) for t in prefix_token_ids) != tuple(commit.prefix_token_ids) or len(prefix_token_ids) != T:
        raise LiveCacheHandoffError("P6 complete prefix history mismatch")
    if not (int(commit.E) == int(commit.D) == T):
        raise LiveCacheHandoffError("P6 physical frontiers are incomplete")
    if int(commit.history_position) != T:
        raise LiveCacheHandoffError("P6 Engram history is not committed at T")
    if any(v < T for v in commit.source_coverage.values()) or any(v < T for v in commit.layer_coverage.values()):
        raise LiveCacheHandoffError("P6 coverage does not reach T")
    setup = commit.final_setup
    runner, manager = setup.block_runner, setup.publication_manager
    if manager.pending_cumulative_by_layer or manager.pending_spans_by_layer or manager.visible_spans:
        raise LiveCacheHandoffError("P6 append commit has pending row-span publication")
    if runner.full_cache_repack_count != 0 or runner.prefill_continuation_exported or setup.arena.prefill_continuation_exported:
        raise LiveCacheHandoffError("P6 append commit used forbidden export/repack")
    if runner.logits_policy.compute_final_prefix_logits or not runner.final_logits_suppressed or runner.final_logits is not None:
        raise LiveCacheHandoffError("P6 serving prefix logits must be suppressed")
    cache = commit.live_cache
    if cache is not runner.working_cache or cache is None or len(cache) != 40:
        raise LiveCacheHandoffError("P6 commit does not own the same live cache")
    for item in cache:
        if getattr(item, "_p6_append_failed", False) or getattr(item, "_p6_append_invalid", False) or getattr(item, "_p6_append_pending", False) or not getattr(item, "_p6_append_sealed", False):
            raise LiveCacheHandoffError("P6 cache is not sealed/admissible")
        if getattr(item, "_p6_owner_token", None) != commit.owner_token:
            raise LiveCacheHandoffError("P6 owner token mismatch")
    _validate_live_cache_structure(cache, runner.language_model._config, T)
    return T


@dataclass(init=False)
class LivePrefillResult:
    """One-shot owner of the exact cache list produced by a committed setup."""
    prefix_token_ids: tuple[int, ...]
    frontier: int
    _setup: PrefillExecutionSetup
    _live_cache: list[Any] | None
    _state: str
    handoff_count: int
    dspark_committed_context: Any | None

    @classmethod
    def from_committed(cls, setup: PrefillExecutionSetup, *, prefix_token_ids: Sequence[int]) -> 'LivePrefillResult':
        actual_setup = setup.final_setup if hasattr(setup, "final_setup") and hasattr(setup, "sealed") else setup
        if actual_setup.handoff_claimed or actual_setup.block_runner.handoff_transferred:
            raise LiveCacheHandoffError('live cache handoff already claimed')
        # Copy request-owned Python token metadata only; never cache tensors.
        ids = tuple(int(t) for t in prefix_token_ids)
        frontier = validate_committed_cache(setup, ids)
        result = cls()
        result.prefix_token_ids, result.frontier = ids, frontier
        if setup is not actual_setup:
            setattr(actual_setup, "p6_commit_authority", setup)
        result._setup, result._live_cache = actual_setup, actual_setup.block_runner.working_cache
        result._state, result.handoff_count = 'ready', 0
        result.dspark_committed_context = getattr(setup, "dspark_committed_context", None)
        capture = getattr(actual_setup.block_runner, 'tap_capture', None)
        if capture is not None:
            result.dspark_committed_context = capture.commit(result, frontier)
        actual_setup.handoff_claimed = True
        actual_setup.block_runner.handoff_reserved = True
        if getattr(actual_setup.block_runner, "scheduling_coordinator", None) is not None:
            actual_setup.block_runner.scheduling_coordinator.revoke()
        return result

    def discard(self) -> None:
        """Burn an untransferred ready lease after protected prefill cancellation."""
        if self._state != 'ready':
            raise LiveCacheHandoffError('only a ready prefill lease can be discarded')
        setup, cache = self._setup, self._live_cache
        from ds41f_mlx.runtime.hidden_taps import CommittedTapReceipt
        if isinstance(self.dspark_committed_context, CommittedTapReceipt):
            self.dspark_committed_context.retire()
            self.dspark_committed_context = None
        self._state, self._live_cache = 'failed', None
        try:
            for item in cache or ():
                item._p6_append_failed = True
                item._p6_append_invalid = True
                item._p6_append_sealed = False
            setup.block_runner.handoff_transferred = True
            setup.continuation = None
            setup.block_runner.close()
        finally:
            setup.__dict__.pop('p6_commit_authority', None)

    @property
    def live_cache(self) -> list[Any]:
        if self._state != 'ready' or self._live_cache is None:
            raise LiveCacheHandoffError('live cache authority has been transferred or invalidated')
        return self._live_cache

    @property
    def cache_authority_owner(self) -> str:
        return {'ready': 'LivePrefillResult', 'transferring': getattr(self, '_decode_authority_label', 'OMLXGenerationSession'),
                'started': getattr(self, '_decode_authority_label', 'BatchGenerator/GenerationBatch'), 'failed': 'invalidated'}[self._state]


def _generation_session_type():
    # Avoid loading the older compatibility admission path at package import.
    from ds41f_mlx.runtime.target_generation import TargetGenerationSession
    return TargetGenerationSession


def handoff_to_generation(result: LivePrefillResult, model: Any, *, terminal_prompt_token: int,
                          config: Any = None, max_tokens: int = 128, sampler: Any = None,
                          session_factory: Any = None) -> 'TargetGenerationSession':
    """Transfer once and bootstrap only the held-out terminal prompt token.

    Admission and start are one operation. Even a failed transfer/start attempt
    cannot be retried with this result: the potentially mutated cache is burned.
    No attempt is made to infer complete token history from the last sweep.
    """
    cache = result.live_cache
    setup = result._setup
    if getattr(model, 'language_model', model) is not setup.block_runner.language_model:
        raise LiveCacheHandoffError('live cache belongs to a different loaded model')
    validate_committed_cache(setup, result.prefix_token_ids)
    terminal = int(terminal_prompt_token)
    if terminal < 0 or terminal >= setup.block_runner.language_model._config.vocab_size or max_tokens < 1:
        raise LiveCacheHandoffError('invalid terminal token or decode bound')
    result._state = 'transferring'
    result.handoff_count = 1
    session = None
    try:
        # Internal qualification may explicitly supply a guarded native MTP
        # session factory. Public/default serving still selects MTP-OFF only.
        session_type = session_factory or _generation_session_type()
        result._decode_authority_label = getattr(session_type, 'cache_authority_label', 'BatchGenerator/GenerationBatch')
        session = session_type.from_prefilled_cache(
            model, cache, result.prefix_token_ids, config, max_tokens=max_tokens, sampler=sampler)
        if session.initial_cache is not cache:
            raise LiveCacheHandoffError('generation admission replaced the live cache list')
        if tuple(session.prefix_tokens) != result.prefix_token_ids:
            raise LiveCacheHandoffError('generation prefix seed differs from request-owned history')
        # Revoke the old executable authority before the scheduler can mutate it.
        setup.block_runner.working_cache = None
        setup.block_runner.handoff_transferred = True
        if getattr(setup.block_runner, "scheduling_coordinator", None) is not None:
            setup.block_runner.scheduling_coordinator.revoke()
        setup.continuation = None
        result._live_cache = None
        seed = result.dspark_committed_context
        from ds41f_mlx.runtime.hidden_taps import CommittedTapReceipt
        if isinstance(seed, CommittedTapReceipt):
            if seed.owner is not result or seed.end != result.frontier:
                raise LiveCacheHandoffError('prefill tap receipt binding mismatch')
            seed.owner = session
            session._prefill_tap_receipt = seed
            result.dspark_committed_context = None
        session.start(terminal)
        if session.prompt_replay_count != 0:
            raise LiveCacheHandoffError('prefix prompt replay during terminal bootstrap')
        offsets = session.active_cache_offsets()
        if offsets is None or len(offsets) != 40 or any(x != result.frontier + 1 for x in offsets):
            raise LiveCacheHandoffError('scheduler cache frontiers did not advance by one terminal token')
        if session.initial_cache:
            raise LiveCacheHandoffError('admitted singleton remains an active authority after bootstrap')
        result._state = 'started'
        return session
    except Exception:
        result._state = 'failed'
        result._live_cache = None
        setup.block_runner.working_cache = None
        setup.block_runner.handoff_transferred = True
        setup.continuation = None
        if session is not None:
            session.close()
        raise
    finally:
        # P6 commit.final_setup points back here. After this one-shot transfer
        # (or burn), the admission backlink is no longer executable authority.
        # Break only that passive cycle: keep the live cache and all revocation
        # guards intact, without waiting for cyclic GC to free old publications.
        setup.__dict__.pop('p6_commit_authority', None)
