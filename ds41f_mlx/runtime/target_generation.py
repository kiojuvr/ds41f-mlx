"""ds41f single-flight target generation authority (OFF and explicit prefix cycles).

Owns scheduling, sampling, history and the live cache lease. Model arithmetic,
packed cache objects and SSD Engram remain the attributed temporary substrate.
A sampled lookahead is never part of committed history until consumed by the
next target call. No scheduler, row extraction, replay or cache conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable

import numpy as np

from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.target_forward import TargetForwardTransaction


@dataclass(frozen=True)
class TargetGenerationStepReport:
    token: int
    frontier_before: int
    frontier_after: int
    latency_s: float
    finish_reason: str | None

    def to_json(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class TargetGenerationMetadata:
    schema: str
    execution_substrate: str
    cache_authority_owner: str
    admitted_frontier: int
    prompt_replay_count: int
    first_input_token: int | None
    token_frontier: int
    generated_count: int
    stopped: bool
    stop_reason: str | None

    def to_json(self):
        return dict(self.__dict__)


@dataclass
class TargetGenerationSession:
    model: Any
    initial_cache: list[Any]
    initial_token_ids: Any
    config: OMLXDecodeConfig = field(default_factory=OMLXDecodeConfig)
    sampler: Callable | None = None
    max_tokens: int = 128
    stream: Any = None

    cache_authority_label = 'ds41f TargetGenerationSession'

    def __post_init__(self):
        if self.config.speculation_enabled or self.config.preserve_mtp not in (False, None):
            raise RuntimeError('target generation requires MTP/DSpark/speculation OFF')
        if self.max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        self.mx = importlib.import_module('mlx.core')
        self.stream = self.stream or self.mx.new_thread_local_stream(self.mx.default_device())
        self.language_model = getattr(self.model, 'language_model', self.model)
        if hasattr(self.language_model, 'configure_mtp'):
            self.language_model.configure_mtp(False, 1)
        self.target_forward = TargetForwardTransaction(self.language_model, self.mx)
        self.sampler = self.sampler or (lambda p: self.mx.argmax(p, axis=-1))
        self.prefix_tokens = [int(t) for t in np.asarray(self.initial_token_ids).reshape(-1)]
        self.admitted_frontier = len(self.prefix_tokens)
        self.token_frontier = self.admitted_frontier
        self.first_input_token = None
        self.bootstrap_inserted_prompt = ()
        self.prompt_bootstrap_responses = []
        self.prompt_replay_count = 0
        self.generated_tokens = []
        self.step_reports = []
        self.stop_token_ids = tuple(self.config.stop_token_ids)
        self.stop_reason = None
        self._start_attempted = self._started = self._stopped = False
        self._failed = False
        self._cycle_active = False
        self._generation_publication_failed = False
        self._cache = None
        self._pending = None
        self._final_cache = self._final_all_tokens = None
        self._history = list(self.prefix_tokens)
        # Single-flight lease: match the qualified device working-set policy,
        # but own restoration instead of relying on BatchGenerator.__del__.
        recommended = self.mx.device_info().get('max_recommended_working_set_size')
        self._old_wired_limit = None if recommended is None else self.mx.set_wired_limit(recommended)

    @classmethod
    def from_prefilled_cache(cls, model, cache, token_ids, config=None, *, max_tokens=128, sampler=None):
        cfg = config or OMLXDecodeConfig()
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if not cache or not len(ids):
            raise RuntimeError('nonempty committed cache/history required')
        for item in cache:
            if (getattr(item, '_p6_append_invalid', False)
                or getattr(item, '_p6_append_failed', False)
                or getattr(item, '_p6_append_pending', False)
                or (hasattr(item, '_p6_append_sealed') and not item._p6_append_sealed)):
                raise RuntimeError('P6 cache is not sealed/admissible for generation')
        if any(n != len(ids) for n in cls.cache_offsets(cache)):
            raise RuntimeError('prefilled cache frontier/history mismatch')
        return cls(model, cache, ids, cfg, sampler, max_tokens)

    def _consume(self, token):
        self._require_unborrowed()
        # Publish history only after the all-layer mutation transaction commits.
        consumed, pending = self.target_forward.execute(
            token, self._cache, self.token_frontier, self.sampler, self.stream)
        self._pending = pending
        self._history.append(consumed)
        self.token_frontier = len(self._history)
        self._publish_taps(self.token_frontier - 1, getattr(self.target_forward, 'tap_rows', None))
        self.target_forward.tap_rows = None
        return consumed

    def disable_proposals(self):
        """Explicit derived-only discard at a coherent idle target boundary.

        Never called as an exception fallback inside a speculative cycle.
        Re-enabling requires a newly qualified full seed, not cache replay.
        """
        self._require_unborrowed()
        if self._failed or not self._started or self._stopped:
            raise RuntimeError('live coherent generation required')
        if set(self.active_cache_offsets()) != {len(self._history)}:
            raise RuntimeError('incoherent proposal disable boundary')
        self._retire_proposal()
        self.target_forward.proposal_child = None
        self.target_forward.tap_rows = None

    def take_tap_receipt(self):
        self._require_unborrowed()
        receipt = getattr(self, '_tap_receipt', None)
        if receipt is None or receipt.retired:
            raise RuntimeError('committed tap receipt unavailable')
        self._tap_receipt = None
        return receipt

    def _publish_taps(self, start, rows):
        old = getattr(self, '_tap_receipt', None)
        if old is not None:
            old.retire()
        self._tap_receipt = None
        if rows is not None:
            from ds41f_mlx.runtime.hidden_taps import CommittedTapReceipt
            self._tap_receipt = CommittedTapReceipt(
                self.target_forward.proposal_child, self, start, rows)

    def _retire_proposal(self):
        producer = getattr(self, '_proposal_producer', None)
        if producer is not None:
            producer.retire()
        for name in ('_tap_receipt', '_prefill_tap_receipt'):
            receipt = getattr(self, name, None)
            if receipt is not None:
                receipt.retire()
                setattr(self, name, None)

    def _invalidate(self):
        self._failed = True
        self._retire_proposal()
        for item in (self._cache or self._final_cache or ()):
            # Existing P6 admission/runner guards also reject passive stale
            # aliases after a potentially partially mutating target failure.
            item._p6_append_failed = True
            item._p6_append_invalid = True

    def start(self, first_input_token, *, max_tokens=None):
        if self._start_attempted:
            raise RuntimeError('generation session start already attempted')
        self._start_attempted = True
        if max_tokens is not None:
            if max_tokens < 1:
                raise ValueError('max_tokens must be positive')
            self.max_tokens = int(max_tokens)
        if any(n != self.admitted_frontier for n in self.cache_offsets(self.initial_cache)):
            raise RuntimeError('admitted cache frontier/history mismatch')
        self.first_input_token = int(first_input_token)
        self.bootstrap_inserted_prompt = (self.first_input_token,)
        self._cache, self.initial_cache = self.initial_cache, []
        try:
            self._consume(self.mx.array([self.first_input_token], self.mx.uint32))
            if any(n != self.token_frontier for n in self.cache_offsets(self._cache)):
                raise RuntimeError('terminal bootstrap frontier mismatch')
            self._started = True
        except BaseException:
            self._invalidate()
            raise

    def next_token(self):
        if self._failed:
            raise RuntimeError('failed generation authority is not reusable')
        if not self._started:
            raise RuntimeError('call start before next_token')
        if self._stopped:
            return None
        before, t0 = self.token_frontier, perf_counter()
        try:
            token = self._consume(self._pending)
        except BaseException:
            self._invalidate()
            raise
        self.generated_tokens.append(token)
        reason = 'length' if len(self.generated_tokens) >= self.max_tokens else None
        if token in self.stop_token_ids:
            reason = 'stop'
        report = TargetGenerationStepReport(token, before, self.token_frontier, perf_counter()-t0, reason)
        self.step_reports.append(report)
        if reason:
            self.stop(reason)
        return report

    def speculative_cycle(self, proposals, *, cancelled=None, _fault=None):
        """Verify external token proposals without granting their producer authority.

        Equality with sequential canonical target samples decides acceptance (not
        probability-ratio rejection sampling). Only causal, retained rows invoke
        the OFF sampler, once per consumed input, including terminal inputs.
        Cancellation rolls back to zero BEFORE sampling; after sampling begins it
        is deferred to the next coherent boundary. Exceptions always burn.
        Proposal producers must not advance this owner's sampler/RNG. The cancel
        predicate is side-effect-free. Public owner observation/reentry is blocked
        until the entire cycle returns. ``_fault`` is a qualification hook, not a
        response/publication callback.
        """
        self._require_unborrowed()
        if self._failed or not self._started:
            raise RuntimeError('live generation authority required')
        if self._stopped:
            return None
        drafts = tuple(int(t) for t in proposals)
        if len(drafts) >= 32 or any(t < 0 or t >= self.language_model._config.vocab_size
                                   for t in drafts):
            raise ValueError('at most 31 vocabulary token proposals required')
        cancel = cancelled or (lambda: False)
        fault = _fault or (lambda phase: None)
        if cancel():
            return dict(cancelled=True, consumed_positions=0, proposal_acceptance_count=0,
                        reports=[])
        before, t0 = self.token_frontier, perf_counter()
        anchor = int(self._pending.item())
        inputs = (anchor,) + drafts
        journal = None
        self._cycle_active = True
        try:
            fault('verify-before')
            journal = self.target_forward.begin_prefix_journal(
                self._cache, before, len(inputs), self.stream)
            for token in inputs:
                journal.advance(self.mx.array([token], self.mx.uint32))
                fault('tentative')
                if cancel():
                    journal.cancel()
                    return dict(cancelled=True, consumed_positions=0,
                                proposal_acceptance_count=0, reports=[])
            journal.complete()
            fault('materialized')
            if cancel():
                journal.cancel()
                return dict(cancelled=True, consumed_positions=0,
                            proposal_acceptance_count=0, reports=[])
            # From this point no recoverable cancellation: sampler/RNG state may
            # change. There is no sampling of a rejected input or rejected tail.
            accepted, consumed, reason, pending = 0, [], None, None
            with self.mx.stream(self.stream):
                for i, token in enumerate(inputs):
                    logits = journal.logits[i]
                    pending = self.sampler(logits - self.mx.logsumexp(
                        logits, axis=-1, keepdims=True))
                    self.mx.eval(pending)
                    if (pending.shape != (1,) or pending.dtype not in
                        (self.mx.uint32, self.mx.int32, self.mx.int64, self.mx.uint64)
                        or not 0 <= int(pending.item()) < self.language_model._config.vocab_size):
                        raise RuntimeError('canonical sampler must return one vocabulary token')
                    consumed.append(token)
                    reason = ('length' if len(self.generated_tokens) + len(consumed)
                              >= self.max_tokens else None)
                    if token in self.stop_token_ids:
                        reason = 'stop'
                    if reason or i == len(drafts) or int(pending.item()) != drafts[i]:
                        break
                    accepted += 1
            fault('acceptance')
            def publish_frontier(end):
                if end != before + len(consumed):
                    raise RuntimeError('generation settlement receipt mismatch')
                self.token_frontier = end
                fault('frontier')
            committed_taps = (self.mx.concatenate(journal.tap_rows[:len(consumed)], 1)
                              if getattr(journal, 'tap_rows', None) else None)
            journal.settle(len(consumed), publish=publish_frontier)
            fault('settled')
            # No user-visible reports/response until the target barrier succeeds.
            # A partial Python publication is unusable, never an OFF fallback.
            reports = [TargetGenerationStepReport(token, before + i, before + i + 1,
                       (perf_counter() - t0) / len(consumed),
                       reason if i == len(consumed) - 1 else None)
                       for i, token in enumerate(consumed)]
            self._pending = pending
            fault('lookahead')
            self._history.extend(consumed)
            fault('history')
            self.generated_tokens.extend(consumed)
            self.step_reports.extend(reports)
            stop_reason = reason or ('cancelled' if cancel() else None)
            if stop_reason:
                # stop's exact-list retirement is internal to this cycle. Keep
                # observers/reentrancy excluded through response construction.
                self._cycle_active = False
                try:
                    self.stop(stop_reason)
                finally:
                    self._cycle_active = True
            fault('terminal')
            fault('response')
            if not self._stopped:
                self._publish_taps(before, committed_taps)
            return dict(cancelled=self.stop_reason == 'cancelled', confirmed_anchor=anchor,
                        proposal_acceptance_count=accepted, consumed_positions=len(consumed),
                        next_lookahead=None if self._pending is None else int(self._pending.item()),
                        reports=reports)
        except BaseException:
            if journal is not None and journal.phase not in ('retired', 'burned'):
                journal.burn()
            self._generation_publication_failed = True
            self._invalidate()
            raise
        finally:
            self._cycle_active = False

    def generate(self, count):
        result = []
        for _ in range(count):
            report = self.next_token()
            if report is None:
                break
            result.append(report)
            if report.finish_reason:
                break
        return result

    def _require_unborrowed(self):
        if self._cycle_active:
            raise RuntimeError('generation cycle is single-flight')
        if any(getattr(item, '_p6_append_invalid', False)
               or getattr(item, '_p6_append_failed', False) for item in self._cache or ()):
            self._failed = True
        if any(getattr(item, '_accepted_prefix_journal', None) is not None
               for item in self._cache or ()):
            raise RuntimeError('generation publication/idle transfer during target borrow')

    def stop(self, reason='cancelled'):
        self._require_unborrowed()
        if self._stopped:
            return
        self._stopped = True
        self._retire_proposal()
        self.stop_reason = reason
        try:
            self.mx.synchronize(self.stream)
        except BaseException:
            self._invalidate()
            raise
        self._pending = None  # sampled but unconsumed: not history/cache state
        if self._started and not self._failed:
            if any(n != len(self._history) for n in self.cache_offsets(self._cache)):
                self._invalidate()
                raise RuntimeError('idle transfer cache frontier/history mismatch')
            # P5 already revoked the old runner/setup. Legacy row extraction
            # incidentally discarded these Python capability markers. Retire
            # them explicitly on the SAME list at the coherent idle boundary;
            # an old P6 certificate still fails its handoff_reserved/transferred
            # guard. This is lease retirement, not cache reconstruction.
            for item in self._cache:
                for name in ('_p6_append_sealed', '_p6_append_invalid',
                             '_p6_append_pending', '_p6_append_failed', '_p6_owner_token'):
                    if hasattr(item, name):
                        delattr(item, name)
            self._final_cache, self._cache = self._cache, None
            self._final_all_tokens = list(self._history)

    cancel = stop

    def extract_final_state(self, reason='turn_boundary'):
        if self._failed:
            raise RuntimeError('failed generation cannot yield a committed continuation')
        self.stop(reason)
        if self._final_cache is None:
            raise RuntimeError('committed generation cache unavailable')
        if any(n != len(self._final_all_tokens) for n in self.cache_offsets(self._final_cache)):
            raise RuntimeError('final cache frontier/history mismatch')
        return self._final_cache, list(self._final_all_tokens)

    def close(self):
        if self._cycle_active:
            raise RuntimeError('generation cycle is single-flight')
        try:
            children = {getattr(item, '_accepted_prefix_journal', None)
                        for item in self._cache or ()} - {None}
            if children:
                # Owner destruction cannot abandon a borrowed, executable list.
                self._invalidate()
                for child in children:
                    child.burn()
            self.stop('closed')
        finally:
            self._retire_proposal()
            self.initial_cache = []
            self._cache = self._pending = self._final_cache = self._final_all_tokens = None
            if self._old_wired_limit is not None:
                self.mx.synchronize(self.stream)
                self.mx.set_wired_limit(self._old_wired_limit)
                self._old_wired_limit = None

    def current_token_history(self):
        if self._generation_publication_failed or self._cycle_active:
            raise RuntimeError('generation history is not publishable')
        return list(self._history)

    @staticmethod
    def cache_offsets(cache):
        return tuple(int(c.size()) for c in cache)

    def active_cache_offsets(self):
        if self._cycle_active or self._generation_publication_failed:
            raise RuntimeError('generation state is not publishable')
        cache = self._final_cache if self._final_cache is not None else self._cache
        return None if cache is None else self.cache_offsets(cache)

    def metadata(self):
        if self._cycle_active or self._generation_publication_failed:
            raise RuntimeError('generation state is not publishable')
        return TargetGenerationMetadata(
            'ds41f.target-generation.metadata.v1',
            'ds41f single-stream target/all-layer transaction; attributed block/cache/Engram primitives',
            self.cache_authority_label, self.admitted_frontier, 0, self.first_input_token,
            self.token_frontier, len(self.generated_tokens), self._stopped, self.stop_reason)

    def timing_summary(self):
        lat = [r.latency_s for r in self.step_reports]
        return dict(steps=len(lat), first_decode_latency_s=lat[0] if lat else None,
                    median_latency_s=median(lat) if lat else None,
                    mean_latency_s=mean(lat) if lat else None,
                    median_tok_s=1/median(lat) if lat else None, latencies_s=lat)
