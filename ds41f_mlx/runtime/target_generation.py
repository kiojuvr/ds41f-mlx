"""ds41f single-flight target generation authority (standard-off).

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
        # This is the only execution entry. Preserve the qualified normalized
        # logprob sampler input and lazy async graph topology. Cache mutations
        # and history commit are on the same consumed-token boundary.
        with self.mx.stream(self.stream):
            logits = self.language_model(token[:, None], cache=self._cache)[:, -1, :]
            logprobs = logits - self.mx.logsumexp(logits, axis=-1, keepdims=True)
            pending = self.sampler(logprobs)
            self.mx.async_eval(pending, logprobs)
            self.mx.eval(token)
            consumed = int(token.item())
        self.mx.synchronize(self.stream)
        self._pending = pending
        self._history.append(consumed)
        self.token_frontier = len(self._history)
        return consumed

    def _invalidate(self):
        self._failed = True
        for item in self._cache or ():
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

    def stop(self, reason='cancelled'):
        if self._stopped:
            return
        self._stopped = True
        self.stop_reason = reason
        self.mx.synchronize(self.stream)
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
        try:
            self.stop('closed')
        finally:
            self.initial_cache = []
            self._cache = self._pending = self._final_cache = self._final_all_tokens = None
            if self._old_wired_limit is not None:
                self.mx.synchronize(self.stream)
                self.mx.set_wired_limit(self._old_wired_limit)
                self._old_wired_limit = None

    def current_token_history(self):
        return list(self._history)

    @staticmethod
    def cache_offsets(cache):
        return tuple(int(c.size()) for c in cache)

    def active_cache_offsets(self):
        cache = self._final_cache if self._final_cache is not None else self._cache
        return None if cache is None else self.cache_offsets(cache)

    def metadata(self):
        return TargetGenerationMetadata(
            'ds41f.target-generation.metadata.v1',
            'ds41f single-stream MLX target engine; temporary attributed model/cache kernels',
            self.cache_authority_label, self.admitted_frontier, 0, self.first_input_token,
            self.token_frontier, len(self.generated_tokens), self._stopped, self.stop_reason)

    def timing_summary(self):
        lat = [r.latency_s for r in self.step_reports]
        return dict(steps=len(lat), first_decode_latency_s=lat[0] if lat else None,
                    median_latency_s=median(lat) if lat else None,
                    mean_latency_s=mean(lat) if lat else None,
                    median_tok_s=1/median(lat) if lat else None, latencies_s=lat)
