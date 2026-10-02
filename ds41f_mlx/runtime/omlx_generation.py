"""Practical oMLX GenerationBatch production decode session.

This module promotes the Milestone-4 P5 no-prompt-replay mechanism into a
runtime-facing session while keeping :class:`OMLXDecodeSession.decode_one` as the
slow diagnostic direct-`_forward` path.

The production path is intentionally the pinned oMLX/mlx-lm lifecycle:

    PrefillContinuationState
      -> OMLXDecodeStateAdapter
      -> request-local DeepseekV41Cache
      -> BatchGenerator.insert(caches=..., all_tokens=...)
      -> GenerationBatch target decode

The new P5 prefill package enters directly through ``from_prefilled_cache``:
its committed cache list is already authoritative, so the compatibility
state/adapter admission shown above is not used. The terminal prompt token is
held out of prefill and supplied once to ``start()``.

For an admitted prefix ``[0, 3]`` at frontier 2, ``start(first_input_token=15)``
passes ``prompts=[[15]]`` and ``all_tokens=[[0, 3]]``.  The bootstrap
``BatchGenerator.next()`` constructs a ``GenerationBatch`` and its constructor
immediately forwards token 15 against the admitted cache.  The first emitted
GenerationBatch response on the following ``next_token()`` call is therefore the
sampled model distribution after consuming token 15.  No token from ``[0, 3]`` is
re-forwarded through the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable
import importlib
import sys

import numpy as np

from ds41f_mlx.prefill_session import PrefillContinuationState
from ds41f_mlx.runtime.omlx_core import DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXAdmissionReport, OMLXDecodeConfig, OMLXDecodeStateAdapter


@dataclass(frozen=True)
class OMLXGenerationStepReport:
    token: int
    frontier_before: int
    frontier_after: int
    latency_s: float
    finish_reason: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "token": int(self.token),
            "frontier_before": int(self.frontier_before),
            "frontier_after": int(self.frontier_after),
            "latency_s": float(self.latency_s),
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class OMLXGenerationMetadata:
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

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class OMLXGenerationSession:
    """Practical single-stream MTP-OFF decode session using BatchGenerator.

    The only live target-cache authority after ``start()`` is the oMLX scheduler
    cache held by ``BatchGenerator``/``GenerationBatch``.  The admitted singleton
    cache is supplied once to ``BatchGenerator.insert``; after the bootstrap, the
    scheduler advances cache offsets, owns generated token history, and returns
    the final request-local cache through ``extract_cache``/finish responses.
    """

    model: Any
    initial_cache: list[Any]
    initial_token_ids: np.ndarray
    config: OMLXDecodeConfig = field(default_factory=OMLXDecodeConfig)
    admission_report: OMLXAdmissionReport | None = None
    sampler: Callable[[Any], Any] | None = None
    max_tokens: int = 128
    stream: Any | None = None

    def __post_init__(self) -> None:
        if self.config.speculation_enabled or self.config.preserve_mtp not in (False, None):
            raise RuntimeError("OMLXGenerationSession is base MTP-OFF decode only")
        root = str(self.config.omlx_path or DEFAULT_OMLX)
        if root not in sys.path:
            sys.path.insert(0, root)
        self.mx = importlib.import_module("mlx.core")
        gen = importlib.import_module("mlx_lm.generate")
        # Load oMLX scheduler monkey patches without starting the HTTP engine.
        try:
            importlib.import_module("omlx.scheduler")
        except Exception:
            pass
        self.BatchGenerator = gen.BatchGenerator
        self.generation_stream = gen.generation_stream
        self.stream = self.stream or self.generation_stream
        self.language_model = getattr(self.model, "language_model", self.model)
        if hasattr(self.language_model, "configure_mtp"):
            self.language_model.configure_mtp(False, 1)
        self.sampler = self.sampler or (lambda logits: self.mx.argmax(logits, axis=-1))
        ids = np.asarray(self.initial_token_ids, dtype=np.int64).reshape(1, -1)
        self.prefix_tokens = [int(x) for x in ids[0].tolist()]
        self.admitted_frontier = len(self.prefix_tokens)
        self.token_frontier = self.admitted_frontier
        self.generated_tokens: list[int] = []
        self.step_reports: list[OMLXGenerationStepReport] = []
        self.prompt_bootstrap_responses: list[dict[str, Any]] = []
        self.uid: int | None = None
        self.first_input_token: int | None = None
        self.prompt_replay_count = 0
        self._inserted = False
        self._start_attempted = False
        self.bootstrap_inserted_prompt: tuple[int, ...] = ()
        self._started = False
        self._stopped = False
        self.stop_reason: str | None = None
        self._final_cache: list[Any] | None = None
        self._final_all_tokens: list[int] | None = None
        stop_token_ids = tuple(int(t) for t in getattr(self.config, "stop_token_ids", ()) or ())
        self.stop_token_ids = stop_token_ids
        self._bg = self.BatchGenerator(
            self.language_model,
            max_tokens=self.max_tokens,
            stop_tokens=[[t] for t in stop_token_ids] if stop_token_ids else None,
            sampler=self.sampler,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=2048,
            stream=self.stream,
        )

    @classmethod
    def from_prefill_state(
        cls,
        model: Any,
        state: PrefillContinuationState,
        config: OMLXDecodeConfig | None = None,
        *,
        max_tokens: int = 128,
        sampler: Callable[[Any], Any] | None = None,
    ) -> "OMLXGenerationSession":
        cfg = config or OMLXDecodeConfig()
        adapter = OMLXDecodeStateAdapter(model=model, omlx_path=cfg.omlx_path)
        cache, report = adapter.admit(state)
        return cls(model, cache, state.token_ids, cfg, report, sampler, max_tokens)

    @classmethod
    def from_prefilled_cache(
        cls,
        model: Any,
        cache: list[Any],
        token_ids: Any,
        config: OMLXDecodeConfig | None = None,
        *,
        max_tokens: int = 128,
        sampler: Callable[[Any], Any] | None = None,
    ) -> "OMLXGenerationSession":
        """Create a GenerationBatch session from a live same-backend cache.

        This is the production zero-repack handoff.  The cache must already be a
        real request-local DeepseekV41Cache list populated by the loaded oMLX
        LanguageModel.  Unlike ``from_prefill_state`` this does not export to or
        re-admit from ``PrefillContinuationState``.
        """
        cfg = config or OMLXDecodeConfig()
        for item in cache:
            if (getattr(item, "_p6_append_invalid", False)
                or getattr(item, "_p6_append_failed", False)
                or getattr(item, "_p6_append_pending", False)
                or (hasattr(item, "_p6_append_sealed") and not getattr(item, "_p6_append_sealed", False))):
                raise RuntimeError("P6 cache is not sealed/admissible for generation")
        ids = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
        frontier = ids.shape[1]
        offsets = cls.cache_offsets(cache)
        if any(offset != frontier for offset in offsets):
            raise RuntimeError(f"prefilled live cache offsets {offsets[:4]} do not match frontier {frontier}")
        return cls(model, cache, ids, cfg, None, sampler, max_tokens)

    @classmethod
    def load_model_and_admit(
        cls,
        state: PrefillContinuationState,
        config: OMLXDecodeConfig,
        *,
        max_tokens: int = 128,
        sampler: Callable[[Any], Any] | None = None,
    ) -> "OMLXGenerationSession":
        rt_cfg = OmlxRuntimeConfig(
            omlx_path=config.omlx_path,
            checkpoint_path=config.checkpoint_path or OmlxRuntimeConfig().checkpoint_path,
            engram_ssd_offload=config.engram_ssd_offload,
            preserve_mtp=False,
            moe_expert_offload_resident_fraction=config.moe_expert_offload_resident_fraction,
        )
        runtime = OmlxRuntime(rt_cfg)
        model, _processor = runtime.load_model()
        session = cls.from_prefill_state(model, state, config, max_tokens=max_tokens, sampler=sampler)
        session._runtime = runtime
        return session

    def start(self, first_input_token: int, *, max_tokens: int | None = None) -> None:
        """Attach admitted cache to BatchGenerator without replaying prefix tokens."""
        if self._start_attempted or self._inserted:
            raise RuntimeError("generation session start already attempted")
        # A failed bootstrap may have mutated scheduler state; never retry it.
        self._start_attempted = True
        if self.admission_report is not None and self.admission_report.token_frontier != self.admitted_frontier:
            raise RuntimeError("admission frontier/token history mismatch")
        before = self.cache_offsets(self.initial_cache)
        if any(offset != self.admitted_frontier for offset in before):
            raise RuntimeError(f"admitted cache offsets {before[:4]} do not match frontier {self.admitted_frontier}")
        self.first_input_token = int(first_input_token)
        self.max_tokens = int(max_tokens or self.max_tokens)
        self.bootstrap_inserted_prompt = (self.first_input_token,)
        uids = self._bg.insert(
            [[self.first_input_token]],
            max_tokens=[self.max_tokens],
            caches=[self.initial_cache],
            all_tokens=[list(self.prefix_tokens)],
            samplers=[self.sampler],
        )
        self.mx.synchronize(self.stream)
        self.uid = int(uids[0])
        self._inserted = True
        # Bootstrap: moves the one-token prompt [first_input_token] into a
        # GenerationBatch and forwards exactly that token in GenerationBatch.__init__.
        prompt_responses, generation_responses = self._bg.next()
        self.mx.synchronize(self.stream)
        self.prompt_bootstrap_responses = [self._response_json(r) for r in prompt_responses]
        if generation_responses:
            raise RuntimeError("unexpected generation response during one-token bootstrap")
        self.prompt_replay_count = self._count_replayed_prefix_tokens(prompt_responses)
        if self.prompt_replay_count != 0:
            raise RuntimeError(f"prompt replay detected: {self.prompt_replay_count} prefix tokens")
        offsets = self.active_cache_offsets()
        expected = self.admitted_frontier + 1
        if offsets is None or len(offsets) != len(before) or any(offset != expected for offset in offsets):
            raise RuntimeError(f"terminal bootstrap scheduler frontiers {offsets} != {expected}")
        self.token_frontier = expected
        self._started = True
        # The original admitted singleton cache has been handed off; scheduler cache is authority now.
        self.initial_cache = []

    def next_token(self) -> OMLXGenerationStepReport | None:
        if not self._started:
            raise RuntimeError("call start(first_input_token) before next_token")
        if self._stopped:
            return None
        frontier_before = self.token_frontier
        t0 = perf_counter()
        prompt_responses, generation_responses = self._bg.next()
        self.mx.synchronize(self.stream)
        latency = perf_counter() - t0
        if prompt_responses:
            extra = self._count_replayed_prefix_tokens(prompt_responses)
            self.prompt_replay_count += extra
            if extra:
                raise RuntimeError(f"prompt replay detected after bootstrap: {extra}")
        if not generation_responses:
            return None
        response = generation_responses[0]
        token = int(response.token)
        self.generated_tokens.append(token)
        self.token_frontier += 1
        finish_reason = response.finish_reason
        report = OMLXGenerationStepReport(token, frontier_before, self.token_frontier, latency, finish_reason)
        self.step_reports.append(report)
        if finish_reason is not None:
            self._stopped = True
            self.stop_reason = finish_reason
            if response.prompt_cache is not None:
                self._final_cache = response.prompt_cache
                backend_tokens = getattr(response, "all_tokens", None)
                self._final_all_tokens = (
                    [int(t) for t in backend_tokens]
                    if backend_tokens is not None else self.current_token_history()
                )
            elif self.uid is not None:
                # M11 evidence showed that waiting until request cleanup after a
                # natural stop can lose the scheduler-owned cache for a finished
                # request.  Extract immediately at the same consumed-token
                # boundary, before later cleanup/HTTP completion can remove it.
                try:
                    extracted = self._bg.extract_cache([self.uid])
                    if self.uid in extracted:
                        self._final_cache, all_tokens = extracted[self.uid]
                        self._final_all_tokens = [int(t) for t in all_tokens]
                        self.token_frontier = len(self._final_all_tokens)
                except Exception:
                    self._final_cache = None
                    self._final_all_tokens = None
        return report

    def generate(self, count: int) -> list[OMLXGenerationStepReport]:
        reports: list[OMLXGenerationStepReport] = []
        for _ in range(count):
            report = self.next_token()
            if report is None:
                break
            reports.append(report)
            if report.finish_reason is not None:
                break
        return reports

    def stop(self, reason: str = "cancelled") -> None:
        """Best-effort cancellation or turn-boundary cache extraction."""
        if self._stopped and self._final_cache is not None and self._final_all_tokens is not None:
            return
        self._stopped = True
        self.stop_reason = self.stop_reason or reason
        if self.uid is not None:
            try:
                extracted = self._bg.extract_cache([self.uid])
                if self.uid in extracted:
                    self._final_cache, all_tokens = extracted[self.uid]
                    self._final_all_tokens = [int(t) for t in all_tokens]
                    self.token_frontier = len(self._final_all_tokens)
            except Exception:
                self._final_cache = None
                self._final_all_tokens = None
            finally:
                # extract_cache observes state; it does not relinquish scheduler
                # ownership. The extracted row is the continuation authority.
                self._bg.remove([self.uid])

    cancel = stop

    def extract_final_state(self, reason: str = "turn_boundary") -> tuple[list[Any], list[int]]:
        """Stop if needed and return the scheduler-owned cache/history for continuation.

        This is the M8 live-session seam: it observes the GenerationBatch state
        through the pinned ``extract_cache`` API and does not rebuild or repack
        cache tensors.  The returned cache is the only executable authority for
        a subsequent append cycle.
        """
        if self._final_cache is None or self._final_all_tokens is None:
            self.stop(reason)
        if self._final_cache is None or self._final_all_tokens is None:
            # A length/stop response may already have removed the request from
            # the scheduler but left the active batch cache observable.
            cache = self._generation_cache()
            if cache is not None:
                self._final_cache = cache
                self._final_all_tokens = self.current_token_history()
        if self._final_cache is None or self._final_all_tokens is None:
            raise RuntimeError("GenerationBatch final cache/history is unavailable for continuation")
        offsets = self.cache_offsets(self._final_cache)
        if any(offset != len(self._final_all_tokens) for offset in offsets):
            raise RuntimeError(f"final cache offsets {offsets[:4]} do not match token history {len(self._final_all_tokens)}")
        return self._final_cache, list(self._final_all_tokens)

    def close(self) -> None:
        self.stop("closed")
        bg = getattr(self, "_bg", None)
        if bg is not None:
            try:
                # A failed insert/bootstrap may not have assigned self.uid.
                # This generator is request-local: drain every possible stage.
                uids = set(getattr(getattr(bg, "_generation_batch", None), "uids", ()))
                uids.update(getattr(getattr(bg, "_prompt_batch", None), "uids", ()))
                uids.update(seq[0] for seq in getattr(bg, "_unprocessed_sequences", ()))
                if uids:
                    bg.remove(list(uids))
            finally:
                bg.close()
        self.initial_cache = []
        self._final_cache = None
        self._final_all_tokens = None
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.close()

    def metadata(self) -> OMLXGenerationMetadata:
        return OMLXGenerationMetadata(
            schema="ds41f.m5.omlx-generation-session.metadata.v1",
            execution_substrate="mlx_lm.generate.BatchGenerator -> GenerationBatch (explicit configured oMLX dependency)",
            cache_authority_owner="BatchGenerator/GenerationBatch scheduler cache after start(); final cache via finish/extract_cache",
            admitted_frontier=self.admitted_frontier,
            prompt_replay_count=self.prompt_replay_count,
            first_input_token=self.first_input_token,
            token_frontier=self.token_frontier,
            generated_count=len(self.generated_tokens),
            stopped=self._stopped,
            stop_reason=self.stop_reason,
        )

    def timing_summary(self) -> dict[str, Any]:
        lat = [r.latency_s for r in self.step_reports]
        return {
            "steps": len(lat),
            "first_decode_latency_s": lat[0] if lat else None,
            "median_latency_s": median(lat) if lat else None,
            "mean_latency_s": mean(lat) if lat else None,
            "median_tok_s": (1.0 / median(lat)) if lat else None,
            "latencies_s": lat,
        }

    def reset(self) -> None:
        raise NotImplementedError("GenerationBatch reset is deferred; create a fresh session from PrefillContinuationState")

    def fork(self) -> "OMLXGenerationSession":
        raise NotImplementedError("GenerationBatch fork seam is deferred; fork from PrefillContinuationState before start()")

    def current_token_history(self) -> list[int]:
        tokens = list(self.prefix_tokens)
        if self.first_input_token is not None and self.token_frontier >= self.admitted_frontier + 1:
            tokens.append(int(self.first_input_token))
        tokens.extend(int(t) for t in self.generated_tokens)
        return tokens

    def active_cache_offsets(self) -> tuple[int, ...] | None:
        cache = self._final_cache or self._generation_cache()
        return None if cache is None else self.cache_offsets(cache)

    @staticmethod
    def cache_offsets(cache: list[Any]) -> tuple[int, ...]:
        return tuple(int(item.size()) for item in cache)

    def _generation_cache(self) -> list[Any] | None:
        gb = getattr(self._bg, "_generation_batch", None)
        cache = getattr(gb, "prompt_cache", None)
        return cache if cache else None

    @staticmethod
    def _response_json(response: Any) -> dict[str, Any]:
        data = dict(getattr(response, "__dict__", {}))
        for key, value in list(data.items()):
            if isinstance(value, np.integer):
                data[key] = int(value)
        return data

    @staticmethod
    def _count_replayed_prefix_tokens(prompt_responses: list[Any]) -> int:
        # PromptProcessingBatch.Response.progress is (processed, total).  In the
        # no-replay P5 bootstrap, the only prompt segment is [first_input_token]
        # and split-to-generation reports end_of_prompt=True without forwarding
        # any prefix token.  Non-generation prompt work would appear as a
        # non-end response with positive progress.
        replayed = 0
        for response in prompt_responses:
            if getattr(response, "end_of_prompt", False):
                continue
            progress = getattr(response, "progress", 0)
            if isinstance(progress, tuple):
                replayed += int(progress[0])
            else:
                replayed += int(progress or 0)
        return replayed
