"""Minimum command-executor setup for DwarfStar FP8/MLX prefill.

This module prepares request-owned state for P3 without selecting production
serving or implementing P5 decode handoff.  Transformer execution order remains
exclusively `SweepPlan.commands` consumed by `OfficialFP8MLXBlockRunner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import importlib

from ds41f_mlx.prefill_fp8_mlx.arena import RequestArena
from ds41f_mlx.prefill_fp8_mlx.block_runner import OfficialFP8MLXBlockRunner
from ds41f_mlx.prefill_fp8_mlx.planner import SweepPlan
from ds41f_mlx.prefill_fp8_mlx.publications import PublicationManager, PublicationTopology
from ds41f_mlx.prefill_fp8_mlx.p8_optimizer import P8ExecutionOptimizer


class PrefillSetupError(RuntimeError):
    """Invalid production prefill setup."""


@dataclass(frozen=True)
class LivePrefillContinuation:
    """Live request-local cache authority for multi-sweep continuation."""

    live_cache: list[Any]

    @classmethod
    def from_cache(cls, live_cache: list[Any]) -> "LivePrefillContinuation":
        obj = cls(live_cache=live_cache)
        _ = obj.frontier
        return obj

    @property
    def frontier(self) -> int:
        if not self.live_cache or len(self.live_cache) != 40:
            raise PrefillSetupError("live continuation requires 40 layer caches")
        if any(getattr(c, "_p6_append_invalid", False) or getattr(c, "_p6_append_failed", False) for c in self.live_cache):
            raise PrefillSetupError("live continuation cache is owned by an invalid/failed P6 append")
        frontiers = tuple(_cache_frontier(c) for c in self.live_cache)
        if any(v != frontiers[0] for v in frontiers):
            raise PrefillSetupError(f"live continuation cache frontiers diverge: {frontiers[:8]}")
        return frontiers[0]


@dataclass
class PrefillExecutionSetup:
    arena: RequestArena
    publication_manager: PublicationManager
    block_runner: OfficialFP8MLXBlockRunner
    continuation: LivePrefillContinuation | None = None
    handoff_claimed: bool = False


class DwarfStarFP8MLXPrefillExecutorSetup:
    """Own request setup for command-driven execution, not layer iteration."""

    def __init__(self, language_model: Any, *, mx: Any | None = None):
        self.language_model = language_model
        if mx is not None:
            self.mx = mx
        else:
            try:
                self.mx = importlib.import_module("mlx.core")
            except Exception as exc:
                raise PrefillSetupError("production prefill setup requires mlx.core; tests must inject an adapter") from exc

    def prepare(
        self,
        plan: SweepPlan,
        token_ids: Sequence[int],
        *,
        base_frontier: int | None = None,
        continuation: LivePrefillContinuation | list[Any] | None = None,
        image_mask: Any = None,
    ) -> PrefillExecutionSetup:
        ids = [int(t) for t in token_ids]
        if len(ids) != plan.count:
            raise PrefillSetupError(f"token slice length {len(ids)} does not match sweep plan count {plan.count}")
        cont = None
        if continuation is not None:
            cont = continuation if isinstance(continuation, LivePrefillContinuation) else LivePrefillContinuation.from_cache(continuation)
            if base_frontier is not None and int(base_frontier) != cont.frontier:
                raise PrefillSetupError("caller-supplied base_frontier contradicts live cache frontier")
            base = cont.frontier
            working_cache = cont.live_cache
        else:
            if base_frontier not in (None, 0):
                raise PrefillSetupError("fresh request cannot use nonzero base_frontier without live continuation cache")
            base = 0
            make_cache = getattr(self.language_model, "make_cache", None)
            if make_cache is None:
                raise PrefillSetupError("language model must construct request-local DeepseekV41Cache")
            working_cache = make_cache()
        prior_history = _get_cache_slot(working_cache[0], 6) if working_cache else None
        input_ids = self._input_ids(ids)
        h = self._embedding(input_ids)
        hashes, history = self._hashes(input_ids, prior_history, image_mask)
        h_current = self._hc_broadcast(h)
        h_next = self.mx.zeros_like(h_current)
        pre = self._initial_pre(h_current)
        arena = RequestArena.from_plan(
            plan,
            token_ids=ids,
            input_ids=input_ids,
            h_current=h_current,
            h_next=h_next,
            pre=pre,
            engram_hashes=hashes,
            engram_history=history,
            base_frontier=base,
        )
        topology = PublicationTopology.from_model_config(self.language_model._config)
        manager = PublicationManager(arena, topology=topology)
        p8_optimizer = getattr(self.language_model, "_p8_optimizer", None) or P8ExecutionOptimizer.from_env(mx=self.mx)
        runner = OfficialFP8MLXBlockRunner(self.language_model, manager, image_mask=image_mask, working_cache=working_cache, p8_optimizer=p8_optimizer)
        if history is not None and working_cache:
            _set_cache_slot(working_cache[0], 6, history)
        return PrefillExecutionSetup(arena=arena, publication_manager=manager, block_runner=runner, continuation=cont)

    def _input_ids(self, ids: list[int]) -> Any:
        return self.mx.array(ids, dtype=self.mx.int64)[None]

    def _embedding(self, input_ids: Any) -> Any:
        embed = getattr(self.language_model, "embed", None)
        if embed is None:
            raise PrefillSetupError("language model must expose embed for production setup")
        return embed(input_ids)

    def _hashes(self, input_ids: Any, prior_history: Any, image_mask: Any) -> tuple[Any, Any]:
        hasher = getattr(self.language_model, "_hasher", None)
        if hasher is None:
            return None, prior_history
        return hasher(input_ids, prior_history, image_mask)

    def _hc_broadcast(self, h: Any) -> Any:
        c = getattr(self.language_model, "_config", None)
        hc_mult = int(getattr(c, "hc_mult", 4))
        return self.mx.repeat(h[..., None, :], hc_mult, -2)

    def _initial_pre(self, h_current: Any) -> Any:
        c = getattr(self.language_model, "_config", None)
        hc_mult = int(getattr(c, "hc_mult", 4))
        return self.mx.broadcast_to((self.mx.arange(hc_mult) == 0).astype(self.mx.float32), h_current.shape[:-1])


def _cache_frontier(cache: Any) -> int:
    size = getattr(cache, "size", None)
    if size is not None:
        return int(size())
    value = _get_cache_slot(cache, 0)
    if value is None:
        return 0
    try:
        return int(value.item())
    except Exception:
        return int(value)


def _get_cache_slot(cache: Any, slot: int) -> Any:
    try:
        return cache[slot]
    except Exception:
        return getattr(cache, "cache", [None] * 7)[slot]


def _set_cache_slot(cache: Any, slot: int, value: Any) -> None:
    try:
        cache[slot] = value
        return
    except Exception:
        pass
    if not hasattr(cache, "cache"):
        raise PrefillSetupError("cache object does not support slot assignment")
    cache.cache[slot] = value
