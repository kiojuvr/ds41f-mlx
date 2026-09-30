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
from ds41f_mlx.prefill_fp8_mlx.publications import PublicationManager


@dataclass
class PrefillExecutionSetup:
    arena: RequestArena
    publication_manager: PublicationManager
    block_runner: OfficialFP8MLXBlockRunner


class DwarfStarFP8MLXPrefillExecutorSetup:
    """Own request setup for command-driven execution, not layer iteration."""

    def __init__(self, language_model: Any):
        self.language_model = language_model
        try:
            self.mx = importlib.import_module("mlx.core")
        except Exception:
            self.mx = None

    def prepare(self, plan: SweepPlan, token_ids: Sequence[int], *, base_frontier: int = 0, image_mask: Any = None) -> PrefillExecutionSetup:
        ids = [int(t) for t in token_ids]
        input_ids = self._input_ids(ids)
        h = self._embedding(input_ids)
        hashes, history = self._hashes(input_ids, image_mask)
        h_current = self._hc_broadcast(h)
        h_next = self._zeros_like(h_current)
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
            base_frontier=base_frontier,
        )
        manager = PublicationManager(arena)
        runner = OfficialFP8MLXBlockRunner(self.language_model, manager, image_mask=image_mask)
        runner.working_cache = self._working_cache(history)
        return PrefillExecutionSetup(arena=arena, publication_manager=manager, block_runner=runner)

    def _input_ids(self, ids: list[int]) -> Any:
        if self.mx is None:
            return ids
        return self.mx.array(ids, dtype=self.mx.int64)[None]

    def _embedding(self, input_ids: Any) -> Any:
        embed = getattr(self.language_model, "embed", None)
        if embed is None:
            return input_ids
        return embed(input_ids)

    def _hashes(self, input_ids: Any, image_mask: Any) -> tuple[Any, Any]:
        hasher = getattr(self.language_model, "_hasher", None)
        if hasher is None:
            return None, None
        return hasher(input_ids, None, image_mask)

    def _hc_broadcast(self, h: Any) -> Any:
        c = getattr(self.language_model, "_config", None)
        hc_mult = int(getattr(c, "hc_mult", 4))
        if self.mx is None:
            return h
        return self.mx.repeat(h[..., None, :], hc_mult, -2)

    def _zeros_like(self, value: Any) -> Any:
        if self.mx is None:
            clone = getattr(value, "clone_empty", None)
            return clone() if clone is not None else value
        return self.mx.zeros_like(value)

    def _initial_pre(self, h_current: Any) -> Any:
        c = getattr(self.language_model, "_config", None)
        hc_mult = int(getattr(c, "hc_mult", 4))
        if self.mx is None:
            factory = getattr(h_current, "initial_pre", None)
            return factory(hc_mult) if factory is not None else h_current
        return self.mx.broadcast_to((self.mx.arange(hc_mult) == 0).astype(self.mx.float32), h_current.shape[:-1])

    def _working_cache(self, history: Any) -> list[Any] | None:
        make_cache = getattr(self.language_model, "make_cache", None)
        if make_cache is None:
            return None
        cache = make_cache()
        if history is not None and cache:
            try:
                cache[0][6] = history
            except Exception:
                if hasattr(cache[0], "cache"):
                    cache[0].cache[6] = history
        return cache
