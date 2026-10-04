"""M8 long-lived single-flight continuation session.

The production authority is always exactly one live ``DeepseekV41Cache`` list.
ds41f ``TargetGenerationSession`` returns that exact list at a turn boundary,
then newly recipe-encoded prompt suffix tokens are appended in-place by
P6/P7 ``DeferredPrefillAppend``. The final suffix token is held out and consumed
once by the next ds41f target bootstrap, preserving the P5 zero-replay
contract across repeated turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Callable, Sequence

from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend, LivePrefillResult, handoff_to_generation
from ds41f_mlx.prefill_fp8_mlx.handoff import validate_committed_cache, _validate_live_cache_structure
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.target_generation import TargetGenerationSession, TargetGenerationStepReport


class M8ContinuationError(RuntimeError):
    """Invalid long-session transition."""


@dataclass(frozen=True)
class M8TurnRecord:
    turn_index: int
    frontier_before: int
    prompt_suffix_tokens: int
    appended_prefill_tokens: int
    terminal_token: int
    generated_tokens: tuple[int, ...]
    frontier_after: int
    prompt_replay_count: int
    full_cache_repack_count: int
    append_seconds: float
    decode_seconds: float
    first_token_latency_s: float | None = None
    cancelled: bool = False

    def to_json(self) -> dict[str, Any]:
        generated_count = len(self.generated_tokens)
        return {
            "turn_index": self.turn_index,
            "frontier_before": self.frontier_before,
            "prompt_suffix_tokens": self.prompt_suffix_tokens,
            "appended_prefill_tokens": self.appended_prefill_tokens,
            "terminal_token": self.terminal_token,
            "generated_tokens": list(self.generated_tokens),
            "frontier_after": self.frontier_after,
            "prompt_replay_count": self.prompt_replay_count,
            "full_cache_repack_count": self.full_cache_repack_count,
            "append_seconds": self.append_seconds,
            "decode_seconds": self.decode_seconds,
            "first_token_latency_s": self.first_token_latency_s,
            "decode_tok_s": None if self.decode_seconds <= 0 or generated_count == 0 else generated_count / self.decode_seconds,
            "cancelled": self.cancelled,
        }


@dataclass
class M8LiveContinuationSession:
    """Single-session append/decode lifecycle for agent-style conversations.

    ``token_history`` is request metadata used for boundary validation and
    scheduler bookkeeping; it is not a second executable state authority.  The
    executable state is ``live_cache`` while idle, or the active
    ``TargetGenerationSession`` while decoding.
    """

    model: Any
    live_cache: list[Any]
    token_history: list[int]
    config: OMLXDecodeConfig = field(default_factory=OMLXDecodeConfig)
    sampler: Callable[[Any], Any] | None = None
    mx: Any | None = None
    generation: TargetGenerationSession | None = None
    closed: bool = False
    turn_records: list[M8TurnRecord] = field(default_factory=list)
    total_prompt_replay_count: int = 0
    total_full_cache_repack_count: int = 0

    @classmethod
    def from_prefill_result(
        cls,
        *,
        model: Any,
        live_result: LivePrefillResult,
        terminal_prompt_token: int,
        config: OMLXDecodeConfig | None = None,
        max_tokens: int = 128,
        sampler: Callable[[Any], Any] | None = None,
    ) -> "M8LiveContinuationSession":
        """Create a long session from the existing M7 P5 handoff seam."""
        cfg = config or OMLXDecodeConfig()
        gen = handoff_to_generation(live_result, model, terminal_prompt_token=terminal_prompt_token, config=cfg, max_tokens=max_tokens, sampler=sampler)
        sess = cls(model=model, live_cache=[], token_history=list(live_result.prefix_token_ids), config=cfg, sampler=sampler)
        sess.generation = gen
        sess.token_history = gen.current_token_history()
        return sess

    @classmethod
    def from_live_cache(
        cls,
        *,
        model: Any,
        live_cache: list[Any],
        token_history: Sequence[int],
        config: OMLXDecodeConfig | None = None,
        sampler: Callable[[Any], Any] | None = None,
        mx: Any | None = None,
    ) -> "M8LiveContinuationSession":
        cfg = config or OMLXDecodeConfig()
        ids = [int(t) for t in token_history]
        if not ids:
            raise M8ContinuationError("long session requires non-empty token history")
        _validate_live_cache_structure(live_cache, getattr(model, "language_model", model)._config, len(ids))
        return cls(model=model, live_cache=live_cache, token_history=ids, config=cfg, sampler=sampler, mx=mx)

    @property
    def frontier(self) -> int:
        return len(self.token_history)

    @property
    def state(self) -> str:
        if self.closed:
            return "closed"
        return "generating" if self.generation is not None else "idle"

    def ensure_idle(self, reason: str = "turn_boundary") -> None:
        if self.closed:
            raise M8ContinuationError("session is closed")
        if self.generation is None:
            return
        generation = self.generation
        cache, history = generation.extract_final_state(reason)
        self.live_cache = cache
        self.token_history = [int(t) for t in history]
        self.total_prompt_replay_count += int(generation.prompt_replay_count)
        try:
            generation.close()
        finally:
            self.generation = None
        self._assert_cache_frontier()

    def begin_turn_from_recipe_tokens(self, full_recipe_token_ids: Sequence[int], *, max_tokens: int = 128) -> None:
        """Append only the new recipe suffix and start decode at its terminal.

        The canonical boundary is an exact token-prefix relation between the
        recipe-encoded next conversation and the completed generated history. If
        the prefix does not match, the turn is rejected before mutating cache.
        """
        self.ensure_idle("before_append")
        full = [int(t) for t in full_recipe_token_ids]
        if len(full) <= len(self.token_history):
            raise M8ContinuationError("next recipe encoding contains no new terminal token")
        if full[: len(self.token_history)] != self.token_history:
            raise M8ContinuationError("next recipe encoding is not an exact extension of generated session history")
        suffix = full[len(self.token_history):]
        self._append_suffix_and_start(suffix, max_tokens=max_tokens)

    def begin_turn_from_suffix(self, suffix_token_ids: Sequence[int], *, max_tokens: int = 128) -> None:
        self.ensure_idle("before_append")
        suffix = [int(t) for t in suffix_token_ids]
        self._append_suffix_and_start(suffix, max_tokens=max_tokens)

    def _append_suffix_and_start(self, suffix: list[int], *, max_tokens: int) -> None:
        if not suffix:
            raise M8ContinuationError("suffix must contain at least the held-out terminal token")
        if max_tokens < 1:
            raise M8ContinuationError("max_tokens must be positive")
        frontier_before = len(self.token_history)
        append_tokens = suffix[:-1]
        terminal = int(suffix[-1])
        append_seconds = 0.0
        full_cache_repack_count = 0
        if append_tokens:
            target_history = self.token_history + append_tokens
            t0 = perf_counter()
            app = DeferredPrefillAppend.create(
                getattr(self.model, "language_model", self.model),
                self.live_cache,
                target_history,
                committed_frontier=frontier_before,
                mx=self.mx,
            )
            app.execute_all()
            append_seconds = perf_counter() - t0
            if app.commit_certificate is None:
                raise M8ContinuationError("append did not produce a sealed P6 commit")
            live = LivePrefillResult.from_committed(app.commit_certificate, prefix_token_ids=target_history)
            full_cache_repack_count = int(getattr(app.commit_certificate, "full_cache_repack_count", 0))
            self.live_cache = live.live_cache
            self.token_history = list(target_history)
        else:
            gen = TargetGenerationSession.from_prefilled_cache(self.model, self.live_cache, self.token_history, self.config, max_tokens=max_tokens, sampler=self.sampler)
            try:
                gen.start(terminal, max_tokens=max_tokens)
            except BaseException:
                # Bootstrap may already have mutated the sole cache lease.
                # Never leave it looking like committed idle continuation.
                self.live_cache = []
                self.closed = True
                gen.close()
                raise
            self.live_cache = []
            self.generation = gen
            self.token_history = gen.current_token_history()
            self.turn_records.append(M8TurnRecord(
                turn_index=len(self.turn_records),
                frontier_before=frontier_before,
                prompt_suffix_tokens=len(suffix),
                appended_prefill_tokens=0,
                terminal_token=terminal,
                generated_tokens=(),
                frontier_after=len(self.token_history),
                prompt_replay_count=int(gen.prompt_replay_count),
                full_cache_repack_count=0,
                append_seconds=append_seconds,
                decode_seconds=0.0,
                first_token_latency_s=None,
            ))
            return
        gen = handoff_to_generation(live, self.model, terminal_prompt_token=terminal, config=self.config, max_tokens=max_tokens, sampler=self.sampler)
        self.live_cache = []
        self.generation = gen
        self.token_history = gen.current_token_history()
        self.total_full_cache_repack_count += full_cache_repack_count
        self.turn_records.append(M8TurnRecord(
            turn_index=len(self.turn_records),
            frontier_before=frontier_before,
            prompt_suffix_tokens=len(suffix),
            appended_prefill_tokens=len(append_tokens),
            terminal_token=terminal,
            generated_tokens=(),
            frontier_after=len(self.token_history),
            prompt_replay_count=int(gen.prompt_replay_count),
            full_cache_repack_count=full_cache_repack_count,
            append_seconds=append_seconds,
            decode_seconds=0.0,
            first_token_latency_s=None,
        ))

    def next_token(self) -> TargetGenerationStepReport | None:
        if self.generation is None:
            raise M8ContinuationError("no active generation; call begin_turn first")
        t0 = perf_counter()
        report = self.generation.next_token()
        elapsed = perf_counter() - t0
        if report is not None:
            self.token_history.append(int(report.token))
            last = self.turn_records[-1]
            self.turn_records[-1] = replace(
                last,
                generated_tokens=tuple(list(last.generated_tokens) + [int(report.token)]),
                frontier_after=len(self.token_history),
                decode_seconds=float(last.decode_seconds + elapsed),
                first_token_latency_s=elapsed if not last.generated_tokens else last.first_token_latency_s,
                prompt_replay_count=int(self.generation.prompt_replay_count),
            )
        return report

    def cancel_turn(self, reason: str = "cancelled") -> None:
        self.ensure_idle(reason)
        if self.turn_records:
            last = self.turn_records[-1]
            self.turn_records[-1] = replace(last, cancelled=True)

    def close(self) -> None:
        if self.closed:
            return
        if self.generation is not None:
            self.ensure_idle("close")
        self.live_cache = []
        self.closed = True

    def diagnostics(self) -> dict[str, Any]:
        offsets = () if not self.live_cache else tuple(int(item.size()) for item in self.live_cache)
        return {
            "schema": "ds41f.m8.live-continuation.diagnostics.v1",
            "state": self.state,
            "frontier": self.frontier,
            "cache_offsets_head": list(offsets[:8]),
            "cache_offsets_all": list(offsets),
            "cache_layer_count": len(offsets),
            "all_cache_offsets_equal_frontier": (not offsets) or all(o == self.frontier for o in offsets),
            "turn_count": len(self.turn_records),
            "total_prompt_replay_count": self.total_prompt_replay_count,
            "total_full_cache_repack_count": self.total_full_cache_repack_count,
            "turns": [r.to_json() for r in self.turn_records[-16:]],
        }

    def _assert_cache_frontier(self) -> None:
        offsets = tuple(int(item.size()) for item in self.live_cache)
        if len(offsets) != 40 or any(o != len(self.token_history) for o in offsets):
            raise M8ContinuationError(f"cache offsets {offsets[:4]} do not match token history {len(self.token_history)}")


class _AdHocCommittedCache:
    """Minimal committed-cache adapter for an already authoritative live cache."""

    def __init__(self, cache: list[Any], model: Any, token_history: Sequence[int]):
        lm = getattr(model, "language_model", model)
        ids = tuple(int(t) for t in token_history)
        T = len(ids)
        self.live_cache = cache
        self.prefix_token_ids = ids
        self.C = self.E = self.D = self.T = T
        self.owner_token = -1
        self.source_coverage = {2: T, 8: T, 14: T, 20: T}
        self.layer_coverage = {i: T for i in range(40)}
        self.history_position = T
        self.sealed = True
        runner = type("_Runner", (), {})()
        runner.working_cache = cache
        runner.language_model = lm
        runner.full_cache_repack_count = 0
        runner.prefill_continuation_exported = False
        runner.handoff_transferred = False
        runner.handoff_reserved = False
        runner.scheduling_coordinator = None
        runner.execution_revoked = False
        runner.logits_policy = type("_Policy", (), {"compute_final_prefix_logits": False})()
        runner.final_logits_suppressed = True
        runner.final_logits = None
        manager = type("_Manager", (), {"pending_cumulative_by_layer": {}, "pending_spans_by_layer": {}, "visible_spans": {}})()
        arena = type("_Arena", (), {"prefill_continuation_exported": False})()
        setup = type("_Setup", (), {})()
        setup.block_runner = runner
        setup.publication_manager = manager
        setup.arena = arena
        setup.handoff_claimed = False
        setup.continuation = None
        self.final_setup = setup
