"""P6 owner-mediated deferred-prefill append planning and execution.

This module is deliberately separate from the legacy ``encoder_only`` /
``resume_encoder`` sweep flags.  It provides request-level ownership, pinned
DwarfStar capacity geometry, private C/E/D/T frontiers, and structural command
plans for deferred source-only and completing decoder segments.  The production
runtime selector is not touched by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from .arena import RequestArena, TensorOwnership
from .block_runner import OfficialFP8MLXBlockRunner, _make_cache_offset
from .executor import PrefillExecutionSetup, PrefillSetupError
from .planner import SweepAllocation, SweepCommand, SweepCommandKind, SweepPhase, SweepPlan
from .publications import PublicationManager, PublicationTopology
from .p7_scheduling import SchedulingTelemetry

P6_CARRY_CAPACITY = 16384
P6_FINAL_TAIL_THRESHOLD = 8192
P6_ROUND = 2048
P6_ENCODER_TILE = 8192
P6_LOCAL_PREPARE_ROWS = 127
P6_LAYER20_QUERY_ROWS = 1 + (39 - 20) * P6_LOCAL_PREPARE_ROWS  # 2414
P6_LAYER20_INPUT_CONE_ROWS = P6_LAYER20_QUERY_ROWS + P6_LOCAL_PREPARE_ROWS  # 2541
_P6_OWNER_COUNTER = 0


def _next_owner_token() -> int:
    global _P6_OWNER_COUNTER
    _P6_OWNER_COUNTER += 1
    return _P6_OWNER_COUNTER


class P6AppendError(RuntimeError):
    """Invalid P6 append lifecycle or geometry."""


class SegmentMode(str, Enum):
    ENCODER_SOURCE_ONLY = "ENCODER_SOURCE_ONLY"
    FINAL_ENCODER_DECODER = "FINAL_ENCODER_DECODER"
    ORDINARY_COMPLETE_RANGE = "ORDINARY_COMPLETE_RANGE"


class AppendState(str, Enum):
    VALID = "valid"
    PENDING = "pending"
    FAILED = "failed"
    SEALED = "sealed"


@dataclass(frozen=True)
class P6SegmentPlan:
    seq: int
    start: int
    count: int
    mode: SegmentMode
    commands: tuple[SweepCommand, ...]
    source_only: bool = False
    completes_decoder: bool = False
    ordinary_tail: bool = False

    @property
    def end(self) -> int:
        return self.start + self.count

    def to_json(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "start": self.start,
            "count": self.count,
            "end": self.end,
            "mode": self.mode.value,
            "source_only": self.source_only,
            "completes_decoder": self.completes_decoder,
            "ordinary_tail": self.ordinary_tail,
            "commands": [c.to_json() for c in self.commands],
        }


@dataclass(frozen=True)
class AppendPlan:
    committed_frontier: int
    target: int
    capacity: int
    segments: tuple[P6SegmentPlan, ...]
    defer_threshold: int = P6_FINAL_TAIL_THRESHOLD

    @property
    def C(self) -> int:
        return self.committed_frontier

    @property
    def T(self) -> int:
        return self.target

    def to_json(self) -> dict[str, object]:
        return {
            "C": self.C,
            "T": self.T,
            "capacity": self.capacity,
            "segments": [s.to_json() for s in self.segments],
        }


@dataclass
class AppendCoverage:
    per_layer: dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(40)})
    source_by_layer: dict[int, int] = field(default_factory=lambda: {2: 0, 8: 0, 14: 0, 20: 0})

    def reaches(self, frontier: int) -> bool:
        return all(v >= frontier for v in self.per_layer.values()) and all(v >= frontier for v in self.source_by_layer.values())


@dataclass
class SegmentExecution:
    segment: P6SegmentPlan
    arena: RequestArena
    manager: PublicationManager
    runner: OfficialFP8MLXBlockRunner


@dataclass(frozen=True)
class P6SegmentRecord:
    mode: SegmentMode
    start: int
    end: int
    command_count: int
    source_generation_count: int
    E_before: int
    E_after: int
    D_before: int
    D_after: int
    materialized: bool
    retired: bool
    cone_rows: int = 0
    cone_origin: int | None = None
    scheduling_events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class P6AppendCommit:
    live_cache: list[Any]
    prefix_token_ids: tuple[int, ...]
    C: int
    E: int
    D: int
    T: int
    owner_token: int
    source_coverage: dict[int, int]
    layer_coverage: dict[int, int]
    history_position: int
    final_setup: PrefillExecutionSetup
    sealed: bool = True
    full_cache_repack_count: int = 0
    final_logits_suppressed: bool = True


@dataclass
class DeferredPrefillAppend:
    """Exclusive owner of one invalid-until-sealed P6 append transaction."""

    language_model: Any
    live_cache: list[Any]
    request_token_history: tuple[int, ...]
    plan: AppendPlan
    mx: Any | None = None
    C: int = field(init=False)
    E: int = field(init=False)
    D: int = field(init=False)
    T: int = field(init=False)
    engram_history_position: int = field(init=False)
    private_history_at_E: Any = field(default=None, init=False)
    state: AppendState = field(default=AppendState.VALID, init=False)
    coverage: AppendCoverage = field(default_factory=AppendCoverage)
    segment_capacity: int = field(default=P6_CARRY_CAPACITY, init=False)
    segment_records: list[P6SegmentRecord] = field(default_factory=list)
    active_execution: SegmentExecution | None = None
    final_execution: SegmentExecution | None = None
    failed_reason: str | None = None
    live_setup: PrefillExecutionSetup | None = None
    commit_certificate: P6AppendCommit | None = None
    owner_token: int = field(init=False)
    source_generation_counts: dict[int, int] = field(default_factory=lambda: {20: 0})
    stale_generation: int = 0

    def __post_init__(self) -> None:
        self.owner_token = _next_owner_token()
        self.C = int(self.plan.C)
        self.E = self.C
        self.D = self.C
        self.T = int(self.plan.T)
        self.engram_history_position = self.C
        self.private_history_at_E = self._cache_history()
        if len(self.request_token_history) != self.T:
            raise P6AppendError("complete request token history length must equal target T")
        if self.plan.C != self._public_frontier():
            raise P6AppendError("append entry frontier does not match public live cache")
        for layer in range(40):
            self.coverage.per_layer[layer] = self.C
        for layer in self.coverage.source_by_layer:
            self.coverage.source_by_layer[layer] = self.C

    @classmethod
    def create(cls, language_model: Any, live_cache: list[Any], request_token_history: Sequence[int], *, committed_frontier: int | None = None, capacity: int = P6_CARRY_CAPACITY, mx: Any | None = None) -> "DeferredPrefillAppend":
        C = _frontier_from_cache(live_cache) if committed_frontier is None else int(committed_frontier)
        plan = P6AppendPlanner(capacity=capacity).plan(C=C, T=len(request_token_history))
        return cls(language_model=language_model, live_cache=live_cache, request_token_history=tuple(int(t) for t in request_token_history), plan=plan, mx=mx)

    @classmethod
    def continue_from_commit(cls, language_model: Any, prior_commit: P6AppendCommit, request_token_history: Sequence[int], *, capacity: int = P6_CARRY_CAPACITY, mx: Any | None = None) -> "DeferredPrefillAppend":
        if not getattr(prior_commit, "sealed", False):
            raise P6AppendError("prior P6 commit is not sealed")
        C, E, D, T = int(prior_commit.C), int(prior_commit.E), int(prior_commit.D), int(prior_commit.T)
        if C > T or E != T or D != T or int(prior_commit.history_position) != T:
            raise P6AppendError("prior P6 commit frontiers are not coherent")
        if len(prior_commit.prefix_token_ids) != T:
            raise P6AppendError("prior P6 commit token history length mismatch")
        setup = prior_commit.final_setup
        runner = setup.block_runner
        cache = prior_commit.live_cache
        if cache is not runner.working_cache or cache is None or len(cache) != 40:
            raise P6AppendError("prior P6 commit does not own the same live cache")
        if setup.handoff_claimed or runner.handoff_reserved or runner.handoff_transferred:
            raise P6AppendError("prior P6 commit has been reserved/transferred for P5")
        if any(v < T for v in prior_commit.source_coverage.values()) or any(v < T for v in prior_commit.layer_coverage.values()):
            raise P6AppendError("prior P6 commit coverage does not reach T")
        if _frontier_from_cache(cache) != T:
            raise P6AppendError("prior P6 public frontier mismatch")
        for item in cache:
            if getattr(item, "_p6_append_failed", False) or getattr(item, "_p6_append_invalid", False) or getattr(item, "_p6_append_pending", False) or not getattr(item, "_p6_append_sealed", False):
                raise P6AppendError("prior P6 cache is not sealed/admissible for continuation")
            if getattr(item, "_p6_owner_token", None) != prior_commit.owner_token:
                raise P6AppendError("prior P6 owner token mismatch")
        plan = P6AppendPlanner(capacity=capacity).plan(C=T, T=len(request_token_history))
        app = cls(language_model=language_model, live_cache=cache, request_token_history=tuple(int(t) for t in request_token_history), plan=plan, mx=mx)
        app.state = AppendState.PENDING
        app.stale_generation += 1
        for item in cache:
            try:
                setattr(item, "_p6_append_sealed", False)
                setattr(item, "_p6_append_invalid", True)
                setattr(item, "_p6_append_pending", True)
                setattr(item, "_p6_owner_token", app.owner_token)
            except Exception:
                pass
        return app

    def begin(self) -> None:
        if self.state is not AppendState.VALID:
            raise P6AppendError("append already begun or not valid")
        self.state = AppendState.PENDING
        self.stale_generation += 1
        for cache in self.live_cache:
            try:
                setattr(cache, "_p6_append_invalid", True)
                setattr(cache, "_p6_append_pending", True)
                setattr(cache, "_p6_owner_token", self.owner_token)
            except Exception:
                pass

    def execute_all(self) -> None:
        if self.state is AppendState.VALID:
            self.begin()
        for segment in self.plan.segments:
            self.execute_segment(segment)
        self.final_seal()

    def execute_segment(self, segment: P6SegmentPlan) -> SegmentExecution:
        if self.state is not AppendState.PENDING:
            raise P6AppendError("segments require the append owner's pending capability")
        if segment.start != self.E:
            self.fail("segment cursor mismatch")
            raise P6AppendError("segment cursor mismatch")
        if self._public_frontier() != self.C:
            self.fail("public frontier changed while append invalid")
            raise P6AppendError("public frontier changed while append invalid")
        try:
            execn = self._make_segment_execution(segment)
            self.active_execution = execn
            e_before, d_before = self.E, self.D
            execn.runner.execute_batch(segment.commands, execn.arena)
            self._ack_segment(segment, execn.arena, e_before=e_before, d_before=d_before)
            if segment.mode is SegmentMode.ENCODER_SOURCE_ONLY:
                execn.manager.retire_row_spans()
                execn.runner.close()
                execn.arena.retire_encoder_range_after_source_boundary()
                self.active_execution = None
            else:
                self.final_execution = execn
                self.active_execution = None
            return execn
        except Exception as exc:
            self.fail(str(exc))
            raise

    def _make_segment_execution(self, segment: P6SegmentPlan) -> SegmentExecution:
        token_slice = self.request_token_history[segment.start:segment.end]
        plan = _sweep_shell(segment.count, segment.commands, encoder_only=segment.source_only)
        h_current, h_next, pre, input_ids, hashes, history = _make_segment_tensors(self.language_model, self.mx, token_slice, self._private_history(), None)
        arena = RequestArena.from_plan(plan, token_ids=token_slice, input_ids=input_ids, h_current=h_current, h_next=h_next, pre=pre, engram_hashes=hashes, engram_history=history, base_frontier=segment.start)
        arena.p6_public_frontier = self.C
        arena.p6_private_start = segment.start
        arena.p6_freeze_public_offsets = True
        arena.p6_segment_mode = segment.mode.value
        arena.p6_segment_origin = segment.start
        manager = PublicationManager(arena, topology=PublicationTopology.from_model_config(self.language_model._config))
        if not self.segment_records:
            manager.begin_append_transaction()
        else:
            # Persistent cumulative handles remain in the live cache; old row spans are not carried.
            manager.append_transaction_active = True
        runner = OfficialFP8MLXBlockRunner(self.language_model, manager, working_cache=self.live_cache, mx=self.mx, p8_optimizer=getattr(self.language_model, "_p8_optimizer", None))
        runner.p6_owner_token = self.owner_token
        return SegmentExecution(segment, arena, manager, runner)

    def _ack_segment(self, segment: P6SegmentPlan, arena: RequestArena, *, e_before: int, d_before: int) -> None:
        if not arena.p6_source_materialized and segment.mode is not SegmentMode.ORDINARY_COMPLETE_RANGE:
            raise P6AppendError("P6 source boundary was not materialized")
        if segment.mode is SegmentMode.ENCODER_SOURCE_ONLY:
            self.E = segment.end
            for layer in range(20):
                self.coverage.per_layer[layer] = self.E
            for layer in self.coverage.source_by_layer:
                self.coverage.source_by_layer[layer] = self.E
            self.source_generation_counts[20] += 1
            self.engram_history_position = self.E
            self.private_history_at_E = arena.engram.history.value
            self._assert_public_frozen()
            self._record_segment(segment, arena, e_before, d_before, retired=True)
            return
        if segment.mode is SegmentMode.FINAL_ENCODER_DECODER:
            self.E = segment.end
            self.D = segment.end
            for layer in range(40):
                self.coverage.per_layer[layer] = self.D
            for layer in self.coverage.source_by_layer:
                self.coverage.source_by_layer[layer] = self.E
            self.source_generation_counts[20] += 1
            self.engram_history_position = self.E
            self.private_history_at_E = arena.engram.history.value
            self._assert_public_frozen()
            self._record_segment(segment, arena, e_before, d_before, retired=False)
            return
        if segment.mode is SegmentMode.ORDINARY_COMPLETE_RANGE:
            self.E = segment.end
            self.D = segment.end
            for layer in range(40):
                self.coverage.per_layer[layer] = self.D
            for layer in self.coverage.source_by_layer:
                self.coverage.source_by_layer[layer] = max(self.coverage.source_by_layer[layer], self.D)
            self.engram_history_position = self.D
            self.private_history_at_E = arena.engram.history.value
            self._assert_public_frozen()
            self._record_segment(segment, arena, e_before, d_before, retired=False)
            return
        raise P6AppendError("unknown segment mode")

    def readiness(self) -> bool:
        return (
            self.state is AppendState.PENDING
            and self.E == self.D == self.T
            and self.coverage.reaches(self.T)
            and self.engram_history_position == self.T
            and len(self.request_token_history) == self.T
            and self._public_frontier() == self.C
        )

    def final_seal(self) -> PrefillExecutionSetup:
        if not self.readiness() or self.final_execution is None:
            self.fail("final readiness check failed")
            raise P6AppendError("final readiness check failed")
        try:
            last = self.final_execution
            if any((r.mode is not SegmentMode.ORDINARY_COMPLETE_RANGE and not r.materialized) for r in self.segment_records):
                raise P6AppendError("P6 materialization record missing at final seal")
            if any(r.mode is not SegmentMode.ORDINARY_COMPLETE_RANGE for r in self.segment_records) and not any(r.cone_rows == P6_LAYER20_INPUT_CONE_ROWS for r in self.segment_records):
                raise P6AppendError("P6 compact decoder cone was not acknowledged")
            if last.manager.pending_cumulative_by_layer or last.manager.pending_spans_by_layer:
                raise P6AppendError("pending publications remain at final seal")
            last.manager.retire_row_spans()
            last.runner.final_logits_suppressed = True
            _set_cache_history(self.live_cache, self.private_history_at_E)
            setup = PrefillExecutionSetup(last.arena, last.manager, last.runner)
            commit = P6AppendCommit(
                live_cache=self.live_cache,
                prefix_token_ids=self.request_token_history,
                C=self.C, E=self.E, D=self.D, T=self.T,
                owner_token=self.owner_token,
                source_coverage=dict(self.coverage.source_by_layer),
                layer_coverage=dict(self.coverage.per_layer),
                history_position=self.engram_history_position,
                final_setup=setup,
            )
            # Only after the certificate is constructed do we advertise T.
            _set_all_public_frontiers(self.live_cache, self.T, self.language_model)
            for cache in self.live_cache:
                try:
                    setattr(cache, "_p6_append_invalid", False)
                    setattr(cache, "_p6_append_pending", False)
                    setattr(cache, "_p6_append_sealed", True)
                    setattr(cache, "_p6_owner_token", self.owner_token)
                except Exception:
                    pass
            last.runner.execution_revoked = True
            self.commit_certificate = commit
            self.live_setup = setup
            self.state = AppendState.SEALED
            return setup
        except Exception as exc:
            self.fail(f"final seal failed: {exc}")
            raise

    def fail(self, reason: str) -> None:
        self.state = AppendState.FAILED
        self.failed_reason = reason
        self.stale_generation += 1
        for cache in self.live_cache:
            try:
                setattr(cache, "_p6_append_failed", True)
            except Exception:
                pass
        for cache in self.live_cache:
            try:
                setattr(cache, "_p6_append_invalid", True)
                setattr(cache, "_p6_append_pending", False)
                setattr(cache, "_p6_append_sealed", False)
                setattr(cache, "_p6_owner_token", None)
            except Exception:
                pass
        for execn in (self.active_execution, self.final_execution):
            if execn is None:
                continue
            execn.manager.fail()
            execn.arena.transaction.fail()
            execn.arena.retire_encoder_range_after_source_boundary()
            execn.runner.close()

    def rebuild_with_tokens(self, token_history: Sequence[int]) -> "DeferredPrefillAppend":
        make_cache = getattr(self.language_model, "make_cache", None)
        if make_cache is None:
            raise P6AppendError("language model cannot rebuild without make_cache")
        fresh = make_cache()
        _set_all_public_frontiers(fresh, 0, self.language_model)
        return DeferredPrefillAppend.create(self.language_model, fresh, tuple(int(t) for t in token_history), committed_frontier=0, mx=self.mx)

    def _record_segment(self, segment: P6SegmentPlan, arena: RequestArena, e_before: int, d_before: int, *, retired: bool) -> None:
        self.segment_records.append(P6SegmentRecord(
            mode=segment.mode,
            start=segment.start,
            end=segment.end,
            command_count=len(segment.commands),
            source_generation_count=self.source_generation_counts.get(20, 0),
            E_before=e_before,
            E_after=self.E,
            D_before=d_before,
            D_after=self.D,
            materialized=arena.p6_source_materialized or segment.mode is SegmentMode.ORDINARY_COMPLETE_RANGE,
            retired=retired,
            cone_rows=arena.p6_final_cone_rows,
            cone_origin=arena.p6_final_cone_origin,
            scheduling_events=tuple(dict(e) for e in getattr(getattr(getattr(self.active_execution, "runner", None), "scheduling_coordinator", None), "telemetry", SchedulingTelemetry()).events),
        ))

    def _private_history(self) -> Any:
        return self.private_history_at_E

    def _cache_history(self) -> Any:
        try:
            return self.live_cache[0][6]
        except Exception:
            return None

    def _public_frontier(self) -> int:
        return _frontier_from_cache(self.live_cache)

    def _assert_public_frozen(self) -> None:
        if self._public_frontier() != self.C:
            raise P6AppendError("public slot0 advanced before final P6 seal")


class P6AppendPlanner:
    """Pinned capacity planner matching the P6 architecture fixtures."""

    def __init__(self, *, capacity: int = P6_CARRY_CAPACITY):
        if capacity != P6_CARRY_CAPACITY:
            raise P6AppendError("P6 qualification is pinned to capacity 16384")
        self.capacity = int(capacity)

    def plan(self, *, C: int, T: int, deferral_enabled: bool = True) -> AppendPlan:
        C, T = int(C), int(T)
        if T < C:
            raise P6AppendError("target precedes committed frontier")
        pos = C
        pending = False
        segments: list[P6SegmentPlan] = []
        seq = 0
        while T - pos > 0:
            remaining = T - pos
            if remaining <= 3 and not pending:
                count = 1
                mode = SegmentMode.ORDINARY_COMPLETE_RANGE
            else:
                count = min(self.capacity, remaining)
                if remaining > P6_FINAL_TAIL_THRESHOLD:
                    count = (count // P6_ROUND) * P6_ROUND
                rem_after = remaining - count
                defer = (count >= self.capacity and rem_after >= P6_FINAL_TAIL_THRESHOLD)
                if defer and deferral_enabled:
                    mode = SegmentMode.ENCODER_SOURCE_ONLY
                    pending = True
                elif pending:
                    if count < P6_FINAL_TAIL_THRESHOLD:
                        raise P6AppendError("unsupported pending-to-small-range transition")
                    mode = SegmentMode.FINAL_ENCODER_DECODER
                    pending = False
                else:
                    mode = SegmentMode.FINAL_ENCODER_DECODER if count >= P6_FINAL_TAIL_THRESHOLD else SegmentMode.ORDINARY_COMPLETE_RANGE
            commands = _segment_commands(seq=seq, count=count, mode=mode)
            segments.append(P6SegmentPlan(seq, pos, count, mode, commands, source_only=mode is SegmentMode.ENCODER_SOURCE_ONLY, completes_decoder=mode is SegmentMode.FINAL_ENCODER_DECODER, ordinary_tail=mode is SegmentMode.ORDINARY_COMPLETE_RANGE))
            pos += count
            seq += 1
        return AppendPlan(committed_frontier=C, target=T, capacity=self.capacity, segments=tuple(segments))

    def matched_control_plan(self, *, C: int, T: int) -> AppendPlan:
        """Qualification-only non-deferred control with the same P6 geometry."""
        return self.plan(C=C, T=T, deferral_enabled=False)

    def fixture(self, name: str) -> AppendPlan:
        fixtures = {
            "tiny_16385": (0, 16385),
            "A_24577": (0, 24577),
            "B_fresh_49155": (0, 49155),
            "B_continued_C24578_T49155": (24578, 49155),
        }
        if name not in fixtures:
            raise P6AppendError(f"unknown P6 fixture {name!r}")
        C, T = fixtures[name]
        return self.plan(C=C, T=T)


def _segment_commands(*, seq: int, count: int, mode: SegmentMode) -> tuple[SweepCommand, ...]:
    commands: list[SweepCommand] = []
    def add(kind: SweepCommandKind, layer: int | None, offset: int, rows: int, phase: SweepPhase, valid: bool = False) -> None:
        commands.append(SweepCommand(len(commands), kind, 0, layer, int(offset), int(rows), phase, 0, valid))
    add(SweepCommandKind.BEGIN_INVALIDATE, None, 0, count, SweepPhase.SWEEP, False)
    if mode in (SegmentMode.ENCODER_SOURCE_ONLY, SegmentMode.FINAL_ENCODER_DECODER):
        add(SweepCommandKind.PREFETCH_ENGRAM0, None, 0, count, SweepPhase.ENCODER, False)
        for layer in range(20):
            if layer == 2:
                add(SweepCommandKind.PREFETCH_ENGRAM1, None, 0, count, SweepPhase.ENCODER, False)
            add(SweepCommandKind.SSD_READ_AHEAD, layer, 0, count, SweepPhase.ENCODER, False)
            _layer_commands(add, layer, count, SweepPhase.ENCODER)
        add(SweepCommandKind.DECODER_PREPARE_SUFFIX, 20, 0, count, SweepPhase.ENCODER, False)
        add(SweepCommandKind.PUBLISH_FRONTIER, 20, 0, count, SweepPhase.ENCODER, False)
        if mode is SegmentMode.ENCODER_SOURCE_ONLY:
            add(SweepCommandKind.P6_SOURCE_COMPLETE_AND_DETACH_CONE, None, 0, 0, SweepPhase.DEFERRED_DECODER, False)
            add(SweepCommandKind.ENCODER_ONLY_COMPLETE_INVALID, None, 0, count, SweepPhase.DEFERRED_DECODER, False)
            return tuple(commands)
        add(SweepCommandKind.P6_SOURCE_COMPLETE_AND_DETACH_CONE, None, count - P6_LAYER20_INPUT_CONE_ROWS, P6_LAYER20_INPUT_CONE_ROWS, SweepPhase.DEFERRED_DECODER, False)
        for layer in range(20, 40):
            q = 1 + (39 - layer) * P6_LOCAL_PREPARE_ROWS
            add(SweepCommandKind.SSD_READ_AHEAD, layer, count - q, q, SweepPhase.DECODER_SUFFIX, False)
            r = q + P6_LOCAL_PREPARE_ROWS
            add(SweepCommandKind.BEGIN_LAYER, layer, count - q, q, SweepPhase.DECODER_SUFFIX, False)
            add(SweepCommandKind.DECODER_PREPARE_SUFFIX, layer, count - r, P6_LOCAL_PREPARE_ROWS, SweepPhase.DECODER_SUFFIX, False)
            add(SweepCommandKind.ENCODE_ROWS, layer, count - q, q, SweepPhase.DECODER_SUFFIX, False)
            add(SweepCommandKind.SWAP_HC_AFTER_LAYER, layer, count - q, q, SweepPhase.DECODER_SUFFIX, False)
            add(SweepCommandKind.PUBLISH_FRONTIER, layer, count - q, q, SweepPhase.DECODER_SUFFIX, False)
            add(SweepCommandKind.END_LAYER, layer, count - q, q, SweepPhase.DECODER_SUFFIX, False)
        return tuple(commands)
    add(SweepCommandKind.PREFETCH_ENGRAM0, None, 0, count, SweepPhase.DECODER_FULL, False)
    for layer in range(40):
        if layer == 2:
            add(SweepCommandKind.PREFETCH_ENGRAM1, None, 0, count, SweepPhase.DECODER_FULL, False)
        add(SweepCommandKind.SSD_READ_AHEAD, layer, 0, count, SweepPhase.DECODER_FULL, False)
        _layer_commands(add, layer, count, SweepPhase.DECODER_FULL)
    return tuple(commands)


def _layer_commands(add, layer: int, count: int, phase: SweepPhase) -> None:
    add(SweepCommandKind.BEGIN_LAYER, layer, 0, count, phase, False)
    off = 0
    while off < count:
        rows = min(P6_ENCODER_TILE, count - off)
        add(SweepCommandKind.ENCODE_ROWS, layer, off, rows, phase, False)
        off += rows
    add(SweepCommandKind.SWAP_HC_AFTER_LAYER, layer, 0, count, phase, False)
    add(SweepCommandKind.PUBLISH_FRONTIER, layer, 0, count, phase, False)
    add(SweepCommandKind.END_LAYER, layer, 0, count, phase, False)


def _sweep_shell(count: int, commands: Sequence[SweepCommand], *, encoder_only: bool) -> SweepPlan:
    allocs = (
        SweepAllocation("batch_cur_hc", 0, 1, "opaque", 0, 10**9, "arena", "hc_pingpong", "stage-local reusable scratch", False, False),
        SweepAllocation("batch_next_hc", 0, 1, "opaque", 0, 10**9, "arena", "hc_pingpong", "stage-local reusable scratch", False, False),
        SweepAllocation("carry.pre", 0, 1, "opaque", 0, 10**9, "arena", "pre", "stage-local reusable scratch", False, False),
        SweepAllocation("decoder_suffix_rows", 0, 1, "opaque", 0, 10**9, "arena", "decoder_suffix", "stage-local reusable scratch", False, True),
    )
    return SweepPlan(count=int(count), prefill_cap=P6_ENCODER_TILE, encoder_chunk=P6_ENCODER_TILE, wide=count >= P6_FINAL_TAIL_THRESHOLD, decoder_suffix=any(c.phase is SweepPhase.DECODER_SUFFIX for c in commands), encoder_only=encoder_only, resume_encoder=False, defer_decoder_candidate=False, checkpoint_valid_during_sweep=False, checkpoint_valid_after_sweep=False, encoder_row_layer_work=0, decoder_suffix_row_layer_work=0, total_row_layer_work=0, allocations=allocs, commands=tuple(commands), command_counts={}, prefetch_event_count=0, checkpoint_transition_count=0)


def _make_segment_tensors(language_model: Any, mx: Any, token_slice: Sequence[int], prior_history: Any, image_mask: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    if mx is None:
        try:
            import mlx.core as mx  # type: ignore
        except Exception:
            mx = None
    if mx is not None:
        input_ids = mx.array(list(token_slice), dtype=mx.int64)[None]
    else:
        input_ids = _SimpleInputIds(token_slice)
    embed = getattr(language_model, "embed", None)
    if embed is None:
        raise PrefillSetupError("language model must expose embed for P6 append")
    h = embed(input_ids)
    if mx is not None:
        hc_mult = int(getattr(language_model._config, "hc_mult", 4))
        h_current = mx.repeat(h[..., None, :], hc_mult, -2)
        h_next = mx.zeros_like(h_current)
        pre = mx.broadcast_to((mx.arange(hc_mult) == 0).astype(mx.float32), h_current.shape[:-1])
    else:
        h_current = h_next = pre = h
    hasher = getattr(language_model, "_hasher", None)
    hashes, history = (None, prior_history) if hasher is None else hasher(input_ids, prior_history, image_mask)
    return h_current, h_next, pre, input_ids, hashes, history


class _SimpleInputIds:
    def __init__(self, ids: Sequence[int]):
        self.ids = list(int(t) for t in ids)
        self.shape = (1, len(self.ids))

    def __getitem__(self, item):
        if item is None:
            return self
        return self.ids[item]

    def __len__(self) -> int:
        return len(self.ids)


def _frontier_from_cache(live_cache: list[Any]) -> int:
    if not live_cache or len(live_cache) != 40:
        raise P6AppendError("P6 append requires one live 40-layer cache list")
    vals = []
    for cache in live_cache:
        size = getattr(cache, "size", None)
        if size is not None:
            vals.append(int(size()))
        else:
            v = cache[0]
            try:
                vals.append(int(v.item()))
            except Exception:
                vals.append(int(v))
    if any(v != vals[0] for v in vals):
        raise P6AppendError("public slot0 frontiers diverge")
    return vals[0]


def _set_cache_history(live_cache: list[Any], history: Any) -> None:
    if live_cache:
        current = live_cache[0][6]
        if hasattr(current, "shape") and not hasattr(history, "shape"):
            return
        live_cache[0][6] = history


def _canonical_public_frontier(frontier: int, language_model: Any) -> Any:
    value = _make_cache_offset(int(frontier), language_model)
    # Some lightweight test doubles expose cache_offset() but return a Python
    # scalar.  That is not the production DeepseekV41Cache slot0 contract: the
    # real cache size() path expects an array-like offset with shape (1,), int32
    # dtype and item() semantics.  Fall back to the same MLX construction used
    # by _make_cache_offset when no model factory exists; do not mutate existing
    # offset objects in place.
    if isinstance(value, int) or not (hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "item")):
        try:
            import mlx.core as mx
            value = mx.array([int(frontier)], mx.int32)
        except Exception:
            pass
    return value


def _set_all_public_frontiers(live_cache: list[Any], frontier: int, language_model: Any) -> None:
    for cache in live_cache:
        cache[0] = _canonical_public_frontier(frontier, language_model)
