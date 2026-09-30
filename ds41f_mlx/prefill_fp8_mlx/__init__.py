"""DwarfStar V4.1 prefill architecture package for official FP8/MLX.

P0-P2 exports guards, planner normalization, request-arena ownership, and
structural telemetry.  It does not implement runtime block execution and does
not alter the production serving selector.
"""

from ds41f_mlx.prefill_fp8_mlx.arena import (
    AllocationState,
    ArenaTransactionError,
    ArenaView,
    CarryState,
    CompressorPendingState,
    EngramState,
    PublicationEvent,
    PublicationState,
    RequestArena,
    TensorOwnership,
    TensorSlot,
    TransactionState,
)
from ds41f_mlx.prefill_fp8_mlx.guards import (
    FORBIDDEN_HOT_PATH_MODULES,
    REFERENCE_VERTICAL_SLICE_CLASSIFICATION,
    GuardReport,
    assert_no_reference_hot_path,
    check_no_reference_hot_path,
)
from ds41f_mlx.prefill_fp8_mlx.planner import (
    ResumeUnavailableError,
    SemanticBoundary,
    SweepAllocation,
    SweepCommand,
    SweepCommandKind,
    SweepPhase,
    SweepPlan,
    SweepPlanner,
    SweepTopologyError,
    UnsupportedTopologyError,
    assert_deepseek_v41_topology,
    iter_command_batches,
    iter_command_windows,
)
from ds41f_mlx.prefill_fp8_mlx.telemetry import (
    ArchitectureCompletion,
    PrefillStructuralTelemetry,
)

__all__ = [
    "AllocationState",
    "ArchitectureCompletion",
    "ArenaTransactionError",
    "ArenaView",
    "CarryState",
    "CompressorPendingState",
    "EngramState",
    "FORBIDDEN_HOT_PATH_MODULES",
    "GuardReport",
    "PrefillStructuralTelemetry",
    "PublicationEvent",
    "PublicationState",
    "REFERENCE_VERTICAL_SLICE_CLASSIFICATION",
    "RequestArena",
    "ResumeUnavailableError",
    "SemanticBoundary",
    "SweepAllocation",
    "SweepCommand",
    "SweepCommandKind",
    "SweepPhase",
    "SweepPlan",
    "SweepPlanner",
    "SweepTopologyError",
    "TensorOwnership",
    "TensorSlot",
    "TransactionState",
    "UnsupportedTopologyError",
    "assert_deepseek_v41_topology",
    "assert_no_reference_hot_path",
    "check_no_reference_hot_path",
    "iter_command_batches",
    "iter_command_windows",
]
