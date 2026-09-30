"""DwarfStar V4.1 prefill architecture package for official FP8/MLX.

P0-P5 exports guards, planner normalization, request-arena ownership,
command-driven block execution, publication management, and one-shot live-cache
handoff. It does not alter the production serving selector.
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
from ds41f_mlx.prefill_fp8_mlx.block_runner import (
    BlockExecutionError,
    CommandExecutionRecord,
    MlxEvaluationPolicy,
    OfficialFP8MLXBlockRunner,
    ServingLogitsPolicy,
)
from ds41f_mlx.prefill_fp8_mlx.executor import (
    DwarfStarFP8MLXPrefillExecutorSetup,
    LivePrefillContinuation,
    PrefillExecutionSetup,
    PrefillSetupError,
)
from ds41f_mlx.prefill_fp8_mlx.handoff import (
    LiveCacheHandoffError,
    LivePrefillResult,
    handoff_to_generation,
    validate_committed_cache,
)
from ds41f_mlx.prefill_fp8_mlx.guards import (
    FORBIDDEN_HOT_PATH_MODULES,
    REFERENCE_VERTICAL_SLICE_CLASSIFICATION,
    GuardReport,
    assert_no_reference_hot_path,
    check_no_reference_hot_path,
)
from ds41f_mlx.prefill_fp8_mlx.omlx_suffix_math import (
    OmlxV41SuffixMath,
    SuffixMathError,
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
from ds41f_mlx.prefill_fp8_mlx.publications import (
    PublicationError,
    PublicationManager,
    PublicationTopology,
    SourceGeneration,
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
    "BlockExecutionError",
    "CarryState",
    "CommandExecutionRecord",
    "CompressorPendingState",
    "DwarfStarFP8MLXPrefillExecutorSetup",
    "EngramState",
    "FORBIDDEN_HOT_PATH_MODULES",
    "GuardReport",
    "LivePrefillContinuation",
    "LiveCacheHandoffError",
    "LivePrefillResult",
    "handoff_to_generation",
    "validate_committed_cache",
    "MlxEvaluationPolicy",
    "OfficialFP8MLXBlockRunner",
    "OmlxV41SuffixMath",
    "PrefillExecutionSetup",
    "PrefillSetupError",
    "PrefillStructuralTelemetry",
    "PublicationError",
    "PublicationEvent",
    "PublicationManager",
    "PublicationState",
    "PublicationTopology",
    "REFERENCE_VERTICAL_SLICE_CLASSIFICATION",
    "RequestArena",
    "ResumeUnavailableError",
    "SemanticBoundary",
    "ServingLogitsPolicy",
    "SourceGeneration",
    "SuffixMathError",
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
