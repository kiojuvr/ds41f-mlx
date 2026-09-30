"""DwarfStar V4.1 prefill architecture package for official FP8/MLX.

P0/P1 exports only guards, planner normalization, and structural telemetry.  It
does not implement runtime block execution and does not alter the production
serving selector.
"""

from ds41f_mlx.prefill_fp8_mlx.guards import (
    FORBIDDEN_HOT_PATH_MODULES,
    REFERENCE_VERTICAL_SLICE_CLASSIFICATION,
    GuardReport,
    assert_no_reference_hot_path,
    check_no_reference_hot_path,
)
from ds41f_mlx.prefill_fp8_mlx.planner import (
    SweepAllocation,
    SweepCommand,
    SweepCommandKind,
    SweepPhase,
    SweepPlan,
    SweepPlanner,
    iter_command_windows,
)
from ds41f_mlx.prefill_fp8_mlx.telemetry import (
    ArchitectureCompletion,
    PrefillStructuralTelemetry,
)

__all__ = [
    "ArchitectureCompletion",
    "FORBIDDEN_HOT_PATH_MODULES",
    "GuardReport",
    "PrefillStructuralTelemetry",
    "REFERENCE_VERTICAL_SLICE_CLASSIFICATION",
    "SweepAllocation",
    "SweepCommand",
    "SweepCommandKind",
    "SweepPhase",
    "SweepPlan",
    "SweepPlanner",
    "assert_no_reference_hot_path",
    "check_no_reference_hot_path",
    "iter_command_windows",
]
