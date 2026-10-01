"""Structural telemetry for the DwarfStar FP8/MLX prefill package.

Telemetry in P0/P1 is architectural bookkeeping only.  It must not be used as a
performance success/failure gate before the structural acceptance gate passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArchitectureCompletion:
    sweep_planner_owns_order: bool = False
    whole_prefix_layer_loop_absent: bool = False
    dwarfstar_carry_lifetime: bool = False
    deferred_decoder_suffix_lifetime: bool = False
    frontiers_drive_execution: bool = False
    scheduling_hooks_effective: bool = False
    materialization_boundaries_explicit: bool = False
    no_cpu_hot_path_roundtrip: bool = False
    no_intermediate_cache_repack: bool = False
    one_live_cache_handoff_no_replay: bool = False
    p7_full_resident_backbone_ssd_engram: bool = False
    p0_p7_structural_gate: bool = False

    @property
    def structural_gate_passed(self) -> bool:
        return all(self.to_json().values())

    def to_json(self) -> dict[str, bool]:
        return {
            "sweep_planner_owns_order": self.sweep_planner_owns_order,
            "whole_prefix_layer_loop_absent": self.whole_prefix_layer_loop_absent,
            "dwarfstar_carry_lifetime": self.dwarfstar_carry_lifetime,
            "deferred_decoder_suffix_lifetime": self.deferred_decoder_suffix_lifetime,
            "frontiers_drive_execution": self.frontiers_drive_execution,
            "scheduling_hooks_effective": self.scheduling_hooks_effective,
            "materialization_boundaries_explicit": self.materialization_boundaries_explicit,
            "no_cpu_hot_path_roundtrip": self.no_cpu_hot_path_roundtrip,
            "no_intermediate_cache_repack": self.no_intermediate_cache_repack,
            "one_live_cache_handoff_no_replay": self.one_live_cache_handoff_no_replay,
            "p7_full_resident_backbone_ssd_engram": self.p7_full_resident_backbone_ssd_engram,
            "p0_p7_structural_gate": self.p0_p7_structural_gate,
        }


@dataclass
class PrefillStructuralTelemetry:
    """Architecture counters and markers, intentionally excluding speed gates."""

    architecture_completion: ArchitectureCompletion = field(default_factory=ArchitectureCompletion)
    planner_summary: dict[str, Any] = field(default_factory=dict)
    guard_report: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "architecture_completion": self.architecture_completion.to_json(),
            "structural_gate_passed": self.architecture_completion.structural_gate_passed,
            "planner_summary": dict(self.planner_summary),
            "guard_report": dict(self.guard_report),
            "notes": list(self.notes),
            "performance_gate_allowed": self.architecture_completion.structural_gate_passed,
        }
