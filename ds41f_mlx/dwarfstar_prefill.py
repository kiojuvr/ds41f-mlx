"""DwarfStar-authoritative prefill engine scaffold for DeepSeek-V4.1.

This module changes the M2 direction: DwarfStar's prefill graph is treated as
the architecture authority.  The current executable adapter still uses oMLX
operations for compatibility/regression diagnostics, but oMLX is not an official
DeepSeek semantics/logits/cache/state authority.

The first engine implementation is intentionally an adapter: it builds an
explicit DwarfStar-derived prefill plan, then executes it through the existing
oMLX-operation layer-major prototype.  This creates the new prefill/decode seam
without importing DwarfStar kernels.  Its logits/cache comparisons are
historical oMLX-compatibility evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from ds41f_mlx.dwarfstar_semantics import build_official_prefill_semantics_contract
from ds41f_mlx.m2_layer_major import layer_major_forward
from ds41f_mlx.m2_state_publication import build_publication_frontiers, cache_digest
from ds41f_mlx.native_prefill import NativePrefillLibrary, native_config_from_model

PrefillStepKind = Literal[
    "upload_tokens",
    "upload_embeddings_hc",
    "begin_layer_commands",
    "prepare_layer_weights",
    "encode_layer_batch",
    "capture_dspark_prefill_layer",
    "publish_state_frontier",
    "seed_router_selected",
    "seed_streaming_expert_cache_layer",
    "end_layer_commands",
    "release_prefill_mask_cache",
    "seed_streaming_expert_cache_from_hotlist",
    "seed_streaming_expert_cache_from_prefill",
    "select_output_row",
    "encode_output_head",
    "read_logits",
]


@dataclass(frozen=True)
class DwarfStarPrefillStep:
    kind: PrefillStepKind
    layer: int | None = None
    note: str = ""
    publishes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DwarfStarPrefillPlan:
    """Layer-major prefill plan derived from DwarfStar graph topology."""

    model: str
    n_layers: int
    start: int
    n_tokens: int
    chunk_lengths: tuple[int, ...]
    output_semantics: str
    steps: tuple[DwarfStarPrefillStep, ...]
    authority: tuple[str, ...] = field(
        default=(
            "$HOME/ds4/ds4.c: metal_graph_prefill_layer_major",
            "$HOME/ds4/ds4.c: metal_graph_encode_layer_batch",
            "$HOME/ds4/ds4.c: metal_graph_dspark_capture_prefill_layer",
            "$HOME/ds4/ds4.c: metal_graph_capture_prefill_seed_router_selected",
            "$HOME/ds4/ds4.c: metal_graph_encode_output_head",
        )
    )

    @classmethod
    def for_deepseek_v41(
        cls,
        config: Any,
        *,
        start: int,
        chunk_lengths: list[int],
        output_semantics: str = "full_chunks_last",
    ) -> "DwarfStarPrefillPlan":
        frontiers = build_publication_frontiers(config)
        n_layers = int(getattr(config, "n_layers", 40) or 40)
        steps: list[DwarfStarPrefillStep] = [
            DwarfStarPrefillStep("upload_tokens", note="metal_graph_upload_prompt_tokens"),
            DwarfStarPrefillStep("upload_embeddings_hc", note="metal_graph_upload_prompt_embeddings_hc"),
        ]
        for layer in range(n_layers):
            frontier = frontiers.get(layer)
            steps.extend(
                [
                    DwarfStarPrefillStep("begin_layer_commands", layer=layer),
                    DwarfStarPrefillStep("prepare_layer_weights", layer=layer, note="stream/map/readahead layer weights when applicable"),
                    DwarfStarPrefillStep("encode_layer_batch", layer=layer, note="attention + FFN using historical oMLX adapter math; compatibility only"),
                    DwarfStarPrefillStep("capture_dspark_prefill_layer", layer=layer, note="preserve DSpark/MTP capture seam"),
                    DwarfStarPrefillStep(
                        "publish_state_frontier",
                        layer=layer,
                        publishes=frontier.publishes if frontier else (),
                    ),
                    DwarfStarPrefillStep("seed_router_selected", layer=layer),
                    DwarfStarPrefillStep("seed_streaming_expert_cache_layer", layer=layer),
                    DwarfStarPrefillStep("end_layer_commands", layer=layer),
                ]
            )
        steps.extend(
            [
                DwarfStarPrefillStep("release_prefill_mask_cache"),
                DwarfStarPrefillStep("seed_streaming_expert_cache_from_hotlist"),
                DwarfStarPrefillStep("seed_streaming_expert_cache_from_prefill"),
                DwarfStarPrefillStep("select_output_row", note="DwarfStar selects final row before output head"),
                DwarfStarPrefillStep("encode_output_head"),
                DwarfStarPrefillStep("read_logits"),
            ]
        )
        return cls(
            model="DeepSeek-V4.1",
            n_layers=n_layers,
            start=start,
            n_tokens=sum(chunk_lengths),
            chunk_lengths=tuple(chunk_lengths),
            output_semantics=output_semantics,
            steps=tuple(steps),
        )

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        frontiers: dict[str, list[str]] = {}
        for step in self.steps:
            by_kind[step.kind] = by_kind.get(step.kind, 0) + 1
            if step.kind == "publish_state_frontier" and step.publishes:
                frontiers[str(step.layer)] = list(step.publishes)
        return {
            "model": self.model,
            "n_layers": self.n_layers,
            "start": self.start,
            "n_tokens": self.n_tokens,
            "chunk_lengths": list(self.chunk_lengths),
            "output_semantics": self.output_semantics,
            "step_counts": by_kind,
            "publication_frontiers": frontiers,
            "authority": list(self.authority),
        }


@dataclass
class DwarfStarPrefillResult:
    logits: Any
    plan: DwarfStarPrefillPlan
    seconds: float
    cache_digest: list[dict[str, Any]]
    publication_records: list[list[dict[str, Any]]]

    def summary(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "tokens_per_second": self.plan.n_tokens / self.seconds if self.seconds > 0 else None,
            "plan": self.plan.summary(),
            "cache_digest": self.cache_digest,
        }


class DwarfStarPrefillEngine:
    """New prefill-engine seam with DwarfStar graph as architecture authority.

    This adapter executes through oMLX operations today.  Future implementations
    should replace this adapter's internals with static carry buffers, explicit
    command ownership, and eventually custom kernels while keeping the plan and
    prefill/decode seam stable.
    """

    def __init__(self, language_model: Any):
        self.language_model = language_model

    def semantics_contract(self) -> dict[str, Any]:
        return build_official_prefill_semantics_contract(self.language_model).to_json()

    def plan(self, chunk_ids: list[list[int]], *, output_semantics: str = "full_chunks_last") -> DwarfStarPrefillPlan:
        return DwarfStarPrefillPlan.for_deepseek_v41(
            self.language_model._config,
            start=0,
            chunk_lengths=[len(c) for c in chunk_ids],
            output_semantics=output_semantics,
        )

    def native_plan(self, chunk_ids: list[list[int]], native: NativePrefillLibrary) -> dict[str, Any]:
        cfg = native_config_from_model(self.language_model, [len(c) for c in chunk_ids])
        return native.build_plan(cfg).to_json()

    def prefill(
        self,
        chunk_ids: list[list[int]],
        cache: Any,
        *,
        output_semantics: Literal["concat", "full_chunks_last"] = "full_chunks_last",
        record_publications: bool = False,
    ) -> DwarfStarPrefillResult:
        plan = self.plan(chunk_ids, output_semantics=output_semantics)
        t0 = perf_counter()
        logits, records = layer_major_forward(
            self.language_model,
            chunk_ids,
            cache,
            final_logits_only=False,
            record_publications=record_publications,
            eval_boundary="none",
            output_mode=output_semantics,
        )
        seconds = perf_counter() - t0
        return DwarfStarPrefillResult(
            logits=logits,
            plan=plan,
            seconds=seconds,
            cache_digest=[cache_digest(c) for c in cache],
            publication_records=records,
        )
