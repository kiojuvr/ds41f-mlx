"""Historical oMLX-compatibility bridge for the DwarfStar prefill graph.

This module is intentionally *not* an official DeepSeek semantics authority.
It is retained to describe the old adapter that executed DwarfStar-shaped graph
steps through oMLX operations.  Its outputs may be useful for compatibility or
regression diagnostics, but agreement with this contract must not be used as
official model correctness evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ds41f_mlx.m2_state_publication import build_publication_frontiers


@dataclass(frozen=True)
class SemanticsBinding:
    graph_step: str
    official_provider: str
    invariant: str
    donor_not_adopted: tuple[str, ...] = ()


@dataclass(frozen=True)
class OfficialPrefillSemanticsContract:
    model: str
    checkpoint_policy: str
    precision_policy: str
    decode_policy: str
    layer_count: int
    frontiers: dict[int, tuple[str, ...]]
    bindings: tuple[SemanticsBinding, ...]
    forbidden_shortcuts: tuple[str, ...] = field(
        default=(
            "Do not use DwarfStar GGUF/Q4/imatrix weights as correctness target.",
            "Do not quantize or rewrite the official checkpoint for this path.",
            "Do not replace oMLX DSpark/MTP decode behavior in this prefill milestone.",
            "Do not promote final-logits-only projection; tiny fixtures showed it is not exact.",
        )
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "historical_only_superseded",
            "classification": "TAINTED_OMLX_DERIVED_CORRECTNESS_IF_USED_AS_OFFICIAL_SEMANTICS",
            "authority_warning": "oMLX providers are compatibility references only; agreement with oMLX is not official DeepSeek correctness.",
            "model": self.model,
            "checkpoint_policy": self.checkpoint_policy,
            "precision_policy": self.precision_policy,
            "decode_policy": self.decode_policy,
            "layer_count": self.layer_count,
            "frontiers": {str(k): list(v) for k, v in self.frontiers.items()},
            "bindings": [
                {
                    "graph_step": b.graph_step,
                    "official_provider": b.official_provider,
                    "invariant": b.invariant,
                    "donor_not_adopted": list(b.donor_not_adopted),
                }
                for b in self.bindings
            ],
            "forbidden_shortcuts": list(self.forbidden_shortcuts),
        }


DEFAULT_BINDINGS = (
    SemanticsBinding(
        graph_step="upload_tokens",
        official_provider="oMLX Processor/tokenizer output ids (compatibility reference only)",
        invariant="token ids match historical accepted oMLX prompt encoding; not an official semantics proof",
    ),
    SemanticsBinding(
        graph_step="upload_embeddings_hc",
        official_provider="oMLX DeepSeek-V4.1 LanguageModel.embed + HC repeat/pre mask (compatibility reference only)",
        invariant="historical adapter embedding/HC behavior preserved against oMLX; not an official semantics proof",
        donor_not_adopted=("DwarfStar GGUF embedding layout",),
    ),
    SemanticsBinding(
        graph_step="prepare_layer_weights",
        official_provider="oMLX safetensors loader / MLX module weights (compatibility reference only)",
        invariant="checkpoint is read-only; no conversion or quantization in this adapter",
        donor_not_adopted=("DwarfStar layer-pack/GGUF streaming format", "Q2/Q4/imatrix layout"),
    ),
    SemanticsBinding(
        graph_step="encode_layer_batch",
        official_provider="oMLX DeepSeek-V4.1 Block.__call__ including Engram, Attention, HC, MoE/FFN (compatibility reference only)",
        invariant="all math, activation packing, kernel selection, and dtype behavior remain oMLX adapter behavior; not official semantics",
        donor_not_adopted=("DwarfStar Metal kernels", "DwarfStar quantized expert kernels"),
    ),
    SemanticsBinding(
        graph_step="capture_dspark_prefill_layer",
        official_provider="oMLX DSpark/MTP contract (compatibility reference only)",
        invariant="prefill may publish DSpark capture seam, but decode remains embedded oMLX DSpark/MTP unless explicitly gated",
    ),
    SemanticsBinding(
        graph_step="publish_state_frontier",
        official_provider="ExplicitStatePublication over oMLX shared-state/cache objects (compatibility reference only)",
        invariant="kv/index_k/idx/candidates/Engram history frontier digests exact against historical accepted oMLX path",
    ),
    SemanticsBinding(
        graph_step="seed_router_selected",
        official_provider="oMLX routed MoE gate/top-k and cache state (compatibility reference only)",
        invariant="historical oMLX routing decisions and dtype behavior preserved",
        donor_not_adopted=("DwarfStar expert-cache ownership policy as correctness source",),
    ),
    SemanticsBinding(
        graph_step="seed_streaming_expert_cache_layer",
        official_provider="no-op in current oMLX adapter; future buffer/cache optimization only after authority-classified gates",
        invariant="must not change routed expert outputs or final cache state in the historical oMLX compatibility adapter",
    ),
    SemanticsBinding(
        graph_step="select_output_row",
        official_provider="historical accepted oMLX full-chunk projection behavior; output retention may keep final chunk only",
        invariant="do not treat final-row projection mismatch against oMLX as official semantics evidence",
    ),
    SemanticsBinding(
        graph_step="encode_output_head",
        official_provider="oMLX RMSNorm + project_logits using checkpoint head weights (compatibility reference only)",
        invariant="last-chunk logits digest exact against historical accepted oMLX full-chunk projection",
    ),
    SemanticsBinding(
        graph_step="read_logits",
        official_provider="MLX array materialization / digest or greedy selection (compatibility reference only)",
        invariant="greedy token and gated logits digest exactness preserved against oMLX adapter behavior",
    ),
)


def build_official_prefill_semantics_contract(language_model: Any) -> OfficialPrefillSemanticsContract:
    config = language_model._config
    frontiers = build_publication_frontiers(config)
    return OfficialPrefillSemanticsContract(
        model="DeepSeek-V4.1-Flash",
        checkpoint_policy="official checkpoint read-only; historical adapter loads it through known-good oMLX loader",
        precision_policy="historical oMLX/MLX precision and activation behavior; not official semantics authority",
        decode_policy="decode remains oMLX-derived DSpark/MTP; compatibility only",
        layer_count=int(getattr(config, "n_layers", 40) or 40),
        frontiers={layer: frontier.publishes for layer, frontier in frontiers.items()},
        bindings=DEFAULT_BINDINGS,
    )
