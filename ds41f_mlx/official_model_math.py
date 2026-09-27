"""Production-facing official model-math seam for DwarfStar prefill work.

This module deliberately exposes a narrow runtime-facing interface around the
already reviewed official-source-derived arithmetic helpers.  It avoids having
the DwarfStar executor import validation script orchestration directly while the
underlying arithmetic is progressively moved into native production components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Existing helpers are official-source-derived validation helpers, not oMLX
# runtime code and not TextBackboneReference/TextEncoderReference control flow.
from tools.run_native_layer0_25_transformer_entry_validation import (  # type: ignore
    DIM,
    HC,
    VOCAB,
    block,
    cfg as load_text_config,
    digest,
    mmap,
    snap,
)
from tools.run_native_layer24_25_connected_validation import roles as layer_roles  # type: ignore


class OfficialModelMath:
    """Narrow model-math provider for bounded prefill executor slices."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint)
        self.config = load_text_config(self.checkpoint)

    @property
    def dim(self) -> int:
        return DIM

    @property
    def hc_mult(self) -> int:
        return HC

    @property
    def vocab_size(self) -> int:
        return VOCAB

    def digest(self, value: Any) -> str:
        return digest(value)

    def snapshot_publications(self, shared: dict[str, Any]) -> dict[str, str | None]:
        return snap(shared)

    def embedding_prefix(self, rows: int) -> np.ndarray:
        return np.ascontiguousarray(
            mmap(self.checkpoint / "model-00002-of-00048.safetensors", "embed.weight", np.uint16, (VOCAB, DIM))[:rows]
        )

    def execute_block(self, layer: int, x_hc_bf16: np.ndarray, pre_f32: np.ndarray, shared: dict[str, Any]) -> dict[str, Any]:
        return block(self.checkpoint, self.config, layer, x_hc_bf16, pre_f32, shared)

    def layer_roles(self, layer: int) -> dict[str, Any]:
        return layer_roles(self.config, layer)
