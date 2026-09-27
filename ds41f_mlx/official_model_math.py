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
from tools.run_native_engram_layer1_validation import layer1_hashes_regenerate  # type: ignore
from tools.run_native_engram_layer14_validation import apply_engram_layer, regen_hashes  # type: ignore


class OfficialModelMath:
    """Narrow model-math provider for bounded prefill executor slices."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint)
        self.config = load_text_config(self.checkpoint)
        self._engram_contract_cache: dict[str, Any] | None = None

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

    def engram_hashes_for_tokens(self, tokens: np.ndarray) -> dict[str, Any]:
        # Current bounded implementation is for the official text fixture used by
        # existing official-source-derived Engram contracts.
        if tokens.reshape(-1).tolist() != [0, 3]:
            raise ValueError("bounded Engram seam currently supports token fixture [0, 3]")
        full_hash, layer1_hash, provenance = layer1_hashes_regenerate()
        try:
            import json
            contract = self._engram_contract()
            cfg_infer = json.loads((self.checkpoint / "inference/config.json").read_text())
            regen_full, regen_layer1, layer14_hash = regen_hashes(self.checkpoint, cfg_infer, contract)
            if digest(regen_full) != digest(full_hash) or digest(regen_layer1) != digest(layer1_hash):
                raise RuntimeError("layer14 hash regeneration disagrees with layer1 Engram contract")
        except Exception:
            raise
        return {
            "full_hash": full_hash,
            "layer1_hash": layer1_hash,
            "layer14_hash": layer14_hash,
            "provenance": {**provenance, "layer14_regenerated_from_same_full_hash": True},
            "full_hash_digest": digest(full_hash),
            "layer1_hash_digest": digest(layer1_hash),
            "layer14_hash_digest": digest(layer14_hash),
        }

    def _engram_contract(self) -> dict[str, Any]:
        if self._engram_contract_cache is None:
            import json
            self._engram_contract_cache = json.loads(Path("artifacts/engram-semantic-foundation-contract.json").read_text())
        return self._engram_contract_cache

    def apply_engram(self, layer: int, x_hc_bf16: np.ndarray, layer_hash: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        post, sparse, flatten, wkv, kv, qk, gate, residual, io = apply_engram_layer(
            self.checkpoint, layer, x_hc_bf16, layer_hash, self._engram_contract()
        )
        evidence = {
            "layer": layer,
            "input_digest": digest(x_hc_bf16),
            "hash_digest": digest(layer_hash),
            "output_digest": digest(post),
            "sparse_embedding": sparse,
            "flatten_seam": flatten,
            "wkv": wkv,
            "key_value_split": kv,
            "qk_weights": qk,
            "gate": gate,
            "residual_update": residual,
            "io_accounting": io,
            "ssd_backed_sparse_rows_only": bool(sparse.get("sparse_random_access_rows_only")),
        }
        return post, evidence
