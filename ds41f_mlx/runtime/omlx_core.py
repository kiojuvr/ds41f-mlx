"""Thin M0 bridge to the local known-good oMLX DeepSeek-V4.1 runtime.

This module intentionally does not reimplement or optimize the model.  M0 uses
it to make the oMLX execution substrate explicit and provenance-recorded before
native optimization work begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import sys
from time import perf_counter

from ds41f_mlx.config import DEFAULT_CHECKPOINT, DEFAULT_OMLX, DEFAULT_RECIPE


@dataclass(frozen=True)
class OmlxRuntimeConfig:
    omlx_path: Path = DEFAULT_OMLX
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    engram_ssd_offload: bool = True
    preserve_mtp: bool | None = None
    moe_expert_offload_resident_fraction: float | None = None
    recipe_path: Path = DEFAULT_RECIPE


class OmlxRuntime:
    """Lazy wrapper around `omlx.patches.deepseek_v41.loading.load`.

    The first implementation goal is direct reproduction, so this wrapper keeps
    oMLX ownership of compatibility behavior, cache layout, and MLX execution
    topology. It is not an official DeepSeek semantics authority.
    """

    def __init__(self, config: OmlxRuntimeConfig | None = None):
        self.config = config or OmlxRuntimeConfig()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.admission = None
        self.admission_seconds = self.binding_seconds = 0.0

    def ensure_import_path(self) -> None:
        root = str(self.config.omlx_path)
        if root not in sys.path:
            sys.path.insert(0, root)

    def load_model(self):
        """Load the official checkpoint through oMLX for compatibility/performance diagnostics.

        This may allocate hundreds of GiB and should only be called by explicit
        smoke/performance runners, not at module import time.
        """
        self.ensure_import_path()
        loading = importlib.import_module("omlx.patches.deepseek_v41.loading")
        # Bounded MTP/diagnostic offload loading remains on its separate path.
        if (self.config.preserve_mtp is False and self.config.engram_ssd_offload
                and self.config.moe_expert_offload_resident_fraction is None):
            from .resource_admission import prepare_resources
            start = perf_counter()
            admission = prepare_resources(self.config.checkpoint_path, self.config.recipe_path)
            self.admission_seconds = perf_counter() - start
            self.admission = admission
            loading = admission.modules['omlx.patches.deepseek_v41.loading']
        try:
            self.model, self.processor = loading.load(
                self.admission._checkpoint if self.admission else self.config.checkpoint_path,
                engram_ssd_offload=self.config.engram_ssd_offload,
                preserve_mtp=self.config.preserve_mtp,
                moe_expert_offload_resident_fraction=self.config.moe_expert_offload_resident_fraction,
            )
            if self.admission is not None:
                start = perf_counter()
                self.admission.bind_model(self.model, self.processor)
                self.binding_seconds = perf_counter() - start
        except BaseException:
            self.close()
            raise
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        return self.model, self.processor

    def close(self) -> None:
        if self.admission is not None:
            self.admission.retire()
        model = self.model
        if model is not None:
            close = getattr(model, "close", None)
            if close is not None:
                close()
        self.model = None
        self.processor = None
        self.tokenizer = None
