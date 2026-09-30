"""Production prefill guards for the DwarfStar FP8/MLX package.

These guards are intentionally lightweight and import-safe.  They do not start
model execution; they protect the future production path from silently falling
back to reference/validation modules while P0/P1 scaffolding is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Iterable

REFERENCE_VERTICAL_SLICE_CLASSIFICATION = "PRODUCTION_PREFILL_REGRESSED_TO_REFERENCE_VERTICAL_SLICE"

FORBIDDEN_HOT_PATH_MODULES: tuple[str, ...] = (
    "ds41f_mlx.dwarfstar_prefill_slice",
    "ds41f_mlx.official_model_math",
    "tools.run_m4_omlx_base_decode_qualification",
    "tools.run_native_layer0_25_transformer_entry_validation",
)


@dataclass(frozen=True)
class GuardReport:
    """Result of a production-hot-path module check."""

    ok: bool
    loaded_forbidden_modules: tuple[str, ...]
    classification: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "loaded_forbidden_modules": list(self.loaded_forbidden_modules),
            "classification": self.classification,
        }


def check_no_reference_hot_path(*, extra_forbidden: Iterable[str] = ()) -> GuardReport:
    """Return whether forbidden reference modules are currently loaded."""

    forbidden = tuple(FORBIDDEN_HOT_PATH_MODULES) + tuple(extra_forbidden)
    loaded = tuple(name for name in forbidden if name in sys.modules)
    return GuardReport(
        ok=not loaded,
        loaded_forbidden_modules=loaded,
        classification=None if not loaded else REFERENCE_VERTICAL_SLICE_CLASSIFICATION,
    )


def assert_no_reference_hot_path(*, extra_forbidden: Iterable[str] = ()) -> None:
    """Raise if the production prefill hot path has touched reference modules."""

    report = check_no_reference_hot_path(extra_forbidden=extra_forbidden)
    if not report.ok:
        raise RuntimeError(
            f"{REFERENCE_VERTICAL_SLICE_CLASSIFICATION}: forbidden reference modules loaded on "
            f"production DwarfStar FP8/MLX prefill hot path: {list(report.loaded_forbidden_modules)}"
        )
