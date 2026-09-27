#!/usr/bin/env python3
"""Inspect the oMLX DeepSeek-V4.1 runtime substrate without loading the model.

This records the native/custom kernel state and source-level startup deltas that
matter for Milestone 4.  It is intentionally a diagnostic/provenance tool; it
must not import the HTTP engine or start DSpark/MTP generation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import DEFAULT_OMLX

DEEPSEEK_ONE_TOKEN_SYMBOLS = (
    "deepseek_v41_grouped_expert",
    "deepseek_v41_packed_attention",
    "deepseek_mxfp4_gather_qmm_pair_concat_blocks",
    "deepseek_affine_gather_qmm_pair_concat_blocks",
    "deepseek_affine_gather_qmm_blocks",
)

SOURCE_FILES = (
    "omlx/patches/deepseek_v41/loading.py",
    "omlx/patches/deepseek_v41/language.py",
    "omlx/patches/deepseek_v41/quantization.py",
    "omlx/patches/deepseek_v41/packed_attention.py",
    "omlx/patches/deepseek_v41/kernels.py",
    "omlx/patches/deepseek_v41/hyper_connection.py",
    "omlx/patches/deepseek_v41/activation.py",
    "omlx/scheduler.py",
    "omlx/engine_core.py",
    "omlx/engine/batched.py",
    "omlx/patches/mlx_lm_mtp/batch_generator.py",
)


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def grep(path: Path, needles: tuple[str, ...]) -> list[dict]:
    out = []
    try:
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if any(n in line for n in needles):
                out.append({"line": i, "text": line.strip()})
    except FileNotFoundError:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    ap.add_argument("--out", default="artifacts/m4/omlx-runtime-substrate/inspect.json")
    args = ap.parse_args()
    omlx = Path(args.omlx).expanduser().resolve()
    sys.path.insert(0, str(omlx))

    rec: dict = {
        "schema": "ds41f.m4.omlx-runtime-substrate-inspection.v1",
        "omlx_path": str(omlx),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "native_required_for_practical_base_claim": True,
        "mtp_off_scope": True,
    }
    try:
        rec["omlx_git_revision"] = subprocess.check_output(["git", "-C", str(omlx), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        rec["omlx_git_error"] = repr(exc)

    try:
        import mlx.core as mx  # type: ignore
        rec["mlx_version"] = getattr(mx, "__version__", None)
        rec["mx_fast_deepseek_symbols"] = [
            name for name in dir(getattr(mx, "fast", object())) if "deepseek" in name or "dspark" in name
        ]
    except Exception as exc:
        rec["mlx_import_error"] = repr(exc)

    native = {}
    try:
        glm_fast = importlib.import_module("omlx.custom_kernels.glm_moe_dsa.fast")
        for name in ("is_native_available", "import_error", "native_symbols"):
            try:
                value = getattr(glm_fast, name)()
                native[name] = repr(value) if isinstance(value, BaseException) else value
            except Exception as exc:
                native[name] = {"error": repr(exc)}
        native["material_one_token_symbols"] = {
            sym: bool(glm_fast.has_symbol(sym)) for sym in DEEPSEEK_ONE_TOKEN_SYMBOLS
        }
    except Exception as exc:
        native["module_import_error"] = repr(exc)
    ext_files = sorted(str(p.relative_to(omlx)) for p in (omlx / "omlx/custom_kernels/glm_moe_dsa").glob("_ext*.so"))
    native["extension_files"] = ext_files
    native["diagnosis"] = (
        "extension_not_built_or_not_installed_into_this_tree"
        if not ext_files and not native.get("is_native_available")
        else "see import_error/native_symbols"
    )
    rec["glm_moe_dsa_fast"] = native

    source = {}
    needles = ("glm_fast", "deepseek_v41_", "GenerationBatch", "preserve_mtp", "make_mtp", "dspark", "patch", "native")
    for rel in SOURCE_FILES:
        p = omlx / rel
        source[rel] = {"sha256": sha256(p), "matches": grep(p, needles)[:80]}
    rec["source_review"] = source
    rec["direct_loader_vs_engine_initialization"] = {
        "direct_loader_used_by_ds41f": "omlx.patches.deepseek_v41.loading.load(checkpoint, engram_ssd_offload, preserve_mtp, moe_expert_offload_resident_fraction)",
        "normal_engine_additional_work_observed_by_static_inspection": [
            "scheduler/GenerationBatch orchestration and prompt replay/prefill chunking",
            "MTP/DSpark batch-generator path when enabled/preserve_mtp permits it",
            "scheduler warning/check for missing native deepseek_v41_grouped_expert",
            "prompt boundary/cache-store logic not relevant to no-replay admitted production C/D",
        ],
        "no_static_evidence_of_extra_model_math_patch_required_for_mtp_off_raw_forward": True,
        "native_extension_availability_is_material": True,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
