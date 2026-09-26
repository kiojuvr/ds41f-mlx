#!/usr/bin/env python3
"""Check imported native core has no active old-repo dependency.

This is a source/build provenance checker only. It does not execute model math.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
FORBIDDEN = [
    "/Volumes/SDXC-512/deepseek-v41-flash-mlx",
    "deepseek-v41-flash-mlx/",
    "../deepseek-v41-flash-mlx",
]
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".c", ".cc", ".mm", ".metal", ".in", ".txt"}


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def main() -> int:
    if not NATIVE.exists():
        fail("native/ does not exist")
    checked = 0
    for path in NATIVE.rglob("*"):
        if not path.is_file():
            continue
        if "build" in path.parts:
            continue
        if path.name == "CMakeLists.txt" or path.suffix in SOURCE_SUFFIXES:
            text = path.read_text(errors="ignore")
            checked += 1
            for token in FORBIDDEN:
                if token in text:
                    fail(f"old repo dependency string {token!r} in {path.relative_to(ROOT)}")
    policy = NATIVE / "include/dsv41/execution_policy.hpp"
    text = policy.read_text()
    for disabled in [
        "DSV41_RUNTIME_PACKED_CHUNK_ATTENTION is disabled in ds41f native import",
        "DSV41_RUNTIME_WIDE_ATTENTION is disabled in ds41f native import",
    ]:
        if disabled not in text:
            fail(f"missing rejected attention selector guard: {disabled}")
    cmake = (NATIVE / "CMakeLists.txt").read_text()
    for forbidden_target in ["context-ladder", "wide-attention", "packed-moe-probe", "benchmark"]:
        if forbidden_target in cmake:
            fail(f"benchmark/diagnostic target leaked into native CMake: {forbidden_target}")
    required_phase3 = [
        "src/mhc/reference.cpp",
        "src/moe/reference.cpp",
        "src/moe/expert_backing.cpp",
        "src/engram/mlx.cpp",
        "src/engram/layer.cpp",
        "src/model/block.cpp",
        "src/model/compressed_block.cpp",
        "src/model/reused_block.cpp",
        "src/model/text_encoder.cpp",
        "src/model/text_decoder.cpp",
        "src/model/text_backbone.cpp",
        "src/model/text_generate.cpp",
        "src/model/sampling.cpp",
        "src/runtime/residency.cpp",
    ]
    for rel in required_phase3:
        if not (NATIVE / rel).exists():
            fail(f"missing Phase 3 native model core source: native/{rel}")
    for required_target in ["dsv41_model_mlx", "dsv41_text_pair_mlx", "dsv41_engram_mlx"]:
        if required_target not in cmake:
            fail(f"missing Phase 3 build target: {required_target}")
    print(
        f"PASS native import dependency check: {checked} files scanned; "
        "old repo production source dependency=0; old repo native build dependency=0; "
        "old repo imported-test dependency=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
