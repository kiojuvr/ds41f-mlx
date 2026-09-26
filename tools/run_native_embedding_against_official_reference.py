#!/usr/bin/env python3
"""Validate native BF16 embedding gather against official ParallelEmbedding fixture.

This uses the existing native embedding gather primitive and compares its output
to `artifacts/parallel-embedding-official-reference-fixture.json`, which is
independently derived from the reviewed official `ParallelEmbedding.forward`
contract. It does not execute oMLX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.native_prefill import (  # noqa: E402
    DS4_AUTHORITY_REMOTE,
    DS4_AUTHORITY_SHA,
    compile_native_prefill_library,
    load_native_prefill_library,
)

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_REFERENCE = "artifacts/parallel-embedding-official-reference-fixture.json"
EMBED_SHARD = "model-00002-of-00048.safetensors"
VOCAB_SIZE = 129280
FULL_DIM = 5120


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def safetensor_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def tensor_memmap(path: Path, name: str, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    header, base = safetensor_header(path)
    meta = header[name]
    begin, _end = meta["data_offsets"]
    return np.memmap(path, mode="r", dtype=dtype, offset=base + begin, shape=shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/native-embedding-official-reference-validation.json")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    if ref.get("classification") != "official_reference_derived" or ref.get("not_omlx_derived") is not True:
        raise ValueError("reference fixture is not official_reference_derived/not_omlx_derived")
    if ref.get("official_reference", {}).get("function_or_class") != "ParallelEmbedding.forward":
        raise ValueError("reference fixture is not for ParallelEmbedding.forward")

    tokens = np.array(ref["inputs"]["tokens"], dtype=np.int32)
    dim = int(ref["operation_contract"]["output_shape"][1])
    if dim <= 0 or dim > FULL_DIM:
        raise ValueError("reference fixture dim out of range")
    expected_digest = ref["digests"]["expected_output_bf16_bits_sha256"]

    ckpt = Path(args.checkpoint)
    shard = ckpt / EMBED_SHARD
    weight = tensor_memmap(shard, "embed.weight", np.uint16, (VOCAB_SIZE, FULL_DIM))
    bounded_weight = np.ascontiguousarray(weight[:, :dim], dtype=np.uint16)

    native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
    out_np, native_result = native.official_embedding_gather_bf16(bounded_weight, tokens)
    native_digest = digest(out_np)
    matches = native_digest == expected_digest

    record: dict[str, Any] = {
        "schema": "ds41f.native-embedding-official-reference-validation.v1",
        "classification": "official_reference_derived_native_validation",
        "not_omlx_derived": True,
        "purpose": "validate existing native BF16 embedding gather against independently derived official ParallelEmbedding.forward fixture",
        "checkpoint": str(ckpt),
        "reference_fixture": args.reference,
        "official_reference": ref["official_reference"],
        "source_tensor": {
            "name": "embed.weight",
            "shard": str(shard),
            "shape": [VOCAB_SIZE, FULL_DIM],
            "dtype": "BF16 raw uint16 bits",
            "bounded_dim": dim,
            "bounded_weight_bf16_bits_sha256": digest(bounded_weight),
        },
        "inputs": {
            "tokens": [int(x) for x in tokens.tolist()],
            "tokens_sha256": ref["inputs"]["tokens_sha256"],
        },
        "native_version": native.version(),
        "ds4_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
        "native_result": native_result,
        "digests": {
            "reference_expected_output_bf16_bits_sha256": expected_digest,
            "native_output_bf16_bits_sha256": native_digest,
        },
        "comparison": {
            "bit_exact": matches,
            "mismatch_count": 0 if matches else None,
        },
        "semantic_status": {
            "official_reference_fixture_used": True,
            "native_embedding_validated_against_parallel_embedding_contract": matches,
            "model_semantics_validated": False,
            "validated_stage": "ParallelEmbedding.forward BF16 gather/all_reduce-equivalent output for bounded tokens/dim only",
        },
        "non_claims": [
            "does not execute oMLX",
            "does not validate tokenizer behavior",
            "does not validate HC repeat/pre-mask, attention, MoE, Engram, DSpark/MTP, logits, cache, or full prefill",
            "does not add or modify native Metal code",
        ],
    }
    record["gates"] = {
        "reference_is_official_reference_derived": ref.get("classification") == "official_reference_derived",
        "reference_not_omlx_derived": ref.get("not_omlx_derived") is True,
        "native_metal_executed": native_result.get("metal_enabled") is True and native_result.get("metal_command_buffers") == 1,
        "shape_exact": native_result.get("n_tokens") == int(tokens.size) and native_result.get("dim") == dim,
        "native_matches_official_reference_fixture": matches,
        "full_model_semantics_not_claimed": record["semantic_status"]["model_semantics_validated"] is False,
    }
    record["ok"] = all(record["gates"].values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
