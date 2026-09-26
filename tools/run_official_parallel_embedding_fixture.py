#!/usr/bin/env python3
"""Generate an official-reference-derived ParallelEmbedding fixture.

This models the official `ParallelEmbedding.forward` source contract using raw
official checkpoint BF16 bits. It does not import or execute oMLX, PyTorch, MLX,
or native Metal code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
MODEL_PY_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
FORWARD_SHA = "fc448dd2ac5f8a483c1ed42264e8ac1e0ba12e3bac3a073c12aae28db406f4bb"
EMBED_SHARD = "model-00002-of-00048.safetensors"
VOCAB_SIZE = 129280
DIM = 5120


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


def parse_tokens(s: str) -> np.ndarray:
    toks = np.array([int(x) for x in s.split(",") if x.strip()], dtype=np.int64)
    if toks.size == 0:
        raise ValueError("at least one token is required")
    if int(toks.min()) < 0 or int(toks.max()) >= VOCAB_SIZE:
        raise ValueError("token out of embed.weight range")
    return toks


def sharded_parallel_embedding(weight: np.ndarray, ids: np.ndarray, *, world_size: int, dim: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if VOCAB_SIZE % world_size != 0:
        raise ValueError("vocab size must be divisible by world_size")
    part = VOCAB_SIZE // world_size
    # Official reference returns the all_reduce sum. For raw BF16 gather, one rank
    # contributes the real row for each id and all other ranks contribute zeros.
    reduced = np.zeros((ids.size, dim), dtype=np.uint16)
    rank_records: list[dict[str, Any]] = []
    for rank in range(world_size):
        start = rank * part
        end = start + part
        mask = (ids < start) | (ids >= end)
        local_ids = ids - start
        local_ids = local_ids.copy()
        local_ids[mask] = 0
        local_weight = weight[start:end, :dim]
        local_y = np.ascontiguousarray(local_weight[local_ids], dtype=np.uint16)
        local_y[mask] = 0
        # Safe raw-bit equivalent for this fixture: exactly one rank is nonzero
        # for each token row, so OR of BF16 bit patterns equals the distributed
        # numeric all_reduce result without executing a framework runtime.
        reduced = np.bitwise_or(reduced, local_y)
        rank_records.append(
            {
                "rank": rank,
                "vocab_start_idx": start,
                "vocab_end_idx": end,
                "masked_positions": [int(i) for i in np.flatnonzero(mask).tolist()],
                "local_ids_after_mask": [int(x) for x in local_ids.tolist()],
                "local_output_bf16_bits_sha256": digest(local_y),
            }
        )
    return np.ascontiguousarray(reduced, dtype=np.uint16), rank_records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--tokens", default="0,3,16159,16160,32319,32320,129279")
    ap.add_argument("--out", default="artifacts/parallel-embedding-official-reference-fixture.json")
    args = ap.parse_args()

    if args.dim <= 0 or args.dim > DIM:
        raise ValueError("dim out of range")
    ckpt = Path(args.checkpoint)
    shard = ckpt / EMBED_SHARD
    ids = parse_tokens(args.tokens)
    weight = tensor_memmap(shard, "embed.weight", np.uint16, (VOCAB_SIZE, DIM))

    expected, ranks = sharded_parallel_embedding(weight, ids, world_size=args.world_size, dim=args.dim)
    direct = np.ascontiguousarray(weight[ids, : args.dim], dtype=np.uint16)
    bounded_weight = np.ascontiguousarray(weight[:, : args.dim], dtype=np.uint16)
    direct_matches = bool(np.array_equal(expected, direct))

    record: dict[str, Any] = {
        "schema": "ds41f.parallel-embedding-official-reference-fixture.v1",
        "classification": "official_reference_derived",
        "not_omlx_derived": True,
        "purpose": "bounded official ParallelEmbedding.forward sharded gather/all_reduce fixture from raw checkpoint BF16 bits",
        "checkpoint": str(ckpt),
        "official_reference": {
            "file": "inference/model.py",
            "file_sha256": MODEL_PY_SHA,
            "function_or_class": "ParallelEmbedding.forward",
            "source_lines": [168, 178],
            "source_sha256": FORWARD_SHA,
        },
        "source_tensor": {
            "name": "embed.weight",
            "shard": str(shard),
            "shape": [VOCAB_SIZE, DIM],
            "dtype": "BF16 raw uint16 bits",
            "bounded_dim": args.dim,
        },
        "operation_contract": {
            "world_size": args.world_size,
            "rank_count_modeled": args.world_size,
            "part_vocab_size": VOCAB_SIZE // args.world_size,
            "operation": "rank-local masked BF16 gather followed by all_reduce sum; exactly one rank contributes each token row",
            "output_shape": [int(ids.size), args.dim],
            "output_dtype": "BF16 raw uint16 bits",
            "expected_provider": "independent NumPy implementation of reviewed ParallelEmbedding.forward masking/all_reduce contract over official checkpoint bits",
        },
        "inputs": {
            "tokens": [int(x) for x in ids.tolist()],
            "tokens_sha256": digest(ids.astype(np.int64)),
        },
        "rank_records": ranks,
        "digests": {
            "source_bounded_weight_bf16_bits_sha256": digest(bounded_weight),
            "expected_output_bf16_bits_sha256": digest(expected),
            "direct_raw_gather_crosscheck_sha256": digest(direct),
        },
        "comparison": {
            "direct_raw_gather_crosscheck_matches": direct_matches,
            "bit_exact": direct_matches,
        },
        "non_claims": [
            "does not execute oMLX",
            "does not execute PyTorch distributed runtime",
            "does not validate tokenizer behavior",
            "does not validate HC repeat/pre-mask, attention, MoE, Engram, DSpark/MTP, logits, cache, or full prefill",
            "does not implement or validate native model math",
        ],
    }
    record["ok"] = bool(direct_matches)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
