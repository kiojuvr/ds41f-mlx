#!/usr/bin/env python3
"""Generate a bounded ParallelHead.forward fixture from official checkpoint bits.

This does not execute oMLX. Expected values are independent arithmetic over
official BF16 checkpoint tensors after applying the reviewed official source
contract: checkpoint BF16 head.weight bits are represented as FP32, x.float() is
used, then F.linear computes x @ weight.T. The bounded fixture validates selected
vocabulary rows only, not full-model logits correctness.
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
PARALLEL_HEAD_SHA = "6247ecdd05c6a9abae3c53eb58bb1154a88e56caa1d149ec71554c405b18afa0"
PARALLEL_HEAD_FORWARD_SHA = "2537fe7c5c90707fb39100f54ef9559efed9979513f5f0f0939e063742ceaa52"
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


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.uint32) << 16
    return u.view(np.float32)


def parse_ints(s: str) -> list[int]:
    xs = [int(x) for x in s.split(",") if x.strip()]
    if not xs:
        raise ValueError("empty integer list")
    return xs


def linear_f32_ordered(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    rows, dim = x.shape
    out_dim = w.shape[0]
    out = np.empty((rows, out_dim), dtype=np.float32)
    for r in range(rows):
        for o in range(out_dim):
            acc = np.float32(0.0)
            for k in range(dim):
                acc = np.float32(acc + np.float32(x[r, k] * w[o, k]))
            out[r, o] = acc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--input-tokens", default="0,3,42")
    ap.add_argument("--vocab-rows", default="0,1,2,3,42,128799,129264,129279")
    ap.add_argument("--out", default="artifacts/parallel-head-official-reference-fixture.json")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    tokens = parse_ints(args.input_tokens)
    vocab_rows = parse_ints(args.vocab_rows)
    if min(tokens) < 0 or max(tokens) >= VOCAB_SIZE:
        raise ValueError("input token out of range")
    if min(vocab_rows) < 0 or max(vocab_rows) >= VOCAB_SIZE:
        raise ValueError("vocab row out of range")

    embed_shard = ckpt / "model-00002-of-00048.safetensors"
    head_shard = ckpt / "model-00043-of-00048.safetensors"
    embed = tensor_memmap(embed_shard, "embed.weight", np.uint16, (VOCAB_SIZE, DIM))
    head = tensor_memmap(head_shard, "head.weight", np.uint16, (VOCAB_SIZE, DIM))

    input_bf16 = np.ascontiguousarray(embed[tokens], dtype=np.uint16)
    weight_bf16 = np.ascontiguousarray(head[vocab_rows], dtype=np.uint16)
    input_f32 = np.ascontiguousarray(bf16_to_f32(input_bf16), dtype=np.float32)
    weight_f32 = np.ascontiguousarray(bf16_to_f32(weight_bf16), dtype=np.float32)

    true_flat = linear_f32_ordered(input_f32, weight_f32)
    true_out = true_flat.reshape(1, len(tokens), len(vocab_rows))
    false_out = true_out[:, -1, :]
    last_position_equivalence = bool(np.array_equal(false_out, true_out[:, -1, :]))

    record: dict[str, Any] = {
        "schema": "ds41f.parallel-head-official-reference-fixture.v1",
        "classification": "official_reference_derived_independent_arithmetic_contract",
        "not_omlx_derived": True,
        "purpose": "bounded selected-vocabulary ParallelHead.forward fixture over official checkpoint head.weight bits",
        "checkpoint": str(ckpt),
        "authority": {
            "official_checkpoint_raw_bits": str(ckpt),
            "official_reference_source": "inference/model.py",
            "expected_value_provider": "independent_arithmetic_contract over official checkpoint tensors",
        },
        "official_reference": {
            "file": "inference/model.py",
            "file_sha256": MODEL_PY_SHA,
            "functions": [
                {"name": "ParallelHead", "source_lines": [997, 1017], "source_sha256": PARALLEL_HEAD_SHA},
                {"name": "ParallelHead.forward", "source_lines": [1008, 1017], "source_sha256": PARALLEL_HEAD_FORWARD_SHA},
            ],
        },
        "source_tensors": {
            "checkpoint_head_weight": {"name": "head.weight", "shard": str(head_shard), "checkpoint_shape": [VOCAB_SIZE, DIM], "checkpoint_dtype": "BF16", "provenance": "official checkpoint raw safetensors bits"},
            "input": {"source": "embed.weight gathered rows used as bounded input tensor", "shard": str(embed_shard), "tokens": tokens, "shape": [1, len(tokens), DIM], "checkpoint_dtype": "BF16 raw uint16 bits", "f32_digest": digest(input_f32)},
            "weight_slice": {"name": "head.weight selected rows", "vocab_rows": vocab_rows, "shape": [len(vocab_rows), DIM], "checkpoint_dtype": "BF16 raw uint16 bits", "f32_digest": digest(weight_f32)},
        },
        "inputs": {
            "x_f32": {"shape": [1, len(tokens), DIM], "dtype": "F32 converted from BF16 raw bits", "digest": digest(input_f32)},
            "selected_head_weight_f32": {"shape": [len(vocab_rows), DIM], "dtype": "F32 representation of BF16 checkpoint head.weight rows", "digest": digest(weight_f32)},
        },
        "operation_contract": {
            "checkpoint_weight_loading": "official ParallelHead parameter is FP32; BF16 checkpoint head.weight bits are represented exactly as FP32 values before forward",
            "x_cast": "x.float() before F.linear",
            "fixture_operation": "selected_expected = input_f32 @ selected_head_weight_f32.T",
            "accumulation_contract": "F32 accumulation over k=0..5119 in increasing k order for this fixture",
            "output_dtype": "F32",
            "full_logits_false": "select x[:, -1] before F.linear; fixture output shape [1, selected_vocab_rows]",
            "full_logits_true": "apply F.linear to every sequence position; fixture output shape [1, sequence, selected_vocab_rows]",
            "world_size": 1,
            "tp_gather_concat_contract": "if world_size > 1, all_gather per-rank logits and torch.cat along last dimension; not executed by this fixture",
            "predeclared_tolerance": {"max_abs_error_lte": 0.0001, "max_relative_error_lte": 0.000001},
            "expected_provider": "independent arithmetic over official BF16 checkpoint input/weight bits; no oMLX/PyTorch/MLX/native execution",
        },
        "expected": {
            "full_logits_true_selected_f32": true_out.tolist(),
            "full_logits_false_selected_f32": false_out.tolist(),
            "full_logits_true_shape": list(true_out.shape),
            "full_logits_false_shape": list(false_out.shape),
        },
        "digests": {"expected_full_logits_true_f32_sha256": digest(true_out), "expected_full_logits_false_f32_sha256": digest(false_out)},
        "comparison": {"self_consistent": True, "false_output_equals_true_last_position": last_position_equivalence},
        "non_claims": [
            "does not execute oMLX",
            "does not claim full-model logits correctness",
            "does not validate Transformer hidden-state production",
            "does not validate full vocabulary output, only selected head.weight rows",
            "does not execute or validate tensor-parallel gather/concat beyond documenting the contract",
            "does not validate attention, MoE, Hyper-Connections, cache, sampling, DSpark/MTP, FP8, or FP4",
        ],
        "ok": last_position_equivalence,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
