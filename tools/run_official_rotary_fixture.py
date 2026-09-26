#!/usr/bin/env python3
"""Generate a bounded official-reference-derived rotary fixture.

This script does not import or execute torch, MLX, oMLX, or native code.  It
implements the reviewed arithmetic contract from inference/model.py for
precompute_freqs_cis() and apply_rotary_emb() over a tiny deterministic input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
MODEL_PY_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
PRECOMPUTE_SHA256 = "19cf7246ca3e6c479158e68d4ab7825d078aa788d3ce494b0c90649bb40117fc"
APPLY_SHA256 = "1c47da553d29f41c798a2eda82b72476473913d1201e32283de0e4977aa30b1a"


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32_hex(x: float) -> str:
    return struct.pack("<f", f32(x)).hex()


def sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def precompute_freqs_cis(dim: int, seqlen: int, original_seq_len: int, base: float, factor: float, beta_fast: int, beta_slow: int) -> list[list[tuple[float, float]]]:
    freqs = [f32(1.0 / f32(base ** f32(i / dim))) for i in range(0, dim, 2)]
    if original_seq_len > 0:
        def corrected_dim(rotations: int) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        out = []
        for j, freq in enumerate(freqs):
            ramp = min(max(f32(f32(j - low) / f32(max(high - low, 1e-3))), 0.0), 1.0)
            smooth = f32(1.0 - ramp)
            out.append(f32(f32(freq / factor) * f32(1.0 - smooth) + f32(freq * smooth)))
        freqs = out

    rows: list[list[tuple[float, float]]] = []
    for pos in range(seqlen):
        row = []
        for freq in freqs:
            angle = f32(pos * freq)
            row.append((f32(math.cos(angle)), f32(math.sin(angle))))
        rows.append(row)
    return rows


def apply_rotary_emb(x: list[list[list[list[float]]]], freqs_cis: list[list[tuple[float, float]]], inverse: bool = False) -> list[list[list[list[float]]]]:
    out: list[list[list[list[float]]]] = []
    for batch in x:
        batch_out = []
        for pos, heads in enumerate(batch):
            pos_out = []
            for values in heads:
                vals_out = []
                for pair_idx in range(0, len(values), 2):
                    xr = f32(values[pair_idx])
                    xi = f32(values[pair_idx + 1])
                    cr, ci = freqs_cis[pos][pair_idx // 2]
                    if inverse:
                        ci = f32(-ci)
                    vals_out.append(f32(f32(xr * cr) - f32(xi * ci)))
                    vals_out.append(f32(f32(xr * ci) + f32(xi * cr)))
                pos_out.append(vals_out)
            batch_out.append(pos_out)
        out.append(batch_out)
    return out


def tensor_f32_hex(t: Any) -> Any:
    if isinstance(t, list):
        return [tensor_f32_hex(x) for x in t]
    return f32_hex(float(t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out", default="artifacts/rotary-official-reference-fixture.json")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=4)
    ap.add_argument("--original-seq-len", type=int, default=0)
    ap.add_argument("--base", type=float, default=10000.0)
    ap.add_argument("--factor", type=float, default=40.0)
    ap.add_argument("--beta-fast", type=int, default=32)
    ap.add_argument("--beta-slow", type=int, default=1)
    args = ap.parse_args()

    params = {
        "dim": args.dim,
        "seqlen": args.seqlen,
        "original_seq_len": args.original_seq_len,
        "base": args.base,
        "factor": args.factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
    }
    if params["dim"] <= 0 or params["dim"] % 2:
        raise ValueError("--dim must be a positive even integer")
    if params["seqlen"] <= 0:
        raise ValueError("--seqlen must be positive")
    x = [[[[f32((i + 1) / 16.0) for i in range(params["dim"])] for _h in range(1)] for _s in range(params["seqlen"])] for _b in range(1)]
    # Make each sequence row distinct while preserving a compact fixture.
    for s in range(params["seqlen"]):
        for i in range(params["dim"]):
            x[0][s][0][i] = f32(x[0][s][0][i] + s / 32.0)

    freqs = precompute_freqs_cis(**params)
    y = apply_rotary_emb(x, freqs, inverse=False)
    y_inv = apply_rotary_emb(y, freqs, inverse=True)

    expected = {
        "freqs_cis_complex64_hex": [[[f32_hex(r), f32_hex(im)] for r, im in row] for row in freqs],
        "input_f32_hex": tensor_f32_hex(x),
        "rotated_f32_hex": tensor_f32_hex(y),
        "inverse_after_forward_f32_hex": tensor_f32_hex(y_inv),
    }

    record = {
        "schema": "ds41f.rotary-official-reference-fixture.v1",
        "purpose": "bounded fixture for reviewed precompute_freqs_cis/apply_rotary_emb arithmetic; no torch/MLX/oMLX/native execution",
        "classification": "official_reference_derived_independent_arithmetic_contract",
        "not_omlx_derived": True,
        "checkpoint": args.checkpoint,
        "official_reference": {
            "file": "inference/model.py",
            "file_sha256": MODEL_PY_SHA256,
            "targets": [
                {"name": "precompute_freqs_cis", "source_lines": [369, 389], "source_sha256": PRECOMPUTE_SHA256},
                {"name": "apply_rotary_emb", "source_lines": [392, 406], "source_sha256": APPLY_SHA256},
            ],
        },
        "operation_contract": {
            "precompute": "freqs=1/(base**(arange(0,dim,2,float32)/dim)); optional YaRN branch omitted by original_seq_len=0; freqs=outer(arange(seqlen),freqs); polar(ones_like(freqs),freqs) as complex64-compatible f32 real/imag bits",
            "apply": "view adjacent f32 element pairs as complex, multiply by per-position freqs_cis, optionally conjugating for inverse, view_as_real and flatten back to f32",
            "fixture_scope": f"shape [batch=1,seqlen={params['seqlen']},heads=1,dim={params['dim']}], original_seq_len={params['original_seq_len']}",
            "expected_provider": "pure-Python independent arithmetic implementation with explicit f32 rounding at stored fixture boundaries; no torch/MLX/oMLX/native execution",
        },
        "inputs": {"params": params, "x_shape": [1, 4, 1, 8], "x_dtype": "float32"},
        "expected": expected,
        "digests": {k + "_sha256": sha256_json(v) for k, v in expected.items()},
        "semantic_status": {
            "official_reference_fixture_used": True,
            "validated_stage": "rotary frequency and application helpers only",
            "native_rotary_validated_against_contract": False,
            "model_semantics_validated": False,
        },
        "non_claims": [
            "does not validate attention, cache publication, logits, layers, or full-model correctness",
            "covers YaRN only when original_seq_len > 0 in fixture params",
            "does not validate native rotary code",
        ],
        "ok": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
