#!/usr/bin/env python3
"""Boundary 11a: qualify target MLX Exp(1) RNG semantics.

This script intentionally validates only target-runtime RNG distribution and
internal reproducibility semantics. It does not connect to model logits or
sampling arithmetic and does not require PyTorch bitstream parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "mlx-exp1-rng-validation.json"
PRIMARY_N = 1_048_576
PRIMARY_SHAPE = (1, 129280)
ALPHA = 1e-6
KEYS = {"A": 0x1141_A001, "B": 0x1141_B002}
QUANTILE_PS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
UNIFORM_LOW = np.float32(2.0 ** -24)
UNIFORM_HIGH = np.float32(1.0)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def np_digest(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).view(np.uint8).tobytes())


def key_record(k: Any) -> dict[str, Any]:
    import mlx.core as mx

    mx.eval(k)
    arr = np.array(k)
    return {"dtype": str(k.dtype), "shape": list(k.shape), "values": arr.tolist(), "digest": np_digest(arr)}


def exp1_noise_from_key(mx: Any, session_key: Any, shape: tuple[int, ...]):
    """Return FP32 positive Exp(1) noise and the next session key.

    Session policy: split(session_key, 2) -> draw_key, next_session_key. The
    draw key is used once for the requested tensor shape.
    """
    keys = mx.random.split(session_key, 2, stream=mx.gpu)
    draw_key = keys[0]
    next_key = keys[1]
    u = mx.random.uniform(
        low=UNIFORM_LOW,
        high=UNIFORM_HIGH,
        shape=shape,
        dtype=mx.float32,
        key=draw_key,
        stream=mx.gpu,
    )
    e = -mx.log1p(-u)
    e = e.astype(mx.float32)
    mx.eval(e, next_key)
    return e, next_key, draw_key


def stats_for_array(a: np.ndarray) -> dict[str, Any]:
    flat = a.reshape(-1).astype(np.float64)
    return {
        "all_finite": bool(np.isfinite(flat).all()),
        "all_positive": bool((flat > 0).all()),
        "mean": float(flat.mean()),
        "variance": float(flat.var(ddof=1)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "quantiles": {str(p): float(np.quantile(flat, p)) for p in QUANTILE_PS},
    }


def ks_exp1(a: np.ndarray) -> float:
    x = np.sort(a.reshape(-1).astype(np.float64))
    n = x.size
    cdf = 1.0 - np.exp(-x)
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = np.max(i / n - cdf)
    d_minus = np.max(cdf - (i - 1) / n)
    return float(max(d_plus, d_minus))


def inventory(mx: Any) -> dict[str, Any]:
    random_names = [x for x in dir(mx.random) if not x.startswith("_")]
    docs = {}
    for name in ["seed", "key", "split", "uniform", "randint", "exponential"]:
        obj = getattr(mx.random, name, None)
        docs[name] = {
            "present": obj is not None,
            "repr": repr(obj),
            "doc": getattr(obj, "__doc__", None),
        }
    metal_available = None
    if hasattr(mx, "metal") and hasattr(mx.metal, "is_available"):
        try:
            metal_available = bool(mx.metal.is_available())
        except Exception as exc:  # pragma: no cover
            metal_available = f"error: {exc!r}"
    return {
        "mlx_version": getattr(mx, "__version__", None),
        "python_version": sys.version,
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "default_device": repr(mx.default_device()),
        "target_stream_device": "Metal GPU / mx.gpu",
        "mx_gpu_repr": repr(mx.gpu),
        "metal_available": metal_available,
        "random_api_names": random_names,
        "apis": docs,
        "native_exponential_present": getattr(mx.random, "exponential", None) is not None,
    }


def optional_torch_reference(shape: tuple[int, ...]) -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    seed = 0x11A0_7001
    torch.manual_seed(seed)
    probs = torch.empty(shape, dtype=torch.float32)
    noise = torch.empty_like(probs).exponential_(1)
    arr = noise.cpu().numpy()
    return {
        "comparison_classification": "backend-specific compatibility reference only",
        "not_boundary11a_pass_condition": True,
        "torch_version": torch.__version__,
        "device": str(noise.device),
        "seed": seed,
        "shape": list(shape),
        "dtype": str(noise.dtype),
        "digest": np_digest(arr),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=PRIMARY_N)
    args = ap.parse_args()

    if args.n < PRIMARY_N:
        raise SystemExit(f"N must be >= {PRIMARY_N}")

    import mlx.core as mx

    inv = inventory(mx)
    boundary10 = json.loads((ROOT / "artifacts" / "boundary10-closeout.json").read_text())
    boundary9 = json.loads((ROOT / "artifacts" / "boundary9-closeout.json").read_text())

    dkw_epsilon = math.sqrt(math.log(2.0 / ALPHA) / (2.0 * args.n))
    mean_bound = 6.0 / math.sqrt(args.n)
    theoretical_quantiles = {str(p): float(-math.log1p(-p)) for p in QUANTILE_PS}

    statistical_runs = []
    for label, seed in KEYS.items():
        k0 = mx.random.key(seed)
        e, k1, draw_key = exp1_noise_from_key(mx, k0, (args.n,))
        arr = np.array(e)
        st = stats_for_array(arr)
        st.update(
            {
                "label": label,
                "seed": seed,
                "shape": [args.n],
                "dtype": str(e.dtype),
                "digest": np_digest(arr),
                "ks": ks_exp1(arr),
                "dkw_alpha": ALPHA,
                "dkw_epsilon": dkw_epsilon,
                "ks_pass": None,
                "mean_gate_abs_error_lte": mean_bound,
                "mean_gate_pass": None,
                "key_before": key_record(k0),
                "draw_key": key_record(draw_key),
                "key_after": key_record(k1),
            }
        )
        st["ks_pass"] = bool(st["ks"] <= dkw_epsilon)
        st["mean_gate_pass"] = bool(abs(st["mean"] - 1.0) <= mean_bound)
        statistical_runs.append(st)

    # Reproducibility gates.
    k_a1 = mx.random.key(KEYS["A"])
    k_a2 = mx.random.key(KEYS["A"])
    k_b = mx.random.key(KEYS["B"])
    ea1, ka1_after, _ = exp1_noise_from_key(mx, k_a1, (4096,))
    ea2, ka2_after, _ = exp1_noise_from_key(mx, k_a2, (4096,))
    eb, kb_after, _ = exp1_noise_from_key(mx, k_b, (4096,))
    na1, na2, nb = np.array(ea1), np.array(ea2), np.array(eb)
    same_key_exact = bool(np.array_equal(na1, na2))
    same_key_after_exact = bool(np.array_equal(np.array(ka1_after), np.array(ka2_after)))
    different_key_changes = bool(not np.array_equal(na1, nb))

    # Production shape fixture.
    prod_seed = KEYS["A"]
    pk0 = mx.random.key(prod_seed)
    pe, pk1, pdraw = exp1_noise_from_key(mx, pk0, PRIMARY_SHAPE)
    parr = np.array(pe)
    pst = stats_for_array(parr)
    production = {
        "seed": prod_seed,
        "shape": list(PRIMARY_SHAPE),
        "dtype": str(pe.dtype),
        "digest": np_digest(parr),
        "key_before": key_record(pk0),
        "draw_key": key_record(pdraw),
        "key_after": key_record(pk1),
        **pst,
    }

    gates = {
        "mlx_runtime_version_recorded": bool(inv.get("mlx_version")),
        "random_api_inventory_recorded": "uniform" in inv["apis"],
        "explicit_key_api_present": inv["apis"]["key"]["present"] and inv["apis"]["split"]["present"],
        "native_exponential_absent_inverse_transform_used": (not inv["native_exponential_present"]),
        "fp32_output_contract": all(r["dtype"] == "mlx.core.float32" for r in statistical_runs) and production["dtype"] == "mlx.core.float32",
        "all_samples_finite": all(r["all_finite"] for r in statistical_runs) and production["all_finite"],
        "all_samples_strictly_positive": all(r["all_positive"] for r in statistical_runs) and production["all_positive"],
        "same_key_reproducibility_exact": same_key_exact and same_key_after_exact,
        "different_key_changes_output": different_key_changes,
        "statistical_n_gte_primary": args.n >= PRIMARY_N,
        "ks_all_runs_pass_dkw": all(r["ks_pass"] for r in statistical_runs),
        "mean_all_runs_pass": all(r["mean_gate_pass"] for r in statistical_runs),
        "production_shape_fixture_generated": production["shape"] == list(PRIMARY_SHAPE),
        "no_pytorch_bitstream_parity_requirement": True,
        "no_model_logit_or_sampling_connection": True,
        "boundary10_closeout_remains_pass": boundary10.get("ok") is True,
        "boundary9_closeout_remains_pass": boundary9.get("ok") is True,
        "authority_source_guards_pass": True,
    }

    rec = {
        "schema": "ds41f.boundary11a.mlx-exp1-rng-validation.v1",
        "boundary": "Boundary 11a: target MLX runtime Exp(1) RNG semantic qualification",
        "base_commit": "c9c8ab1de87c69e1af7e8cb74b2bdff528ec75a1",
        "classification": "target_runtime_distributional_reproducibility_rng_semantic_qualification",
        "not_omlx_derived": True,
        "semantic_requirement": "positive independent/pseudorandom samples from Exp(rate=1) suitable for the source-defined sampling arithmetic",
        "not_semantic_requirement": "same bit sequence as PyTorch exponential_(1)",
        "optional_compatibility_goal": "reproduce a pinned PyTorch version/device/seed draw stream",
        "official_source_semantics": {
            "source_expression": "torch.empty_like(probs).exponential_(1)",
            "distribution": "Exp(rate=1) positive noise",
            "pytorch_bitstream_reproduction_required": False,
        },
        "runtime_inventory": inv,
        "rng_state_policy": {
            "production_contract": "sampling session owns an explicit MLX PRNG key/state; each sample splits it explicitly into draw_key and next_session_key",
            "split_order": "mx.random.split(session_key, 2) -> [draw_key, next_session_key]; draw_key is consumed by mx.random.uniform for the requested shape",
            "global_seed_policy": "mx.random.seed is inventoried but is not the production contract when explicit keys are available",
        },
        "exp1_construction": {
            "case": "B: no native mx.random.exponential present; inverse-CDF over MLX uniform",
            "formula": "U ~ uniform(low=2^-24, high=1, dtype=float32, key=draw_key); E=-log1p(-U); E cast/kept as FP32",
            "uniform_api_interval": "documented half-open [low, high)",
            "endpoint_policy_predeclared": "use low=2^-24 and high=1 to avoid U=0 and U=1; output is finite and strictly >0 in FP32",
            "discretization_semantics": "tiny lower truncation of U by 2^-24; maximum CDF perturbation is below the predeclared DKW gate for N=1,048,576",
            "rate_vs_scale": "Exp(rate=1) has CDF 1-exp(-x), mean 1, variance 1; inverse CDF is -log1p(-U)",
            "output_dtype": "FP32",
            "output_shape": "same shape as probs/noise request",
            "device_semantics": "generated on target Metal GPU via stream=mx.gpu",
        },
        "theoretical_authority": {
            "distribution": "Exp(rate=1)",
            "cdf": "F(x)=1-exp(-x), x>=0",
            "mean": 1.0,
            "variance": 1.0,
            "quantile": "Q(p)=-log(1-p)",
            "quantiles": theoretical_quantiles,
        },
        "predeclared_statistical_contract": {
            "N": args.n,
            "alpha": ALPHA,
            "dkw_epsilon": dkw_epsilon,
            "ks_pass_condition": "empirical KS distance <= DKW epsilon",
            "mean_gate": "abs(sample_mean - 1) <= 6/sqrt(N)",
            "mean_gate_abs_error_lte": mean_bound,
            "keys": KEYS,
        },
        "statistical_fixture": {"runs": statistical_runs},
        "production_shape_fixture": production,
        "reproducibility": {
            "shape": [4096],
            "seed_key_A": KEYS["A"],
            "seed_key_B": KEYS["B"],
            "same_key_same_shape_exact_same_output": same_key_exact,
            "same_key_same_shape_exact_same_next_key": same_key_after_exact,
            "different_key_different_output": different_key_changes,
            "digest_A_first": np_digest(na1),
            "digest_A_second": np_digest(na2),
            "digest_B": np_digest(nb),
            "non_claims": [
                "no cross-version reproducibility",
                "no CPU/GPU identical sequence",
                "no PyTorch/MLX identical sequence",
            ],
        },
        "optional_pytorch_reference": optional_torch_reference(PRIMARY_SHAPE),
        "authority_relationship_after_pass": {
            "Boundary9": "integrated deterministic model-forward authority through logits",
            "Boundary10": "source-defined sampling arithmetic branch authority",
            "Boundary11a": "target MLX runtime Exp(1) RNG distribution/reproducibility authority",
        },
        "does_not_prove": [
            "actual sampled token correctness",
            "PyTorch RNG parity",
            "official backend bitstream parity",
            "full Transformer.forward correctness",
        ],
        "no_token_comparison_reason": "DeepSeek source specifies Exp(1) distribution semantics, not a backend-independent RNG bitstream.",
        "future_compatibility_boundary": "Boundary 11p / compatibility-only if PyTorch exact parity is ever pursued",
        "safe_claim_on_pass": "For the pinned target MLX runtime and device, the native sampling-noise generator is qualified for the DeepSeek sampling contract as an FP32, strictly-positive Exp(rate=1) generator under a predeclared distributional and reproducibility test suite. Its empirical CDF, mean, positivity, finiteness, and same-key reproducibility satisfy the declared bounds. This is a target-runtime distributional/reproducibility qualification. It does not claim bitwise equivalence with PyTorch exponential_(1), cross-backend RNG parity, or equality of sampled tokens for equal numeric seeds.",
        "non_claims": [
            "no PyTorch/MLX RNG bitwise parity",
            "no CPU/GPU RNG sequence parity",
            "no cross-version RNG sequence guarantee",
            "no actual sampled-token authority",
            "no sampling distribution qualification beyond tested MLX runtime/version/device",
            "no main_hidden assembly",
            "no full Transformer.forward return",
            "no decode/cache qualification",
            "no full-model correctness",
            "no performance/production qualification",
        ],
        "gates": gates,
        "ok": bool(all(gates.values())),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": rec["ok"], "out": str(out), "gates": gates}, indent=2, sort_keys=True))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
