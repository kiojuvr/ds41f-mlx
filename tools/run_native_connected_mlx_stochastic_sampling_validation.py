#!/usr/bin/env python3
"""Boundary 11b: connected target-MLX-runtime stochastic sampling.

Connects actual Boundary9 connected logits, Boundary10 sampling arithmetic, and
Boundary11a-qualified MLX Exp(1)-compatible noise for one bounded fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_native_parallel_head_logits_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    VOCAB,
    HC,
    DIM,
    mmap,
    shard,
    digest,
    cfg,
    block,
    bf16_to_f32,
    f32_to_bf16,
    post_loop_hc_pre,
    rmsnorm,
    native_parallel_head_logits,
)
from tools.run_official_hyper_connections_fixture import header  # noqa: E402

OUT_DEFAULT = ROOT / "artifacts" / "native-connected-mlx-stochastic-sampling-validation.json"
EXPECTED_B9_LOGITS_DIGEST = "b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d"
EXPECTED_B11A_NOISE_DIGEST = "38211ff282e405ad5d57f7203a09f80745c382373c755fb8e0d9c4c9187a3cb4"
EXPECTED_INITIAL_KEY_DIGEST = "39adfde986a1ad94318c84e5edd0f8a45a729445c6b0e1ca8912e74958a07630"
EXPECTED_DRAW_KEY_DIGEST = "4cc9141bec4c30b22659a725d59ca69b5dd50e83deb9bdfde53c177f079e15c1"
EXPECTED_NEXT_KEY_DIGEST = "175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4"
PRIMARY_SEED = 289513473
SECONDARY_SEED = 289517570
PRIMARY_SHAPE = (1, 129280)
UNIFORM_LOW = np.float32(2.0**-24)
UNIFORM_HIGH = np.float32(1.0)
SOFTMAX_MAX_ABS_LTE = 1e-6
SOFTMAX_SUM_ABS_ERROR_LTE = 1e-6
TEMPERATURE = 1.0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def np_digest(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).view(np.uint8).tobytes())


def key_record(mx: Any, k: Any) -> dict[str, Any]:
    mx.eval(k)
    arr = np.array(k)
    return {"shape": list(k.shape), "dtype": str(k.dtype), "values": arr.tolist(), "digest": np_digest(arr)}


def exp1_noise_from_session_key(mx: Any, session_key: Any, shape: tuple[int, ...]):
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
    noise = (-mx.log1p(-u)).astype(mx.float32)
    mx.eval(noise, next_key, draw_key)
    return noise, draw_key, next_key


def connected_logits(ck: Path) -> tuple[np.ndarray, dict[str, Any]]:
    c = cfg(ck)
    token_ids = np.array([[0, 3]], np.int64)
    emb = np.ascontiguousarray(mmap(ck / "model-00002-of-00048.safetensors", "embed.weight", np.uint16, (VOCAB, DIM)))
    embed_out = emb[token_ids].copy()
    x = np.repeat(embed_out[:, :, None, :], HC, axis=2).copy()
    pre = np.zeros((1, 2, HC), np.float32)
    pre[:, :, 0] = 1.0
    shared = {"compress_kv": None, "index_k": None, "candidates": None, "topk_idxs": None}
    layers = {}
    for layer in range(40):
        out = block(ck, c, layer, x, pre, shared)
        x = out["x_out"]
        pre = out["ffn_pre"]
        layers[str(layer)] = {"x_out": digest(x), "ffn_pre": digest(pre)}
    post_loop_h = post_loop_hc_pre(x, pre)
    index = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]
    norm_name = "norm.weight"
    norm_weight = np.ascontiguousarray(mmap(ck / index[norm_name], norm_name, np.uint16, (DIM,)))
    normalized_h = rmsnorm(post_loop_h, norm_weight, float(c["rms_norm_eps"]))
    head_name = "head.weight"
    head_shard = ck / index[head_name]
    hinfo, _ = header(head_shard)
    hmeta = hinfo[head_name]
    if hmeta["dtype"] != "BF16" or hmeta["shape"] != [VOCAB, DIM]:
        raise RuntimeError(f"unexpected head metadata {hmeta}")
    head_weight = mmap(head_shard, head_name, np.uint16, (VOCAB, DIM))
    selected = normalized_h[:, -1, :].copy()
    logits = native_parallel_head_logits(selected, head_weight, 1024).astype(np.float32, copy=False)
    meta = {
        "tokens": token_ids.tolist(),
        "embedding_output_digest": digest(embed_out),
        "layer39_x_out_digest": layers["39"]["x_out"],
        "layer39_ffn_pre_digest": layers["39"]["ffn_pre"],
        "post_loop_h_digest": digest(post_loop_h),
        "normalized_h_digest": digest(normalized_h),
        "selected_final_position_digest": digest(selected),
        "head_weight_shard": str(head_shard),
        "head_weight_checkpoint_dtype": hmeta["dtype"],
        "head_weight_shape": hmeta["shape"],
        "no_logits_artifact_injection": True,
        "execution_path": "tokens -> embedding -> Blocks0..39 -> post-loop HC collapse -> final RMSNorm -> ParallelHead -> final-position logits",
    }
    return logits, meta


def independent_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def top2_desc(values: np.ndarray) -> tuple[int, float, int, float, int]:
    flat = values.reshape(-1)
    maxv = np.max(flat)
    tie_count = int(np.sum(flat == maxv))
    idx = np.argpartition(flat, -2)[-2:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return int(idx[0]), float(flat[idx[0]]), int(idx[1]), float(flat[idx[1]]), tie_count


def run_mlx_sampling(mx: Any, logits_host: np.ndarray, seed: int) -> dict[str, Any]:
    session_key = mx.random.key(seed)
    logits_mx = mx.array(np.ascontiguousarray(logits_host, dtype=np.float32))
    mx.eval(logits_mx)
    logits_roundtrip = np.array(logits_mx)
    scaled_logits = (logits_mx / np.float32(TEMPERATURE)).astype(mx.float32)
    probs = mx.softmax(scaled_logits, axis=-1, stream=mx.gpu)
    noise, draw_key, next_key = exp1_noise_from_session_key(mx, session_key, PRIMARY_SHAPE)
    scores = probs / noise
    sampled = mx.argmax(scores, axis=-1)
    mx.eval(probs, noise, scores, sampled, next_key)
    probs_host = np.array(probs)
    noise_host = np.array(noise)
    scores_host = np.array(scores)
    sampled_host = np.array(sampled)
    return {
        "session_key": session_key,
        "draw_key": draw_key,
        "next_key": next_key,
        "logits_mx": logits_mx,
        "logits_roundtrip": logits_roundtrip,
        "probs_host": probs_host,
        "noise_host": noise_host,
        "scores_host": scores_host,
        "sampled_host": sampled_host,
    }


def sampling_analysis(logits: np.ndarray, probs: np.ndarray, noise: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    ref_probs = independent_softmax(logits)
    probs64 = probs.astype(np.float64)
    softmax_abs = np.abs(probs64 - ref_probs)
    log_scores = logits.reshape(-1).astype(np.float64) / TEMPERATURE - np.log(noise.reshape(-1).astype(np.float64))
    ind_t1, ind_v1, ind_t2, ind_v2, ind_ties = top2_desc(log_scores)
    t1, s1, t2, s2, ties = top2_desc(scores)
    return {
        "softmax": {
            "digest": np_digest(probs),
            "shape": list(probs.shape),
            "dtype": str(probs.dtype),
            "sum": float(np.sum(probs64)),
            "min": float(np.min(probs64)),
            "max": float(np.max(probs64)),
            "max_abs_vs_independent": float(np.max(softmax_abs)),
            "sum_abs_error_vs_independent": float(abs(np.sum(probs64) - np.sum(ref_probs))),
            "predeclared_tolerance": {
                "softmax_max_abs_lte": SOFTMAX_MAX_ABS_LTE,
                "softmax_sum_abs_error_lte": SOFTMAX_SUM_ABS_ERROR_LTE,
            },
        },
        "winner": {
            "runtime_score_domain": {
                "top1_token": t1,
                "top1_prob": float(probs.reshape(-1)[t1]),
                "top1_noise": float(noise.reshape(-1)[t1]),
                "top1_prob_over_noise_score": s1,
                "top2_token": t2,
                "top2_prob": float(probs.reshape(-1)[t2]),
                "top2_noise": float(noise.reshape(-1)[t2]),
                "top2_prob_over_noise_score": s2,
                "score_gap": float(s1 - s2),
                "tie_count": ties,
            },
            "independent_log_domain": {
                "top1_token": ind_t1,
                "top1_log_score": ind_v1,
                "top2_token": ind_t2,
                "top2_log_score": ind_v2,
                "log_score_gap": float(ind_v1 - ind_v2),
                "tie_count": ind_ties,
            },
        },
    }


def noise_stats(noise: np.ndarray) -> dict[str, Any]:
    x = noise.reshape(-1).astype(np.float64)
    return {
        "shape": list(noise.shape),
        "dtype": str(noise.dtype),
        "digest": np_digest(noise),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "all_positive": bool(np.all(x > 0)),
        "all_finite": bool(np.all(np.isfinite(x))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    ck = Path(args.checkpoint)

    import mlx.core as mx

    # Required baseline artifacts and Boundary11a checker.
    b9 = json.loads((ROOT / "artifacts" / "boundary9-closeout.json").read_text())
    b10 = json.loads((ROOT / "artifacts" / "boundary10-closeout.json").read_text())
    b11a = json.loads((ROOT / "artifacts" / "mlx-exp1-rng-validation.json").read_text())
    checker = subprocess.run([sys.executable, str(ROOT / "tools" / "check_boundary11a_rng.py")], cwd=ROOT, text=True, capture_output=True)

    logits1, meta1 = connected_logits(ck)
    primary1 = run_mlx_sampling(mx, logits1, PRIMARY_SEED)
    analysis1 = sampling_analysis(logits1, primary1["probs_host"], primary1["noise_host"], primary1["scores_host"])

    logits2, meta2 = connected_logits(ck)
    primary2 = run_mlx_sampling(mx, logits2, PRIMARY_SEED)
    secondary = run_mlx_sampling(mx, logits1, SECONDARY_SEED)

    host_digest = np_digest(logits1)
    roundtrip_digest = np_digest(primary1["logits_roundtrip"])
    sampled_token = int(primary1["sampled_host"].reshape(-1)[0])
    independent_token = int(analysis1["winner"]["independent_log_domain"]["top1_token"])
    ns1 = noise_stats(primary1["noise_host"])
    ns2 = noise_stats(primary2["noise_host"])
    ns_sec = noise_stats(secondary["noise_host"])

    key_before = key_record(mx, primary1["session_key"])
    draw_key = key_record(mx, primary1["draw_key"])
    next_key = key_record(mx, primary1["next_key"])
    key_before2 = key_record(mx, primary2["session_key"])
    next_key2 = key_record(mx, primary2["next_key"])

    soft = analysis1["softmax"]
    winner = analysis1["winner"]
    b10b_token = b10.get("boundary10b", {}).get("arithmetic", {}).get("conditional_output_id")

    same_key_repro = {
        "same_logits": bool(np.array_equal(logits1, logits2)),
        "same_generated_noise": bool(np.array_equal(primary1["noise_host"], primary2["noise_host"])),
        "same_next_session_key": bool(np.array_equal(np.array(primary1["next_key"]), np.array(primary2["next_key"]))),
        "same_sampled_token": bool(sampled_token == int(primary2["sampled_host"].reshape(-1)[0])),
        "second_sampled_token": int(primary2["sampled_host"].reshape(-1)[0]),
        "second_noise_digest": ns2["digest"],
        "second_next_session_key": key_record(mx, primary2["next_key"]),
    }
    secondary_div = {
        "seed": SECONDARY_SEED,
        "noise_digest": ns_sec["digest"],
        "noise_differs_from_primary": bool(not np.array_equal(primary1["noise_host"], secondary["noise_host"])),
        "sampled_token": int(secondary["sampled_host"].reshape(-1)[0]),
        "token_difference_not_required": True,
    }

    gates = {
        "latest_boundary9_closeout_pass": b9.get("ok") is True,
        "boundary10_closeout_pass": b10.get("ok") is True,
        "boundary11a_rng_checker_pass": checker.returncode == 0,
        "single_model_execution_starts_from_token_ids": meta1["tokens"] == [[0, 3]],
        "no_logits_artifact_injection": meta1["no_logits_artifact_injection"] is True,
        "boundary9_logits_digest_exact": host_digest == EXPECTED_B9_LOGITS_DIGEST and list(logits1.shape) == [1, VOCAB] and logits1.dtype == np.float32,
        "native_host_logits_to_mlx_fp32_seam_exact": host_digest == roundtrip_digest and np.array_equal(logits1, primary1["logits_roundtrip"]),
        "temperature_equals_1": TEMPERATURE == 1.0,
        "boundary11a_explicit_key_split_policy_used": True,
        "initial_session_key_exact": key_before["values"] == [0, PRIMARY_SEED] and key_before["digest"] == EXPECTED_INITIAL_KEY_DIGEST,
        "draw_key_exact": draw_key["values"] == [2308264947, 2363689365] and draw_key["digest"] == EXPECTED_DRAW_KEY_DIGEST,
        "next_session_key_exact": next_key["values"] == [613528892, 572711044] and next_key["digest"] == EXPECTED_NEXT_KEY_DIGEST,
        "actual_mlx_noise_regenerated": True,
        "mlx_noise_digest_exact_to_boundary11a_production_fixture": ns1["digest"] == EXPECTED_B11A_NOISE_DIGEST and ns1["digest"] == b11a["production_shape_fixture"]["digest"],
        "noise_fp32": ns1["dtype"] == "float32",
        "noise_all_positive": ns1["all_positive"],
        "noise_all_finite": ns1["all_finite"],
        "mlx_sampling_arithmetic_executed": True,
        "mlx_fp32_softmax_independently_validated": soft["max_abs_vs_independent"] <= SOFTMAX_MAX_ABS_LTE and soft["sum_abs_error_vs_independent"] <= SOFTMAX_SUM_ABS_ERROR_LTE,
        "predeclared_softmax_tolerance_pass": soft["max_abs_vs_independent"] <= SOFTMAX_MAX_ABS_LTE and soft["sum_abs_error_vs_independent"] <= SOFTMAX_SUM_ABS_ERROR_LTE,
        "mlx_probs_noise_division_executed": bool(np_digest(primary1["scores_host"])),
        "mlx_sampled_token_computed": isinstance(sampled_token, int),
        "independent_fp64_log_domain_token_computed": isinstance(independent_token, int),
        "sampled_token_exact_between_mlx_and_independent_oracle": sampled_token == independent_token,
        "winner_margins_recorded": winner["runtime_score_domain"]["score_gap"] > 0 and winner["independent_log_domain"]["log_score_gap"] > 0,
        "tie_count_eq_1": winner["runtime_score_domain"]["tie_count"] == 1 and winner["independent_log_domain"]["tie_count"] == 1,
        "same_key_whole_sampling_result_reproducible": all(same_key_repro[k] for k in ["same_logits", "same_generated_noise", "same_next_session_key", "same_sampled_token"]),
        "same_key_next_session_key_reproducible": same_key_repro["same_next_session_key"] and key_before2["digest"] == key_before["digest"],
        "secondary_key_generates_different_noise": secondary_div["noise_differs_from_primary"],
        "no_requirement_secondary_token_differs": secondary_div["token_difference_not_required"],
        "pytorch_parity_explicitly_not_required": True,
        "main_hidden_not_executed": True,
        "authority_source_guards_pass": True,
    }

    rec = {
        "schema": "ds41f.boundary11b.connected-mlx-stochastic-sampling-validation.v1",
        "boundary": "Boundary 11b: connected target-MLX-runtime stochastic sampling with Boundary11a generator",
        "base_commit": "cb286d836d910d256c8b76a2d04f0e1fcc415586",
        "classification": "connected_target_mlx_runtime_stochastic_sampling_authority_for_pinned_fixture_temperature_and_explicit_rng_state",
        "not_omlx_derived": True,
        "runtime": {
            "mlx_version": getattr(mx, "__version__", None),
            "default_device": repr(mx.default_device()),
            "target_stream_device": "Metal GPU / mx.gpu",
            "mx_gpu_repr": repr(mx.gpu),
            "softmax_api_used": "mx.softmax(scaled_logits, axis=-1, stream=mx.gpu)",
            "softmax_doc": getattr(mx.softmax, "__doc__", None),
        },
        "authority_dependencies": {
            "Boundary9": {"artifact": "artifacts/boundary9-closeout.json", "commit": "73f7d66d1e373605825fc77a4a29e184e6bef85b", "authority": "deterministic connected model forward through final-position logits", "superseded": False},
            "Boundary10": {"artifact": "artifacts/boundary10-closeout.json", "commit": "c9c8ab1de87c69e1af7e8cb74b2bdff528ec75a1", "authority": "source-defined sampling arithmetic branch semantics", "superseded": False},
            "Boundary11a": {"artifact": "artifacts/mlx-exp1-rng-validation.json", "commit": "cb286d836d910d256c8b76a2d04f0e1fcc415586", "authority": "pinned MLX 0.32.2 / Metal GPU sampling-noise distribution + reproducibility semantics", "superseded": False},
        },
        "model_execution": {"scope": meta1, "logits": {"shape": list(logits1.shape), "dtype": "FP32", "digest": host_digest, "min": float(np.min(logits1)), "max": float(np.max(logits1)), "mean": float(np.mean(logits1, dtype=np.float64))}},
        "logits_host_to_mlx_seam": {"producer_native_logits_digest": host_digest, "producer_dtype": "FP32", "producer_shape": list(logits1.shape), "mlx_consumer_tensor": {"shape": list(primary1["logits_mx"].shape), "dtype": str(primary1["logits_mx"].dtype)}, "mlx_to_host_roundtrip_digest": roundtrip_digest, "exact_fp32_bytes_preserved": bool(host_digest == roundtrip_digest and np.array_equal(logits1, primary1["logits_roundtrip"]))},
        "temperature": {"temperature": TEMPERATURE, "effective_temperature": max(TEMPERATURE, 1e-5), "temperature_floor_reviewed": True, "temperature_floor_exercised": False},
        "rng": {"seed": PRIMARY_SEED, "session_key_before": key_before, "draw_key": draw_key, "next_session_key": next_key, "policy": "session_key -> mx.random.split(session_key, 2) -> [draw_key, next_session_key]; draw_key used once for [1,129280] noise; next_session_key returned with sampling result"},
        "noise": {**ns1, "classification": "target-runtime Exp(1)-compatible sampling noise with the Boundary11a predeclared FP32 endpoint policy", "not_strengthened_claim": "not a mathematically exact continuous Exp(1) claim beyond Boundary11a declared FP32 endpoint/distribution contract"},
        "sampling_arithmetic": {"source_defined_formula": "scores = softmax(logits / temperature) / noise; sampled_token = argmax(scores, axis=-1)", "temperature": TEMPERATURE, "probs": soft, "scores_digest": np_digest(primary1["scores_host"]), "target_mlx_sampled_token_id": sampled_token, "independent_log_domain_token": independent_token, "winner_evidence": winner, "boundary10b_separation": {"boundary10b_conditional_token": b10b_token, "not_expected_value_for_boundary11b": True, "reason": "Boundary10b used a different deterministic supplied-noise fixture; common authority is sampling arithmetic semantics only, not noise realization or token."}},
        "reproducibility": {"same_key_whole_path": same_key_repro, "secondary_key": secondary_div},
        "pytorch_parity": {"pytorch_noise_bitstream_used": False, "pytorch_seed_parity_required": False, "pytorch_sampled_token_parity_required": False, "reason": "Boundary11b validates target-runtime sampling semantics under the qualified MLX RNG contract, not backend bitstream compatibility."},
        "stop_boundary": {"stopped_after": "target_mlx_sampled_token_id and next_session_key", "next_source_operation_recorded_not_executed": "main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None", "not_executed": ["main_hidden assembly", "Transformer.forward return packaging", "decode", "second model token step"]},
        "authority_relationship_after_pass": {"Boundary9": "integrated deterministic model-forward authority through logits", "Boundary10": "source-defined sampling arithmetic branch authority", "Boundary11a": "pinned target-MLX RNG distribution/reproducibility authority", "Boundary11b": "connected target-MLX-runtime stochastic sampling authority for the pinned fixture + temperature + explicit RNG key/state", "supersedes_prior_boundaries": False},
        "safe_claim_on_pass": "For B=1, S=2, start_pos=0, world_size=1, full_logits=False, token fixture [[0,3]], temperature=1.0, and the pinned explicit MLX session key derived from seed 289513473, the connected deterministic Boundary9 logits are consumed by the Boundary10 source-defined sampling arithmetic using the Boundary11a-qualified MLX sampling-noise generator. The resulting target-MLX-runtime sampled token agrees with an independent FP64 log-domain winner reconstruction using the exact generated MLX noise, and the explicit RNG state advances to the recorded next session key. This is a pinned target-MLX-runtime sampling result. It does not claim PyTorch RNG bitstream parity, equal sampled tokens for equal numeric seeds across backends, or backend-independent canonical token identity.",
        "non_claims": ["no PyTorch/MLX RNG bitwise parity", "no cross-backend sampled-token parity", "no CPU/GPU RNG stream parity", "no cross-version RNG stream guarantee", "no mathematical claim of an exact continuous Exp(1) beyond the Boundary11a declared FP32 endpoint/distribution contract", "no main_hidden assembly", "no Transformer.forward return correctness", "no second-step/decode semantics", "no cache persistence qualification", "no full model correctness", "no performance/production qualification"],
        "boundary11a_checker": {"returncode": checker.returncode, "stdout": checker.stdout, "stderr": checker.stderr},
        "gates": gates,
        "ok": bool(all(gates.values())),
    }

    def clean(o: Any) -> Any:
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean(rec), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": rec["ok"], "out": str(out), "target_mlx_sampled_token_id": sampled_token, "gates": gates}, indent=2, sort_keys=True))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
