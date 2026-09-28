#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig, OMLXDecodeSession
from tools.run_m4_block1_remainder_and_layer2_entry import ordered_bf16

SCHEMA = "ds41f.m4.actual-layer2-capture.v2"
FAILED_V1 = ROOT / "artifacts/m4/actual-layer2-capture/result.json"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def arr_info(a: np.ndarray | None) -> dict[str, Any] | None:
    if a is None:
        return None
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha(a), "nbytes": int(np.ascontiguousarray(a).nbytes)}


def rope_params_from_config(config: Any, compressed: bool) -> dict[str, Any]:
    return {
        "rope_head_dim": int(getattr(config, "rope_head_dim")),
        "rope_theta": float(getattr(config, "rope_theta")),
        "compress_rope_theta": float(getattr(config, "compress_rope_theta")),
        "effective_theta": float(getattr(config, "compress_rope_theta") if compressed else getattr(config, "rope_theta")),
        "original_seq_len": int(getattr(config, "original_seq_len")),
        "beta_fast": int(getattr(config, "beta_fast")),
        "beta_slow": int(getattr(config, "beta_slow")),
        "rope_factor": float(getattr(config, "rope_factor")),
        "compressed": bool(compressed),
    }


def rope_params_from_dict(c: dict[str, Any], compressed: bool) -> dict[str, Any]:
    return {
        "rope_head_dim": int(c["qk_rope_head_dim"]),
        "rope_theta": float(c["rope_theta"]),
        "compress_rope_theta": float(c["compress_rope_theta"]),
        "effective_theta": float(c["compress_rope_theta"] if compressed else c["rope_theta"]),
        "original_seq_len": int(c.get("original_seq_len", c.get("original_max_position_embeddings", c.get("rope_scaling", {}).get("original_max_position_embeddings", 0)))),
        "beta_fast": int(c["rope_scaling"]["beta_fast"] if isinstance(c.get("rope_scaling"), dict) and "beta_fast" in c["rope_scaling"] else c.get("beta_fast", 32)),
        "beta_slow": int(c["rope_scaling"]["beta_slow"] if isinstance(c.get("rope_scaling"), dict) and "beta_slow" in c["rope_scaling"] else c.get("beta_slow", 1)),
        "rope_factor": float(c["rope_scaling"]["factor"] if isinstance(c.get("rope_scaling"), dict) and "factor" in c["rope_scaling"] else c.get("rope_factor", 1.0)),
        "compressed": bool(compressed),
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.ascontiguousarray(v) for k, v in arrays.items() if v is not None})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class Progress:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", buffering=1)
        self.mx = None

    def close(self) -> None:
        self.f.close()

    def mark(self, stage: str, **extra: Any) -> None:
        rec = {"time": time.time(), "stage": stage, "memory": self.memory()}
        rec.update(extra)
        self.f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def memory(self) -> dict[str, Any]:
        rss = maxrss = None
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            maxrss = int(ru.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
        except Exception:
            pass
        try:
            # macOS: ru_maxrss is bytes; Linux handled above. ps gives current RSS in KiB.
            out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
            rss = int(out) * 1024 if out else None
        except Exception:
            pass
        if self.mx is None:
            try:
                self.mx = importlib.import_module("mlx.core")
            except Exception:
                self.mx = False
        def metric(name: str) -> int | None:
            try:
                if self.mx and hasattr(self.mx, name):
                    return int(getattr(self.mx, name)())
            except Exception:
                return None
            return None
        return {
            "rss_bytes": rss,
            "maxrss_bytes": maxrss,
            "mlx_active_memory_bytes": metric("get_active_memory"),
            "mlx_cache_memory_bytes": metric("get_cache_memory"),
            "mlx_peak_memory_bytes": metric("get_peak_memory"),
        }


def to_np(mx: Any, a: Any) -> np.ndarray | None:
    if a is None:
        return None
    mx.eval(a)
    try:
        if "bfloat16" in str(a.dtype):
            return np.asarray(a.view(mx.uint16)).astype(np.uint16, copy=False)
    except Exception:
        pass
    return np.asarray(a)


def summary_mx(mx: Any, a: Any, *, full: bool = False) -> dict[str, Any] | None:
    if a is None:
        return None
    arr = to_np(mx, a)
    out = arr_info(arr)
    if full or (arr is not None and arr.size <= 512):
        out["values"] = arr.tolist()
    return out


# Phase E: all source reconstruction imports and arithmetic are intentionally local to this process only.
def phase_expected(args: argparse.Namespace) -> int:
    ck = Path(args.checkpoint); omlx = Path(args.omlx_path)
    if str(omlx) not in sys.path:
        sys.path.insert(0, str(omlx))
    import mlx.core as mx
    from omlx.patches.deepseek_v41.quantization import pack_activation
    from omlx.patches.deepseek_v41.packed_attention import rounded_packed_attention
    from tools.run_m4_layer2_sparse_topology import build_repaired_entry, u16_from_mx
    from tools.run_native_first_incremental_block1_layer2_entry_validation import project, cfg, DIM, D, MIX, HCD, rms_eps
    from tools.run_native_layer24_25_connected_validation import moe_layer
    from tools.run_official_hyper_connections_fixture import mmap, shard, hc_mixes, hc_pre, hc_post

    c, prefill, b1, l2 = build_repaired_entry(ck)
    q = mx.array(np.ascontiguousarray(l2["attn_path"]["q"])).view(mx.bfloat16)
    win = mx.array(np.ascontiguousarray(l2["attn_path"]["window_post"])).view(mx.bfloat16)
    comp = mx.array(np.ascontiguousarray(prefill.visible_value_arrays["compress_kv.2.visible"][:, :1, :])).view(mx.bfloat16)
    packed_win = pack_activation(win, bits=8, group_size=32, e4m3_scale=False)
    packed_comp = pack_activation(comp, bits=4, group_size=16, e4m3_scale=True)
    wi = np.full((1, 1, 128), -1, np.int32); wi[0, 0, -3:] = [0, 1, 2]
    ci = np.asarray(l2["indexer"]["topk_omlx_ci"], np.int32)
    sink = np.ascontiguousarray(mmap(shard(ck, "layers.2.attn.attn_sink"), "layers.2.attn.attn_sink", np.float32, (64,)))
    scale = float(np.float32(D ** -0.5))
    op = rounded_packed_attention(q, packed_win, packed_comp, mx.array(wi), mx.array(ci), mx.array(sink), scale)
    op_u16 = u16_from_mx(mx, op)
    pre_inverse_rope = op_u16
    inv, _woa, attn_out = project(2, op_u16, l2["attn_path"]["cos"], l2["attn_path"]["sin"])
    eps = float(c["rms_norm_eps"]); x = b1["x_out"]
    hsh = shard(ck, "layers.2.hc_attn_fn")
    afn = np.ascontiguousarray(mmap(hsh, "layers.2.hc_attn_fn", np.float32, (MIX, HCD)))
    abase = np.ascontiguousarray(mmap(hsh, "layers.2.hc_attn_base", np.float32, (MIX,)))
    ascale = np.ascontiguousarray(mmap(hsh, "layers.2.hc_attn_scale", np.float32, (3,)))
    _, _, _, _, attn_pre, attn_post, attn_comb = hc_mixes(x, afn, ascale, abase, eps, int(c["hc_sinkhorn_iters"]), float(c["hc_eps"]))
    post_attn = hc_post(attn_out, x, attn_post, attn_comb)
    fsh = shard(ck, "layers.2.hc_ffn_fn")
    ffn = np.ascontiguousarray(mmap(fsh, "layers.2.hc_ffn_fn", np.float32, (MIX, HCD)))
    fbase = np.ascontiguousarray(mmap(fsh, "layers.2.hc_ffn_base", np.float32, (MIX,)))
    fscale = np.ascontiguousarray(mmap(fsh, "layers.2.hc_ffn_scale", np.float32, (3,)))
    _, _, _, _, ffn_pre, ffn_post, ffn_comb = hc_mixes(post_attn, ffn, fscale, fbase, eps, int(c["hc_sinkhorn_iters"]), float(c["hc_eps"]))
    fh = hc_pre(post_attn, attn_pre)
    fnw = np.ascontiguousarray(mmap(shard(ck, "layers.2.ffn_norm.weight"), "layers.2.ffn_norm.weight", np.uint16, (DIM,)))
    moe_in = rms_eps(fh.reshape(1, DIM), fnw, eps).reshape(1, 1, DIM)
    moe = moe_layer(ck, c, 2, moe_in)
    xout = hc_post(moe["final"], post_attn, ffn_post, ffn_comb)

    arrays = {
        "pre_inverse_rope": pre_inverse_rope,
        "sparse_q": np.ascontiguousarray(l2["attn_path"]["q"]),
        "sparse_packed_window_kv": to_np(mx, packed_win),
        "sparse_packed_compressed_kv": to_np(mx, packed_comp),
        "sparse_wi": wi,
        "sparse_ci": ci,
        "sparse_sink": sink,
        "sparse_scale": np.array(scale, dtype=np.float32),
        "inverse_rope": inv,
        "attention_output": attn_out,
        "post_attention_hc": post_attn,
        "moe_input": moe_in,
        "route_ids": moe["idx"],
        "route_weights": moe["weights"],
        "moe_output": moe["final"],
        "output_h": xout,
        "returned_pre": ffn_pre,
        "pending_kv": l2["compressor"]["kv_state_slot0"],
        "pending_gate": l2["compressor"]["score_state_slot0"],
    }
    out_npz = Path(args.expected_npz); save_npz(out_npz, arrays)
    meta = {
        "schema": "ds41f.m4.layer2-expected-boundaries.v1",
        "checkpoint": str(ck), "omlx_path": str(omlx),
        "process_isolation": "Phase E source-derived reconstruction only; process exits before Phase A model load",
        "arrays": {k: arr_info(v) for k, v in arrays.items()},
        "rope": {"params": rope_params_from_dict(c, True), "positions": [2], "absolute_position": 2, "layer": 2},
        "sparse_topology_artifact": "artifacts/m4/layer2-sparse-topology/result.json",
        "contains_checkpoint_weights": False, "contains_expert_weight_tensors": False,
    }
    Path(args.expected_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.expected_json).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(args.expected_json); print(args.expected_npz)
    return 0


def phase_actual(args: argparse.Namespace) -> int:
    ck = Path(args.checkpoint); omlx = Path(args.omlx_path)
    progress = Progress(Path(args.progress))
    progress.mark("process_start", phase="A")
    rt = None
    try:
        from tools.run_m4_omlx_base_decode_qualification import build_prefill_state
        progress.mark("prefill_state_begin")
        prefill_result = build_prefill_state(ck, Path(args.native_out_dir), [0, 3], require_ok=True)
        state = prefill_result.continuation_state
        prefill_artifact_summary = {"ok": bool(prefill_result.ok), "token_frontier": int(state.token_frontier), "committed": bool(state.committed)}
        progress.mark("prefill_state_complete", prefill=prefill_artifact_summary)
        del prefill_result
        gc.collect()
        progress.mark("prefill_scaffolding_released")
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                before = progress.memory(); mx.clear_cache(); after = progress.memory()
                progress.mark("mlx_cache_clear", before=before, after=after)
        except Exception as exc:
            progress.mark("mlx_cache_clear_unavailable", error=repr(exc))

        if str(omlx) not in sys.path:
            sys.path.insert(0, str(omlx))
        import mlx.core as mx
        lang = importlib.import_module("omlx.patches.deepseek_v41.language")
        cfg = OMLXDecodeConfig(omlx_path=omlx, checkpoint_path=ck, engram_ssd_offload=True, preserve_mtp=False)
        progress.mark("omlx_runtime_construct_begin")
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=omlx, checkpoint_path=ck, engram_ssd_offload=True, preserve_mtp=False))
        progress.mark("omlx_runtime_construct_complete")
        progress.mark("model_load_begin")
        model, _processor = rt.load_model()
        lm = getattr(model, "language_model", model)
        progress.mark("model_load_complete")
        progress.mark("admission_begin")
        sess = OMLXDecodeSession.from_prefill_state(model, state, cfg)
        pre_offsets = [int(c.size()) for c in sess.cache]
        progress.mark("admission_complete", pre_offsets=pre_offsets)
        del state
        gc.collect()

        cap: dict[str, Any] = {"arrays": {}, "summaries": {}, "events": []}
        active = {"block2": False, "attn2": False}
        target_block = lm.layers[2]; target_attn = target_block.attn; target_moe = target_block.ffn; target_gate = target_moe.gate
        orig_block = lang.Block.__call__; orig_attn = lang.Attention.__call__; orig_moe = lang.MoE.__call__; orig_gate = lang.Gate.__call__; orig_rope = lang.rope; orig_hp = lang.hc_post; orig_sparse = lang.packed_sparse_attention

        def layer_idx(self: Any) -> int | None:
            for i, b in enumerate(lm.layers):
                if b is self:
                    return i
            return None

        def cache_digest_summary(cache: Any, *, include_pending_rows: bool = False) -> dict[str, Any]:
            out = {}
            for i in [0, 1, 2, 3, 4, 5]:
                full = include_pending_rows and i in (4, 5)
                out[str(i)] = summary_mx(mx, cache.cache[i], full=full)
            return out

        def shared_summary(shared: dict[str, Any]) -> dict[str, Any]:
            out = {}
            for k, v in shared.items():
                if hasattr(v, "shape"):
                    out[k] = summary_mx(mx, v, full=(k == "idx"))
            return out

        def block_call(self, h, pre, cache, shared, start, image_mask):
            li = layer_idx(self)
            if li in (0, 1, 2): progress.mark(f"layer{li}_enter")
            if self is target_block:
                progress.mark("layer2_enter")
                active["block2"] = True
                cap["arrays"]["block_input_h"] = to_np(mx, h)
                cap["arrays"]["block_input_pre"] = to_np(mx, pre)
                cap["summaries"]["block_entry_cache"] = cache_digest_summary(cache, include_pending_rows=True)
                out = orig_block(self, h, pre, cache, shared, start, image_mask)
                mx.eval(out[0], out[1])
                cap["arrays"]["output_h"] = to_np(mx, out[0])
                cap["arrays"]["returned_pre"] = to_np(mx, out[1])
                cap["summaries"]["block_exit_cache"] = cache_digest_summary(cache, include_pending_rows=True)
                cap["summaries"]["block_exit_shared"] = shared_summary(shared)
                active["block2"] = False
                progress.mark("layer2_exit")
            else:
                out = orig_block(self, h, pre, cache, shared, start, image_mask)
            if li in (0, 1, 2): progress.mark(f"layer{li}_exit")
            return out

        def attn_call(self, x, cache, shared, start):
            if self is target_attn:
                progress.mark("layer2_attention_enter")
                active["attn2"] = True
                out = orig_attn(self, x, cache, shared, start)
                mx.eval(out)
                cap["arrays"]["attention_output"] = to_np(mx, out)
                cap["summaries"]["attention_exit_cache"] = cache_digest_summary(cache, include_pending_rows=True)
                cap["summaries"]["attention_exit_shared"] = shared_summary(shared)
                active["attn2"] = False
                progress.mark("layer2_attention_exit")
                return out
            return orig_attn(self, x, cache, shared, start)

        def sparse_wrap(q, kv, pooled, idx, ci, sink, scale):
            if active["attn2"] and "pre_inverse_rope" not in cap["arrays"]:
                cap["arrays"]["sparse_q"] = to_np(mx, q)
                cap["arrays"]["sparse_packed_window_kv"] = to_np(mx, kv)
                cap["arrays"]["sparse_packed_compressed_kv"] = to_np(mx, pooled)
                cap["arrays"]["sparse_wi"] = to_np(mx, idx)
                cap["arrays"]["sparse_ci"] = to_np(mx, ci)
                cap["arrays"]["sparse_sink"] = to_np(mx, sink)
                cap["arrays"]["sparse_scale"] = np.array(float(scale), dtype=np.float32)
            out = orig_sparse(q, kv, pooled, idx, ci, sink, scale)
            if active["attn2"] and "pre_inverse_rope" not in cap["arrays"]:
                mx.eval(out); cap["arrays"]["pre_inverse_rope"] = to_np(mx, out)
            return out

        def rope_wrap(x, positions, config, compressed, inverse=False):
            if active["attn2"] and inverse and "pre_inverse_rope" not in cap["arrays"]:
                cap["arrays"]["pre_inverse_rope"] = to_np(mx, x)
            out = orig_rope(x, positions, config, compressed, inverse=inverse)
            if active["attn2"] and inverse and "inverse_rope" not in cap["arrays"]:
                mx.eval(out)
                cap["arrays"]["inverse_rope"] = to_np(mx, out)
                cap["arrays"]["inverse_rope_positions"] = to_np(mx, positions)
                cap["summaries"]["inverse_rope_call"] = {
                    "positions": arr_info(cap["arrays"].get("inverse_rope_positions")),
                    "positions_values": None if cap["arrays"].get("inverse_rope_positions") is None else cap["arrays"]["inverse_rope_positions"].tolist(),
                    "compressed": bool(compressed),
                    "inverse": bool(inverse),
                    "params": rope_params_from_config(config, bool(compressed)),
                }
            return out

        def hp_wrap(x, residual, post, comb):
            out = orig_hp(x, residual, post, comb)
            if active["block2"]:
                mx.eval(out)
                arr = to_np(mx, out)
                if "post_attention_hc" not in cap["arrays"]:
                    cap["arrays"]["post_attention_hc"] = arr
                else:
                    cap["arrays"].setdefault("post_ffn_hc", arr)
            return out

        def moe_call(self, x, image_mask):
            if self is target_moe:
                progress.mark("layer2_moe_enter")
                cap["arrays"]["moe_input"] = to_np(mx, x)
                out = orig_moe(self, x, image_mask)
                mx.eval(out); cap["arrays"]["moe_output"] = to_np(mx, out)
                progress.mark("layer2_moe_exit")
                return out
            return orig_moe(self, x, image_mask)

        def gate_call(self, x, image_mask):
            if self is target_gate:
                out = orig_gate(self, x, image_mask)
                mx.eval(out[0], out[1])
                cap["arrays"]["route_ids"] = to_np(mx, out[0])
                cap["arrays"]["route_weights"] = to_np(mx, out[1])
                return out
            return orig_gate(self, x, image_mask)

        try:
            progress.mark("wrapper_install_begin")
            lang.Block.__call__ = block_call; lang.Attention.__call__ = attn_call; lang.packed_sparse_attention = sparse_wrap; lang.rope = rope_wrap; lang.hc_post = hp_wrap; lang.MoE.__call__ = moe_call; lang.Gate.__call__ = gate_call
            progress.mark("wrapper_install_complete")
            progress.mark("forward_begin")
            logits = lm._forward(mx.array([[15]], mx.int64), cache=sess.cache)
            mx.eval(logits)
            progress.mark("forward_complete")
        finally:
            lang.Block.__call__ = orig_block; lang.Attention.__call__ = orig_attn; lang.packed_sparse_attention = orig_sparse; lang.rope = orig_rope; lang.hc_post = orig_hp; lang.MoE.__call__ = orig_moe; lang.Gate.__call__ = orig_gate

        final_offsets = [int(c.size()) for c in sess.cache]
        arrays = {k: v for k, v in cap["arrays"].items() if isinstance(v, np.ndarray)}
        save_npz(Path(args.actual_npz), arrays)
        meta = {
            "schema": "ds41f.m4.layer2-actual-boundaries.v1", "checkpoint": str(ck), "omlx_path": str(omlx),
            "attempt_status": "COMPLETE", "real_loaded_omlx": True, "token": 15,
            "production_target_path": "qualified PrefillContinuationState -> OMLXDecodeStateAdapter -> actual loaded model -> LanguageModel._forward(token15)",
            "phase_a_excludes_source_reconstruction": True,
            "pre_offsets": pre_offsets, "no_prefix_replay": all(x == 2 for x in pre_offsets), "final_offsets": final_offsets,
            "arrays": {k: arr_info(v) for k, v in arrays.items()}, "summaries": cap["summaries"],
        }
        Path(args.actual_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.actual_json).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        progress.mark("capture_written", actual_json=args.actual_json, actual_npz=args.actual_npz)
        return 0
    finally:
        if rt is not None:
            try: rt.close()
            finally: progress.mark("runtime_close_complete")
        progress.close()


def align_boundary_shape(actual: np.ndarray | None, expected: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if actual is None or expected is None or actual.shape == expected.shape:
        return actual, expected
    a = np.squeeze(actual)
    e = np.squeeze(expected)
    if a.shape == e.shape:
        return a, e
    return actual, expected


def cmp_exact(actual: np.ndarray | None, expected: np.ndarray | None) -> dict[str, Any]:
    actual, expected = align_boundary_shape(actual, expected)
    ok = actual is not None and expected is not None and actual.shape == expected.shape and np.array_equal(actual, expected)
    return {"contract": "exact", "actual": arr_info(actual), "expected": arr_info(expected), "within_contract": bool(ok), "exact": bool(ok)}


def cmp_bf16(actual: np.ndarray | None, expected: np.ndarray | None, tol: int, contract: str) -> dict[str, Any]:
    actual, expected = align_boundary_shape(actual, expected)
    if actual is None or expected is None or actual.shape != expected.shape or actual.dtype != np.uint16 or expected.dtype != np.uint16:
        return {"contract": contract, "actual": arr_info(actual), "expected": arr_info(expected), "within_contract": False}
    u = np.abs(ordered_bf16(actual) - ordered_bf16(expected))
    return {"contract": contract, "actual": arr_info(actual), "expected": arr_info(expected), "max_bf16_ulp": int(u.max()) if u.size else 0, "max_bf16_ulp_lte": int(tol), "within_contract": bool((int(u.max()) if u.size else 0) <= tol), "exact": bool(np.array_equal(actual, expected))}


def cmp_f32(actual: np.ndarray | None, expected: np.ndarray | None, tol: float, contract: str) -> dict[str, Any]:
    actual, expected = align_boundary_shape(actual, expected)
    if actual is None or expected is None or actual.shape != expected.shape:
        return {"contract": contract, "actual": arr_info(actual), "expected": arr_info(expected), "within_contract": False}
    d = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {"contract": contract, "actual": arr_info(actual), "expected": arr_info(expected), "max_abs_diff": float(d.max()) if d.size else 0.0, "max_abs_lte": float(tol), "within_contract": bool((float(d.max()) if d.size else 0.0) <= tol), "exact": bool(np.array_equal(actual, expected))}


def cmp_sparse_input(name: str, actual: np.ndarray | None, expected: np.ndarray | None) -> dict[str, Any]:
    if actual is None or expected is None:
        return {"contract": "sparse input present and exact/equivalent", "actual": arr_info(actual), "expected": arr_info(expected), "within_contract": False}
    if actual.dtype == np.uint16 or expected.dtype == np.uint16:
        return cmp_bf16(actual, expected, 1, f"{name}: BF16 sparse-input boundary max ULP <= 1")
    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(expected.dtype, np.floating):
        return cmp_f32(actual, expected, 0.0, f"{name}: exact float sparse scalar/weight boundary")
    return cmp_exact(actual, expected)


def phase_compare(args: argparse.Namespace) -> int:
    exp = load_npz(Path(args.expected_npz)); act = load_npz(Path(args.actual_npz))
    actual_meta = json.loads(Path(args.actual_json).read_text())
    expected_meta = json.loads(Path(args.expected_json).read_text()) if Path(args.expected_json).exists() else {}
    comps = {
        "pre_inverse_rope": cmp_bf16(act.get("pre_inverse_rope"), exp.get("pre_inverse_rope"), 1, "qualified Layer2 padded sparse BF16 output max ULP <= 1"),
        "inverse_rope": cmp_bf16(act.get("inverse_rope"), exp.get("inverse_rope"), 1, "attention output projection inverse rotary BF16 max ULP <= 1"),
        "attention_return": cmp_bf16(act.get("attention_output"), exp.get("attention_output"), 1, "attention output projection final BF16 max ULP <= 1"),
        "post_attention_hc": cmp_bf16(act.get("post_attention_hc"), exp.get("post_attention_hc"), 1, "HC attention post BF16 max ULP <= 1"),
        "moe_input": cmp_bf16(act.get("moe_input"), exp.get("moe_input"), 1, "HC FFN/MoE subblock norm BF16 max ULP <= 1"),
        "route_ids": cmp_exact(act.get("route_ids"), exp.get("route_ids")),
        "route_weights": cmp_f32(act.get("route_weights"), exp.get("route_weights"), 1e-3, "MoE gate scaled routing weights FP32 max abs <= 1e-3"),
        "moe_output": cmp_bf16(act.get("moe_output"), exp.get("moe_output"), 1, "full MoE output BF16 max ULP <= 1"),
        "block_output_h": cmp_bf16(act.get("output_h"), exp.get("output_h"), 1, "HC FFN post Block output BF16 max ULP <= 1"),
        "block_returned_pre": cmp_f32(act.get("returned_pre"), exp.get("returned_pre"), 1e-4, "returned HC pre-mix FP32 exact/HC f32 max abs <= 1e-4"),
    }
    sparse_input_comparisons = {}
    if not comps["pre_inverse_rope"].get("within_contract", False):
        for name in ["sparse_q", "sparse_packed_window_kv", "sparse_packed_compressed_kv", "sparse_wi", "sparse_ci", "sparse_sink", "sparse_scale"]:
            sparse_input_comparisons[name] = cmp_sparse_input(name, act.get(name), exp.get(name))
    ok = all(v.get("within_contract", False) for v in comps.values())
    state_ok = {
        "final_merged_layer2_frontier_eq_3": bool(actual_meta.get("final_offsets", [None, None, None])[2] == 3),
        "window_cache_includes_token15": bool(actual_meta.get("final_offsets", [None, None, None])[2] == 3),
        "source2_slot2_digest_recorded": bool(actual_meta.get("summaries", {}).get("block_exit_cache", {}).get("2", {}).get("sha256")),
        "source2_slot3_digest_recorded": bool(actual_meta.get("summaries", {}).get("block_exit_cache", {}).get("3", {}).get("sha256")),
        "pending_kv_row_recorded": bool(actual_meta.get("summaries", {}).get("block_exit_cache", {}).get("4", {}).get("values") is not None),
        "pending_gate_row_recorded": bool(actual_meta.get("summaries", {}).get("block_exit_cache", {}).get("5", {}).get("values") is not None),
        "shared_kv_digest_recorded": bool(actual_meta.get("summaries", {}).get("attention_exit_shared", {}).get("kv", {}).get("sha256")),
        "shared_idx_full_recorded": bool(actual_meta.get("summaries", {}).get("attention_exit_shared", {}).get("idx", {}).get("values") is not None),
    }
    ok = ok and all(state_ok.values())
    first = None if ok else next((k for k, v in comps.items() if not v.get("within_contract", False)), None) or next((k for k, v in state_ok.items() if not v), "unknown")
    if not comps["pre_inverse_rope"].get("within_contract", False):
        classification = "SPARSE_OUTPUT_DIVERGENCE"
    elif not comps["inverse_rope"].get("within_contract", False):
        classification = "INVERSE_ROPE_DIVERGENCE"
    elif comps["pre_inverse_rope"].get("within_contract", False) and comps["inverse_rope"].get("within_contract", False):
        classification = "INVERSE_ROPE_COMPLETE / NEXT PROJECTION FRONTIER"
    else:
        classification = "NUMERICAL CONTRACT INCOMPLETE"
    rec = {
        "schema": SCHEMA, "attempt_status": "COMPLETE", "failed_attempt_artifact": str(FAILED_V1),
        "historical_failed_attempt_preserved": FAILED_V1.exists(), "comparison_phase": "C lightweight process",
        "classification": classification,
        "rope": {"actual": actual_meta.get("summaries", {}).get("inverse_rope_call"), "expected": expected_meta.get("rope")},
        "comparisons": comps, "sparse_input_comparisons": sparse_input_comparisons, "post_layer2_state": state_ok,
        "layer2_incremental_block": "COMPLETE" if ok else "INCOMPLETE", "ok": bool(ok), "first_unresolved_boundary": first,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(out); print("ok", ok, "frontier", first)
    return 0 if ok else 2


def run_subphase(phase: str, args: argparse.Namespace) -> int:
    cmd = [sys.executable, __file__, "--phase", phase, "--checkpoint", args.checkpoint, "--omlx-path", args.omlx_path, "--expected-npz", args.expected_npz, "--expected-json", args.expected_json, "--actual-npz", args.actual_npz, "--actual-json", args.actual_json, "--progress", args.progress, "--native-out-dir", args.native_out_dir, "--out", args.out]
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "E", "A", "C"], default="all")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    ap.add_argument("--omlx-path", default=os.environ.get("DS41F_OMLX", str(DEFAULT_OMLX)))
    base = "artifacts/m4/actual-layer2-capture"
    ap.add_argument("--expected-npz", default=f"{base}/expected-boundaries.npz")
    ap.add_argument("--expected-json", default=f"{base}/expected-boundaries.json")
    ap.add_argument("--actual-npz", default=f"{base}/actual-boundaries.npz")
    ap.add_argument("--actual-json", default=f"{base}/actual-boundaries.json")
    ap.add_argument("--progress", default=f"{base}/phase-a-progress.jsonl")
    ap.add_argument("--native-out-dir", default=f"{base}/native")
    ap.add_argument("--out", default=f"{base}/completed-comparison.json")
    args = ap.parse_args()
    if args.phase == "E": return phase_expected(args)
    if args.phase == "A": return phase_actual(args)
    if args.phase == "C": return phase_compare(args)
    for ph in ["E", "A", "C"]:
        rc = run_subphase(ph, args)
        if rc != 0:
            print(f"phase {ph} failed with {rc}")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
