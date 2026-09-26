#!/usr/bin/env python3
"""Inspect oMLX per-layer state needed for an M2 layer-major prototype.

This is a tiny historical oMLX-compatibility/topology inspection harness.  It
loads the checkpoint through the known-good oMLX loader only when explicitly
invoked, runs a small text-only forward pass, and records layer/cache shapes.
It is not a performance benchmark, not official DeepSeek qualification, and must
not be used as a production path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig


def arr_info(x: Any) -> dict[str, Any] | None:
    if x is None:
        return None
    return {
        "type": type(x).__name__,
        "shape": list(getattr(x, "shape", [])),
        "dtype": str(getattr(x, "dtype", "")),
        "size": int(getattr(x, "size", 0) or 0),
    }


def cache_info(cache: Any) -> dict[str, Any]:
    slots = getattr(cache, "cache", [])
    info = {
        "type": type(cache).__name__,
        "compress_ratio": getattr(cache, "compress_ratio", None),
        "slots": [arr_info(x) for x in slots],
    }
    try:
        info["size"] = int(cache.size())
    except Exception as exc:
        info["size_error"] = f"{type(exc).__name__}: {exc}"
    return info


class LayerProbe:
    def __init__(self, layer: Any, index: int, records: list[dict[str, Any]]):
        self._layer = layer
        self._index = index
        self._records = records

    def __contains__(self, key: str) -> bool:
        return key in self._layer

    def __getitem__(self, key: str) -> Any:
        return self._layer[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)

    def __call__(self, h, pre, cache, shared, start, image_mask):
        rec: dict[str, Any] = {
            "layer": self._index,
            "has_engram": "engram" in self._layer,
            "start": int(start),
            "h_in": arr_info(h),
            "pre_in": arr_info(pre),
            "cache_before": cache_info(cache),
            "shared_keys_before": sorted(shared.keys()),
        }
        t0 = time.perf_counter()
        h2, pre2 = self._layer(h, pre, cache, shared, start, image_mask)
        rec.update(
            seconds=time.perf_counter() - t0,
            h_out=arr_info(h2),
            pre_out=arr_info(pre2),
            cache_after=cache_info(cache),
            shared_keys_after=sorted(shared.keys()),
        )
        self._records.append(rec)
        return h2, pre2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ping")
    ap.add_argument("--target-tokens", type=int, default=32, help="Repeat prompt until at least this many raw tokens, then truncate")
    ap.add_argument("--out", default="artifacts/m2/layer-state/inspect-small.json")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "ds41f.m2.v41-layer-state-inspection.v1",
        "purpose": "small historical oMLX-compatibility/topology inspection for layer-major prototype planning; not performance; not official qualification",
        "classification": "historical_only_omlx_compatibility_reference_not_official_qualification",
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
        "prompt": args.prompt,
        "target_tokens": args.target_tokens,
    }
    started = time.perf_counter()
    rt = OmlxRuntime(
        OmlxRuntimeConfig(
            omlx_path=Path(args.omlx),
            checkpoint_path=Path(args.checkpoint),
            engram_ssd_offload=True,
            preserve_mtp=True,
        )
    )
    try:
        model, processor = rt.load_model()
        record["load_seconds"] = time.perf_counter() - started
        tokenizer = processor.tokenizer
        text = args.prompt
        ids = tokenizer.encode(text, add_special_tokens=False)
        while len(ids) < args.target_tokens:
            text = text + "\n" + args.prompt
            ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[: args.target_tokens]
        record["input_token_count"] = len(ids)
        record["input_ids_sha256"] = __import__("hashlib").sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        lm = model.language_model
        cfg = getattr(lm, "_config", None)
        record["config"] = {
            "n_layers": getattr(cfg, "n_layers", None),
            "dim": getattr(cfg, "dim", None),
            "hc_mult": getattr(cfg, "hc_mult", None),
            "n_heads": getattr(cfg, "n_heads", None),
            "head_dim": getattr(cfg, "head_dim", None),
            "window_size": getattr(cfg, "window_size", None),
            "compress_ratios": list(getattr(cfg, "compress_ratios", []) or []),
            "kv_source_layers": list(getattr(cfg, "kv_source_layers", []) or []),
            "index_source_layers": list(getattr(cfg, "index_source_layers", []) or []),
            "engram_layer_ids": list(getattr(cfg, "engram_layer_ids", []) or []),
            "candidate_source_layer": getattr(cfg, "candidate_source_layer", None),
            "candidate_topk_blocks": getattr(cfg, "candidate_topk_blocks", None),
            "candidate_block_size": getattr(cfg, "candidate_block_size", None),
        }
        cache = lm.make_cache()
        record["initial_cache"] = [cache_info(c) for c in cache]
        layer_records: list[dict[str, Any]] = []
        original_layers = list(lm.layers)
        lm.layers = [LayerProbe(layer, i, layer_records) for i, layer in enumerate(original_layers)]
        import mlx.core as mx  # type: ignore

        run_start = time.perf_counter()
        logits = lm._forward(mx.array(ids)[None], cache=cache)
        mx.eval(logits)
        mx.synchronize()
        record["forward_seconds"] = time.perf_counter() - run_start
        record["logits"] = arr_info(logits)
        record["final_cache"] = [cache_info(c) for c in cache]
        record["layers"] = layer_records
        record["ok"] = True
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        try:
            if "lm" in locals() and "original_layers" in locals():
                lm.layers = original_layers
        except Exception:
            pass
        rt.close()
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
