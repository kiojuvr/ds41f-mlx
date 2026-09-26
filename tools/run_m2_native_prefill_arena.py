#!/usr/bin/env python3
"""Smoke native prefill arena/carry-buffer ownership.

This is a C-side ownership skeleton only: token upload and ping-pong carry arena
allocation/swap.  It does not execute DeepSeek math yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.native_prefill import (  # noqa: E402
    DS4_AUTHORITY_CAVEAT,
    DS4_AUTHORITY_REMOTE,
    DS4_AUTHORITY_SHA,
    compile_native_prefill_library,
    load_native_prefill_library,
    native_config_from_model,
)
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--chunk-tokens", type=int, default=2048)
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/native-arena-4k.json")
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "ds41f.m2.dwarfstar-native-prefill-arena.v1",
        "purpose": "C-side token/carry-buffer ownership skeleton for DwarfStar-authoritative prefill",
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
        "target_tokens": args.target_tokens,
        "chunk_tokens": args.chunk_tokens,
        "ds4_authority": {
            "remote": DS4_AUTHORITY_REMOTE,
            "commit": DS4_AUTHORITY_SHA,
            "caveat": DS4_AUTHORITY_CAVEAT,
        },
    }
    try:
        lib_path = compile_native_prefill_library(Path(args.native_out_dir))
        native = load_native_prefill_library(lib_path)
        record["native_library"] = str(lib_path)
        record["native_version"] = native.version()
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=Path(args.omlx), checkpoint_path=Path(args.checkpoint), engram_ssd_offload=True, preserve_mtp=True))
        try:
            t0 = time.perf_counter()
            model, _processor = rt.load_model()
            record["load_seconds"] = time.perf_counter() - t0
            chunks = []
            remaining = args.target_tokens
            token = 1
            while remaining > 0:
                n = min(args.chunk_tokens, remaining)
                chunks.append(list(range(token, token + n)))
                token += n
                remaining -= n
            cfg = native_config_from_model(model.language_model, [len(c) for c in chunks])
            plan = native.build_plan(cfg).to_json()
            with native.create_context(cfg) as ctx:
                before = ctx.info()
                offset = 0
                for chunk in chunks:
                    ctx.upload_tokens(offset, chunk)
                    offset += len(chunk)
                after_upload = ctx.info()
                addr0 = ctx.carry_addresses()
                ctx.swap_carry()
                after_swap = ctx.info()
                addr1 = ctx.carry_addresses()
            record["native_plan"] = plan
            record["arena"] = {
                "before": before,
                "after_upload": after_upload,
                "after_swap": after_swap,
                "carry_addresses_before_swap": addr0,
                "carry_addresses_after_swap": addr1,
            }
            record["gates"] = {
                "native_library_loaded": bool(record["native_version"]),
                "arena_allocated_expected_bytes": before["carry_buffer_bytes"] == plan["carry_buffer_bytes"],
                "token_buffer_sized": before["token_buffer_bytes"] == args.target_tokens * 4,
                "tokens_uploaded_exact": after_upload["n_tokens_uploaded"] == args.target_tokens,
                "carry_swap_toggled": before["current_carry_index"] != after_swap["current_carry_index"],
                "carry_addresses_swapped": addr0["current"] == addr1["next"] and addr0["next"] == addr1["current"],
            }
            record["ok"] = all(record["gates"].values())
        finally:
            rt.close()
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
