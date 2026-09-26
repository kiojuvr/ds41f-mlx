#!/usr/bin/env python3
"""Smoke native carry-buffer mutation through the command stream.

This is still not DeepSeek math.  It replaces the previous no-op encode command
with a deterministic native write into the next carry buffer, followed by the
native swap command.  The goal is to prove command-driven buffer mutation and
ownership before official-math kernels are wired in.
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
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/native-carry-mutation-4k.json")
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "ds41f.m2.dwarfstar-native-carry-mutation.v1",
        "purpose": "native command-driven carry-buffer mutation skeleton; no DeepSeek math yet",
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
        native = load_native_prefill_library(compile_native_prefill_library(Path(args.native_out_dir)))
        record["native_version"] = native.version()
        rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=Path(args.omlx), checkpoint_path=Path(args.checkpoint), engram_ssd_offload=True, preserve_mtp=True))
        try:
            t0 = time.perf_counter()
            model, _processor = rt.load_model()
            record["load_seconds"] = time.perf_counter() - t0
            chunks = []
            remaining = args.target_tokens
            while remaining > 0:
                n = min(args.chunk_tokens, remaining)
                chunks.append([0] * n)
                remaining -= n
            cfg = native_config_from_model(model.language_model, [len(c) for c in chunks])
            plan = native.build_plan(cfg).to_json()
            with native.create_context(cfg) as ctx:
                ctx.build_commands()
                before = ctx.info()
                submitted = ctx.execute_noop_graph()
                after = ctx.info()
                summary = ctx.command_summary()
            expected_encode = plan["layer_steps"] * len(chunks)
            record["native_plan"] = plan
            record["command_summary"] = summary
            record["submitted_commands"] = submitted
            record["before"] = before
            record["after"] = after
            record["gates"] = {
                "native_library_loaded": bool(record["native_version"]),
                "all_commands_submitted": submitted == sum(summary.values()),
                "encode_chunks_executed_exact": after["executed_encode_chunks"] == expected_encode,
                "last_layer_exact": after["last_encoded_layer"] == plan["layer_steps"] - 1,
                "last_chunk_exact": after["last_encoded_chunk"] == len(chunks) - 1,
                "carry_checksum_changed": after["current_carry_checksum"] != before["current_carry_checksum"] or after["next_carry_checksum"] != before["next_carry_checksum"],
                "final_carry_index_even_swaps": after["current_carry_index"] == before["current_carry_index"],
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
