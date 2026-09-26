#!/usr/bin/env python3
"""Smoke native DwarfStar prefill command stream ownership.

This moves beyond arena allocation: the C side now builds a layer-major command
stream over chunks and can submit a no-op graph that exercises carry-buffer swap
ownership.  It still does not execute DeepSeek model math.
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
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/native-commands-4k.json")
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "ds41f.m2.dwarfstar-native-prefill-commands.v1",
        "purpose": "C-side layer-major command stream and no-op graph submission skeleton",
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
            token = 1
            while remaining > 0:
                n = min(args.chunk_tokens, remaining)
                chunks.append(list(range(token, token + n)))
                token += n
                remaining -= n
            cfg = native_config_from_model(model.language_model, [len(c) for c in chunks])
            plan = native.build_plan(cfg).to_json()
            with native.create_context(cfg) as ctx:
                off = 0
                for chunk in chunks:
                    ctx.upload_tokens(off, chunk)
                    off += len(chunk)
                ctx.build_commands()
                command_count = ctx.command_count()
                command_summary = ctx.command_summary()
                first_commands = [ctx.command_at(i) for i in range(min(12, command_count))]
                last_commands = [ctx.command_at(i) for i in range(max(0, command_count - 8), command_count)]
                before = ctx.info()
                submitted = ctx.execute_noop_graph()
                after = ctx.info()
            expected_commands = 2 + plan["layer_steps"] * (4 + len(chunks)) + 2
            record["native_plan"] = plan
            record["command_count"] = command_count
            record["expected_command_count"] = expected_commands
            record["command_summary"] = command_summary
            record["first_commands"] = first_commands
            record["last_commands"] = last_commands
            record["before_noop_submit"] = before
            record["after_noop_submit"] = after
            record["submitted_commands"] = submitted
            record["gates"] = {
                "native_library_loaded": bool(record["native_version"]),
                "command_count_exact": command_count == expected_commands,
                "one_encode_per_layer_chunk": command_summary.get("encode_layer_chunk") == plan["layer_steps"] * len(chunks),
                "one_swap_per_layer": command_summary.get("swap_carry") == plan["layer_steps"],
                "one_begin_end_per_layer": command_summary.get("begin_layer") == plan["layer_steps"] and command_summary.get("end_layer") == plan["layer_steps"],
                "publish_command_per_layer": command_summary.get("publish_frontier") == plan["layer_steps"],
                "output_head_and_read_once": command_summary.get("encode_output_head") == 1 and command_summary.get("read_logits") == 1,
                "noop_submitted_all_commands": submitted == command_count,
                "tokens_remain_uploaded": after["n_tokens_uploaded"] == args.target_tokens,
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
