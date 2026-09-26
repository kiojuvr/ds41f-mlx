#!/usr/bin/env python3
"""Build and smoke the C-side DwarfStar prefill planner.

This is the first C/Metal-leaning step for the new prefill engine.  It does not
replace model math yet; it moves graph/arena planning into a native ABI.  Any
historical semantics bridge included in the output is oMLX-compatibility
metadata only, not official qualification.
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

from ds41f_mlx.dwarfstar_prefill import DwarfStarPrefillEngine  # noqa: E402
from ds41f_mlx.native_prefill import (  # noqa: E402
    DS4_AUTHORITY_CAVEAT,
    DS4_AUTHORITY_REMOTE,
    DS4_AUTHORITY_SHA,
    compile_native_prefill_library,
    load_native_prefill_library,
)
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--chunk-tokens", type=int, default=2048)
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/native-plan-4k.json")
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "ds41f.m2.dwarfstar-native-prefill-plan.v1",
        "purpose": "C-side DwarfStar prefill graph/arena planner; no model math replacement yet",
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
            lm = model.language_model
            chunks = []
            remaining = args.target_tokens
            while remaining > 0:
                n = min(args.chunk_tokens, remaining)
                chunks.append([0] * n)
                remaining -= n
            engine = DwarfStarPrefillEngine(lm)
            py_plan = engine.plan(chunks).summary()
            native_plan = engine.native_plan(chunks, native)
            record["python_plan"] = py_plan
            record["native_plan"] = native_plan
            record["semantics_contract"] = engine.semantics_contract()
            record["gates"] = {
                "native_library_loaded": bool(record["native_version"]),
                "layer_steps_match_python_plan": native_plan["layer_steps"] == py_plan["step_counts"]["encode_layer_batch"],
                "token_count_exact": native_plan["n_tokens"] == args.target_tokens,
                "publication_frontier_count_exact": native_plan["publication_frontiers"] == len(py_plan["publication_frontiers"]),
                "carry_buffer_is_full_prompt_ping_pong": native_plan["carry_buffer_bytes"] == 2 * native_plan["full_prompt_hidden_bytes"],
                "last_chunk_logits_smaller_than_full_logits_when_multichunk": native_plan["last_chunk_logits_bytes"] <= native_plan["full_prompt_logits_bytes"],
                "official_semantics_contract_present": bool(record["semantics_contract"].get("bindings")),
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
