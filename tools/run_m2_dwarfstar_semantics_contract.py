#!/usr/bin/env python3
"""Record the superseded oMLX-compatibility bridge for DwarfStar prefill."""

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
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/official-semantics-contract.json")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "ds41f.m2.dwarfstar-prefill-omlx-compatibility-bridge.v2",
        "purpose": "historical bridge from DwarfStar prefill graph architecture to oMLX adapter behavior; not official DeepSeek semantics",
        "classification": "historical_only_superseded_not_official_qualification",
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
    }
    rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=Path(args.omlx), checkpoint_path=Path(args.checkpoint), engram_ssd_offload=True, preserve_mtp=True))
    try:
        t0 = time.perf_counter()
        model, _processor = rt.load_model()
        engine = DwarfStarPrefillEngine(model.language_model)
        record["load_seconds"] = time.perf_counter() - t0
        record["semantics_contract"] = engine.semantics_contract()
        bindings = record["semantics_contract"]["bindings"]
        graph_steps = {b["graph_step"] for b in bindings}
        required = {
            "upload_tokens",
            "upload_embeddings_hc",
            "prepare_layer_weights",
            "encode_layer_batch",
            "capture_dspark_prefill_layer",
            "publish_state_frontier",
            "seed_router_selected",
            "seed_streaming_expert_cache_layer",
            "select_output_row",
            "encode_output_head",
            "read_logits",
        }
        record["gates"] = {
            "all_required_steps_bound_for_historical_omlx_adapter": required <= graph_steps,
            "decode_policy_is_omlx_compatibility": "oMLX" in record["semantics_contract"]["decode_policy"],
            "checkpoint_read_only_policy_recorded": "read-only" in record["semantics_contract"]["checkpoint_policy"],
            "frontiers_recorded": bool(record["semantics_contract"]["frontiers"]),
            "forbidden_shortcuts_recorded": bool(record["semantics_contract"]["forbidden_shortcuts"]),
        }
        record["ok"] = all(record["gates"].values())
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        rt.close()
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
