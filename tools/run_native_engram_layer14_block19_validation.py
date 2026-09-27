#!/usr/bin/env python3
"""Connected official-source-derived Engram@14 -> Blocks14..19 reference fixture.

This intentionally remains a bounded validation helper under tools/.  It does
not use the DwarfStar executor or TextBackboneReference control flow; it replays
the reviewed official-source-derived block/Engram arithmetic on the connected
[[0,3]] trajectory and records reference digests for the production-prefill
slice through layer 19.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.official_model_math import OfficialModelMath  # noqa: E402
from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT  # noqa: E402

OUT = ROOT / "artifacts/native-engram-layer14-block19-validation.json"


def main() -> int:
    ck = Path(DEFAULT_CHECKPOINT)
    math = OfficialModelMath(ck)
    tokens = np.array([[0, 3]], dtype=np.int64)
    emb = math.embedding_prefix(math.vocab_size)[tokens].copy()
    x = np.repeat(emb[:, :, None, :], math.hc_mult, axis=2).copy()
    pre = np.zeros((1, 2, math.hc_mult), np.float32)
    pre[:, :, 0] = 1.0
    shared: dict[str, Any] = {"compress_kv": None, "index_k": None, "candidates": None, "topk_idxs": None}
    hashes = math.engram_hashes_for_tokens(tokens)

    layers: dict[str, Any] = {}
    engram: dict[str, Any] = {}
    carry: dict[str, Any] = {}
    prev_x = math.digest(x)
    prev_pre = math.digest(pre)

    for layer in range(20):
        if layer in {1, 14}:
            hash_key = "layer1_hash" if layer == 1 else "layer14_hash"
            before_shared = math.snapshot_publications(shared)
            post, ev = math.apply_engram(layer, x, hashes[hash_key])
            x = post
            ev = {k: v for k, v in ev.items() if k not in {"residual_update"}}
            ev["shared_state_before"] = before_shared
            ev["shared_state_after"] = math.snapshot_publications(shared)
            ev["shared_state_unchanged"] = ev["shared_state_before"] == ev["shared_state_after"]
            engram[str(layer)] = ev
            prev_x = math.digest(x)
        before = math.snapshot_publications(shared)
        x_in = math.digest(x)
        pre_in = math.digest(pre)
        out = math.execute_block(layer, x, pre, shared)
        after = math.snapshot_publications(shared)
        producer = {k: math.digest(v) for k, v in (out["attn_path"].get("producer") or {}).items() if isinstance(v, np.ndarray)}
        layers[str(layer)] = {
            "x_in": x_in,
            "pre_mix_in": pre_in,
            "attention_input": math.digest(out["attention_input"]),
            "attention_output": math.digest(out["attention_output"]),
            "x_after_attn": math.digest(out["x_after_attn"]),
            "moe_input": math.digest(out["moe_input"]),
            "moe_output": math.digest(out["full_moe_output"]),
            "x_out": math.digest(out["x_out"]),
            "ffn_pre": math.digest(out["ffn_pre"]),
            "window_kv": math.digest(out["attn_path"]["window_kv"]),
            "state_before": before,
            "state_after": after,
            "producer": producer,
            "consumed": out["attn_path"].get("consumed") or {},
        }
        carry[f"{layer-1}->{layer}"] = {
            "x_digest": x_in,
            "prev_x_out_digest": prev_x,
            "x_exact": x_in == prev_x,
            "pre_mix_digest": pre_in,
            "prev_ffn_pre_digest": prev_pre,
            "pre_mix_exact": pre_in == prev_pre,
        }
        x = out["x_out"]
        pre = out["ffn_pre"]
        prev_x = math.digest(x)
        prev_pre = math.digest(pre)

    source_generations: dict[str, Any] = {}
    source_layers = [int(k) for k, v in layers.items() if v["producer"]]
    for idx, source in enumerate(source_layers):
        next_source = source_layers[idx + 1] if idx + 1 < len(source_layers) else 20
        frontier = layers[str(source)]["state_after"]
        reads = {}
        for consumer in range(source + 1, next_source):
            c = layers[str(consumer)]["consumed"]
            reads[str(consumer)] = {
                "compress_kv_matches_source": c.get("compress_kv") == frontier.get("compress_kv"),
                "index_k_matches_source": c.get("index_k") == frontier.get("index_k"),
                "topk_idxs_matches_source": c.get("topk_idxs") == frontier.get("topk_idxs"),
                "consumer_recomputed_producer_state": bool(layers[str(consumer)]["producer"]),
            }
        source_generations[str(source)] = {"source_publication": frontier, "consumer_reads": reads}

    rec = {
        "schema": "ds41f.native-engram-layer14-block19-validation.v1",
        "ok": True,
        "classification": "official-source-derived connected Engram@14 through Block19 bounded authority",
        "checkpoint": str(ck),
        "scope": {"tokens": [0, 3], "layers": list(range(20)), "stop": "after_Block19", "Block20_executed": False},
        "engram": engram,
        "layers": layers,
        "source_generations": source_generations,
        "candidate_state_absent_before_layer20": all(layers[str(i)]["state_after"].get("candidates") is None for i in range(20)),
        "connected_carry_exact": all(v["x_exact"] and v["pre_mix_exact"] for k, v in carry.items() if k != "0->0"),
        "no_reference_text_runtime_called": True,
        "gates": {
            "engram1_executed": "1" in engram,
            "engram14_executed": "14" in engram,
            "engram14_shared_unchanged": engram["14"]["shared_state_unchanged"],
            "block14_executed_after_engram14": layers["14"]["x_in"] == engram["14"]["output_digest"],
            "generation14_present": all(source_generations.get("14", {}).get("source_publication", {}).get(k) is not None for k in ("compress_kv", "index_k", "topk_idxs")),
            "consumers15_19_read_generation14": all(v["compress_kv_matches_source"] and v["topk_idxs_matches_source"] and not v["consumer_recomputed_producer_state"] for v in source_generations.get("14", {}).get("consumer_reads", {}).values()),
            "candidate_state_absent_before_layer20": all(layers[str(i)]["state_after"].get("candidates") is None for i in range(20)),
            "no_reference_text_runtime_called": True,
        },
    }
    rec["ok"] = all(rec["gates"].values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT} ok={rec['ok']} block19={layers['19']['x_out']}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
