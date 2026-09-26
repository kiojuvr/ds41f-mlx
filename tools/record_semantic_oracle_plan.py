#!/usr/bin/env python3
"""Record the repaired semantic-oracle plan.

This emits metadata only. It does not execute model math or generate numerical
expected tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_PY_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
KERNEL_PY_SHA = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
DS4 = "https://github.com/antirez/ds4.git@0aaea5a238fb41a35106a551e73c8409dfb751ac"


def stage(name: str, classification: str, status: str, authority: dict[str, Any], next_action: str, non_claims: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "classification": classification,
        "status": status,
        "authority": authority,
        "next_action": next_action,
        "non_claims": non_claims,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/semantic-oracle-plan.json")
    args = ap.parse_args()

    record = {
        "schema": "ds41f.semantic-oracle-plan.v1",
        "purpose": "metadata plan for replacing oMLX-derived correctness evidence with official-reference-derived fixtures",
        "model_math_status": "frozen_until_official_semantics_review",
        "authorities": {
            "model_data": CHECKPOINT,
            "candidate_model_semantics": "docs/official-semantics-authority.md",
            "execution_architecture": DS4,
            "omlx_role": "compatibility/performance/implementation donor only",
        },
        "forbidden_promotions": [
            "matches_oMLX_forward_to_official_correctness",
            "matches_oMLX_logits_to_official_logits_oracle",
            "matches_oMLX_cache_state_to_official_semantics",
            "dwarfstar_topology_to_model_math_correctness",
        ],
        "required_fixture_metadata": [
            "classification",
            "not_omlx_derived",
            "checkpoint_tensor_or_reference_file_identity",
            "operation_contract",
            "input_digests",
            "expected_provider",
            "non_claims",
        ],
        "stages": [
            stage(
                "official_reference_review",
                "CLEAN_PROVENANCE_ONLY_NOT_NUMERICAL_VALIDATION",
                "pending_review",
                {"doc": "docs/official-semantics-authority.md", "model_py_sha256": MODEL_PY_SHA, "kernel_py_sha256": KERNEL_PY_SHA},
                "verify local reference snapshot and M1 HF revision against official DeepSeek release material",
                ["does not generate logits", "does not authorize model-math expansion by itself"],
            ),
            stage(
                "raw_embedding_gather",
                "CLEAN_OFFICIAL_CHECKPOINT_RAW_BITS",
                "available",
                {"artifact": "artifacts/m2/dwarfstar-prefill/native-official-embedding.json", "tensor": "embed.weight"},
                "keep as model-data primitive; do not widen to embedding semantics without reference review",
                ["no full-model semantics", "no attention/MoE/HC semantics"],
            ),
            stage(
                "bf16_dense_linear_primitive",
                "CLEAN_INDEPENDENT_ARITHMETIC_CONTRACT",
                "available",
                {"artifact": "artifacts/m2/dwarfstar-prefill/native-official-projection.json", "tensor": "layers.0.ffn.gate.weight"},
                "keep as isolated primitive; define official full-layer dtype behavior separately",
                ["no quantized expert semantics", "no official full-layer correctness"],
            ),
            stage(
                "parallel_embedding_semantics",
                "PLANNED_OFFICIAL_REFERENCE_DERIVED",
                "blocked_until_reference_review",
                {"file": "inference/model.py", "sha256": MODEL_PY_SHA, "class": "ParallelEmbedding.forward"},
                "review tensor-parallel masking/all_reduce behavior and define bounded fixture",
                ["must not use oMLX processor/model output as expected value"],
            ),
            stage(
                "bf16_linear_reference_semantics",
                "PLANNED_OFFICIAL_REFERENCE_DERIVED",
                "blocked_until_reference_review",
                {"file": "inference/model.py", "sha256": MODEL_PY_SHA, "functions": ["linear", "Linear.forward"]},
                "review non-quantized BF16/F.linear path and dtype behavior",
                ["current NumPy primitive is arithmetic evidence, not full official layer semantics"],
            ),
            stage(
                "rmsnorm_semantics",
                "PLANNED_OFFICIAL_REFERENCE_DERIVED",
                "blocked_until_reference_review",
                {"file": "inference/model.py", "sha256": MODEL_PY_SHA, "class": "RMSNorm.forward"},
                "define F32 variance/rsqrt and output cast contract from official reference",
                ["do not compare to oMLX RMSNorm output as oracle"],
            ),
            stage(
                "attention_moe_hc_dspark",
                "PLANNED_COMPLEX_REVIEW_REQUIRED",
                "blocked",
                {"files": [{"path": "inference/model.py", "sha256": MODEL_PY_SHA}, {"path": "inference/kernel.py", "sha256": KERNEL_PY_SHA}]},
                "split into separate review plans before any native implementation",
                ["no cache/logits/state oracle from oMLX", "no broad full-model claim"],
            ),
        ],
        "ok": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
