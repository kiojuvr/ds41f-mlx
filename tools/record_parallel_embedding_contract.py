#!/usr/bin/env python3
"""Record the ParallelEmbedding official semantics contract metadata.

This records source-span identity and operation contracts only. It does not run
PyTorch/oMLX/native model math and does not generate expected tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_PY = "inference/model.py"
MODEL_PY_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
PARALLEL_EMBEDDING_SHA = "1b8428b4e639bed65674a7e6e3a0387181dc31d110b5c45542d111f54e0b81f4"
FORWARD_SHA = "fc448dd2ac5f8a483c1ed42264e8ac1e0ba12e3bac3a073c12aae28db406f4bb"
CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/parallel-embedding-semantics-contract.json")
    args = ap.parse_args()

    record: dict[str, Any] = {
        "schema": "ds41f.parallel-embedding-semantics-contract.v1",
        "purpose": "manual operation contract for official DeepSeek ParallelEmbedding semantics; no numerical fixture generated",
        "classification": "OFFICIAL_REFERENCE_SEMANTICS_CONTRACT_METADATA_NOT_NUMERICAL_VALIDATION",
        "not_omlx_derived": True,
        "checkpoint": CHECKPOINT,
        "official_reference": {
            "file": MODEL_PY,
            "file_sha256": MODEL_PY_SHA,
            "targets": [
                {
                    "name": "ParallelEmbedding",
                    "lineno": 152,
                    "end_lineno": 178,
                    "source_sha256": PARALLEL_EMBEDDING_SHA,
                },
                {
                    "name": "ParallelEmbedding.forward",
                    "lineno": 168,
                    "end_lineno": 178,
                    "source_sha256": FORWARD_SHA,
                },
            ],
        },
        "operation_contracts": {
            "single_rank": {
                "preconditions": ["world_size == 1", "rank == 0", "all input ids are valid rows in weight"],
                "operation": "output = weight[input_ids]",
                "output_shape": "input_ids.shape + [dim]",
                "dtype_contract": "output dtype/bits are exactly selected embedding weight dtype/bits",
                "expected_provider": "direct raw checkpoint tensor gather; no oMLX execution",
                "fixture_classification": "official_checkpoint_raw_bits",
            },
            "multi_rank": {
                "preconditions": ["vocab_size % world_size == 0", "one rank-local weight slice per rank"],
                "operation": "rank-local masked embedding gather followed by sum all_reduce over ranks",
                "mask": "(ids < rank * part_vocab_size) | (ids >= (rank + 1) * part_vocab_size)",
                "local_ids": "ids - rank * part_vocab_size; masked ids replaced with 0 before lookup",
                "local_output": "weight_rank[local_ids]; masked output rows zeroed after lookup",
                "global_output": "sum(local_output over ranks)",
                "expected_provider": "independent sharded gather/all_reduce implementation over official checkpoint slices; no oMLX execution",
                "fixture_classification": "official_reference_derived",
            },
        },
        "existing_clean_artifacts": [
            {
                "path": "artifacts/m2/dwarfstar-prefill/native-official-embedding.json",
                "classification": "CLEAN_OFFICIAL_CHECKPOINT_RAW_BITS",
                "coverage": "single-rank bounded raw BF16 gather subset",
            }
        ],
        "future_fixture_required_metadata": [
            "classification",
            "not_omlx_derived",
            "source_tensor_name_shard_dtype_shape_digest",
            "ParallelEmbedding.forward source span identity",
            "world_size_rank_part_vocab_size",
            "input_ids_digest",
            "expected_provider",
            "non_claims",
        ],
        "non_claims": [
            "does not validate tokenizer behavior",
            "does not validate HC repeat/pre-mask behavior",
            "does not validate attention/Engram/MoE/DSpark/MTP",
            "does not validate full prefill logits/cache/state",
            "does not authorize native model-math expansion by itself",
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
