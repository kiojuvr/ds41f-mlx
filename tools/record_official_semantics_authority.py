#!/usr/bin/env python3
"""Record the candidate official DeepSeek-V4.1-Flash semantics authority.

This does not execute model math.  It pins the local official checkpoint snapshot's
reference implementation files so future semantic-oracle work can be reviewed
against explicit source identities instead of oMLX behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"
HF_REVISION_FROM_M1 = "dba1be0a40aa45a94ad051997016db3960a90277"

SEMANTICS_FILES = [
    "README.md",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "inference/README.md",
    "inference/config.json",
    "inference/model.py",
    "inference/kernel.py",
    "inference/engram.py",
    "inference/generate.py",
    "inference/image_processor.py",
    "inference/vision.py",
    "inference/requirements.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(base: Path, rel: str) -> dict[str, Any]:
    path = base / rel
    if not path.exists():
        return {"path": rel, "exists": False}
    return {
        "path": rel,
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--out", default="artifacts/official-semantics-authority.json")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    files = [file_record(ckpt, rel) for rel in SEMANTICS_FILES]
    record = {
        "schema": "ds41f.official-semantics-authority.v1",
        "purpose": "pin candidate official DeepSeek-V4.1-Flash reference implementation identity before future semantic oracle work",
        "classification": "CLEAN_PROVENANCE_ONLY_NOT_NUMERICAL_VALIDATION",
        "checkpoint": str(ckpt),
        "hf_revision_from_m1_checkpoint_provenance": HF_REVISION_FROM_M1,
        "authority_policy": {
            "model_data": "official checkpoint raw tensors/config/tokenizer are authoritative",
            "model_semantics": "official DeepSeek reference implementation / published architecture must be used; oMLX is not a semantic oracle",
            "execution_architecture": "DwarfStar is architecture/scheduling authority only",
        },
        "files": files,
        "required_files_present": all(f.get("exists") for f in files if f["path"].startswith("inference/") or f["path"] in {"README.md", "config.json"}),
        "model_py_sha256": next((f.get("sha256") for f in files if f["path"] == "inference/model.py"), None),
        "kernel_py_sha256": next((f.get("sha256") for f in files if f["path"] == "inference/kernel.py"), None),
        "ok": True,
        "non_claims": [
            "does not execute model math",
            "does not validate logits/cache/state",
            "does not prove local files match an unverified remote revision beyond recorded M1 HF revision metadata",
            "does not authorize further native model math by itself",
        ],
    }
    record["ok"] = bool(record["required_files_present"] and record["model_py_sha256"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
