#!/usr/bin/env python3
"""Record function-level official reference review metadata.

This is source-identity/review metadata only. It parses the candidate official
reference source and records source spans/hashes for functions/classes that may
later be used for official-reference-derived fixtures. It does not execute model
math.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"

TARGETS = {
    "inference/model.py": [
        "ParallelEmbedding",
        "ParallelEmbedding.forward",
        "linear",
        "Linear",
        "Linear.forward",
        "RMSNorm",
        "RMSNorm.forward",
        "precompute_freqs_cis",
        "apply_rotary_emb",
        "Compressor",
        "Indexer",
        "Attention",
        "Gate",
        "Expert",
        "MoE",
        "Block",
        "ParallelHead",
        "ParallelHead.forward",
        "Transformer",
    ],
    "inference/kernel.py": [
        "act_quant",
        "fp4_act_quant",
        "fp8_gemm",
        "fp4_gemm",
        "sparse_attn",
        "hc_split_sinkhorn",
    ],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def node_name_stack(tree: ast.AST) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        out[f"{node.name}.{child.name}"] = child
    return out


def source_segment(lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return ""
    return "\n".join(lines[start - 1 : end]) + "\n"


def signature_for(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = []
    for a in node.args.posonlyargs:
        args.append(a.arg)
    if node.args.posonlyargs:
        args.append("/")
    for a in node.args.args:
        args.append(a.arg)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    elif node.args.kwonlyargs:
        args.append("*")
    for a in node.args.kwonlyargs:
        args.append(a.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"{node.name}({', '.join(args)})"


def file_review(base: Path, rel: str, targets: list[str]) -> dict[str, Any]:
    path = base / rel
    if not path.exists():
        return {"path": rel, "exists": False, "targets": []}
    text = path.read_text()
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    nodes = node_name_stack(tree)
    records = []
    for target in targets:
        node = nodes.get(target)
        if node is None:
            records.append({"name": target, "exists": False})
            continue
        seg = source_segment(lines, node)
        records.append(
            {
                "name": target,
                "exists": True,
                "kind": type(node).__name__,
                "lineno": getattr(node, "lineno", None),
                "end_lineno": getattr(node, "end_lineno", None),
                "source_sha256": sha256_bytes(seg.encode()),
                "signature": signature_for(node),
                "docstring_present": bool(ast.get_docstring(node)),
            }
        )
    return {
        "path": rel,
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "targets": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--out", default="artifacts/official-reference-review.json")
    args = ap.parse_args()

    base = Path(args.checkpoint)
    files = [file_review(base, rel, targets) for rel, targets in TARGETS.items()]
    missing = [t for f in files for t in f.get("targets", []) if not t.get("exists")]
    record = {
        "schema": "ds41f.official-reference-review.v1",
        "purpose": "function-level source identity for candidate official DeepSeek reference; no model math executed",
        "classification": "CLEAN_PROVENANCE_REVIEW_METADATA_NOT_NUMERICAL_VALIDATION",
        "checkpoint": str(base),
        "files": files,
        "review_status": "source_identity_recorded_manual_semantics_review_required",
        "next_allowed_step": "manual review of selected source spans and dtype/arithmetic contracts before generating official-reference-derived fixtures",
        "missing_targets": missing,
        "ok": not missing,
        "non_claims": [
            "does not execute model math",
            "does not validate logits/cache/state",
            "does not make oMLX authoritative",
            "does not authorize native model-math expansion by itself",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
