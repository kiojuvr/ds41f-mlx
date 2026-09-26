#!/usr/bin/env python3
"""Check Phase 1 legacy evidence import integrity.

This checker is provenance-only. It does not load checkpoints, execute model math,
or require the legacy repository to be mounted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/provenance/legacy-import.json"
RECONCILIATION = ROOT / "artifacts/legacy-evidence-boundary-reconciliation.json"

REJECTED_NAME_FRAGMENTS = [
    "padded-rank3",
    "rank-3",
    "dwarfstar-style",
    "direct-packed",
    "one-dispatch-mma",
    "rectangular",
    "wide-row-serial",
    "wide_chunk",
    "wide-attention",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def main() -> int:
    require(MANIFEST.exists(), f"missing manifest {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("schema") == "ds41f.legacy_import_manifest.v1", "unexpected manifest schema")
    require(isinstance(manifest.get("imports"), list), "imports must be a list")
    require(isinstance(manifest.get("excluded"), list), "excluded must be a list")

    seen = set()
    for imp in manifest["imports"]:
        for key in [
            "legacy_path",
            "content_sha256",
            "legacy_commit",
            "asset_type",
            "evidence_class",
            "claim_supported",
            "destination_path",
            "import_mode",
            "transformed",
            "destination_sha256",
        ]:
            require(key in imp, f"import missing {key}: {imp}")
        require(imp["import_mode"] in {"byte_copy", "manual_extraction", "normalized_copy", "future_adapter", "not_imported"},
                f"unexpected import_mode for {imp['destination_path']}")
        if imp["import_mode"] == "byte_copy":
            require(imp["transformed"] is False, f"byte_copy must not be transformed: {imp['destination_path']}")
        dest_rel = Path(imp["destination_path"])
        require(not dest_rel.is_absolute(), f"destination must be relative: {dest_rel}")
        dest = ROOT / dest_rel
        require(inside_repo(dest), f"destination escapes repo: {dest_rel}")
        require(dest.exists(), f"missing imported file: {dest_rel}")
        digest = sha256(dest)
        require(digest == imp["destination_sha256"], f"destination sha mismatch: {dest_rel}")
        if imp["import_mode"] == "byte_copy":
            require(digest == imp["content_sha256"], f"byte-copy sha mismatch vs source content: {dest_rel}")
        require(str(dest_rel) not in seen, f"duplicate destination: {dest_rel}")
        seen.add(str(dest_rel))
        lowered = str(dest_rel).lower()
        for frag in REJECTED_NAME_FRAGMENTS:
            if frag in lowered:
                require(imp.get("production_reachable") is False,
                        f"rejected/diagnostic candidate is production-reachable: {dest_rel}")
                require(imp.get("classification") in {"DIAGNOSTIC_ONLY", "ARCHIVE_ONLY", "REJECT"},
                        f"rejected/diagnostic candidate lacks diagnostic classification: {dest_rel}")

    excluded_text = "\n".join(json.dumps(x, sort_keys=True).lower() for x in manifest["excluded"])
    for required in ["omlx", "dwarfstar", "rectangular", "wide", "direct packed"]:
        require(required in excluded_text, f"excluded list missing expected rejected/archive component: {required}")

    require(RECONCILIATION.exists(), f"missing reconciliation proof graph {RECONCILIATION}")
    recon = json.loads(RECONCILIATION.read_text())
    local = recon.get("local_legacy_import_resolution")
    require(isinstance(local, dict), "reconciliation missing local_legacy_import_resolution")
    require(local.get("manifest") == "artifacts/provenance/legacy-import.json", "reconciliation points at wrong manifest")
    for entry in local.get("entries", []):
        lp = entry.get("local_path")
        require(lp, "local resolution entry missing local_path")
        dest = ROOT / lp
        require(inside_repo(dest), f"local proof path escapes repo: {lp}")
        require(dest.exists(), f"local proof path missing: {lp}")
        if "sha256" in entry:
            require(sha256(dest) == entry["sha256"], f"local proof path sha mismatch: {lp}")

    print(f"PASS legacy import integrity: {len(manifest['imports'])} imports verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
