#!/usr/bin/env python3
"""Static self-containment checks for canonical ds41f-mlx HEAD.

No model execution; this verifies documentation, provenance paths, and active source/build
references after canonical documentation consolidation.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOCS = [
    Path("README.md"),
    Path("docs/README.md"),
    Path("docs/architecture.md"),
    Path("docs/correctness.md"),
    Path("docs/session-state.md"),
    Path("docs/attention.md"),
    Path("docs/moe-hc.md"),
    Path("docs/engram.md"),
    Path("docs/generation.md"),
    Path("docs/api.md"),
    Path("docs/performance.md"),
    Path("docs/provenance.md"),
    Path("docs/qualification.md"),
]
OLD_REPO_LIVE_TOKENS = [
    "/Volumes/SDXC-512/deepseek-v41-flash-mlx",
    "../deepseek-v41-flash-mlx",
    "inspect the old repo",
    "look at the old repo",
    "live qualification oracle",
]
CHRONOLOGY_TOKENS = [
    "M0", "M0.5", "M1", "M2", "Boundary7", "Boundary8", "Boundary9",
    "Boundary10", "Boundary11", "Boundary12", "Boundary13", "next boundary", "repair phase",
]
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def read(path: Path) -> str:
    return (ROOT / path).read_text(errors="ignore")


def check_links(path: Path, text: str) -> None:
    base = (ROOT / path).parent
    for target in LINK_RE.findall(text):
        if "://" in target or target.startswith("#") or target.startswith("mailto:"):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = (base / target).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            fail(f"link escapes repository: {path} -> {target}")
        if not resolved.exists():
            fail(f"broken markdown link: {path} -> {target}")


def main() -> int:
    for doc in CANONICAL_DOCS:
        if not (ROOT / doc).exists():
            fail(f"missing canonical doc: {doc}")

    old_live_count = 0
    chronology_count = 0
    for doc in CANONICAL_DOCS:
        text = read(doc)
        check_links(doc, text)
        if "](docs/archive/" in text or "](archive/" in text:
            fail(f"canonical doc links to archive for normative definitions: {doc}")
        for token in OLD_REPO_LIVE_TOKENS:
            if token in text:
                old_live_count += 1
                fail(f"live old-repo dependency wording in {doc}: {token}")
        for token in CHRONOLOGY_TOKENS:
            if token in text:
                chronology_count += 1
                fail(f"chronological development token in canonical doc {doc}: {token}")

    provenance = read(Path("docs/provenance.md"))
    if "deepseek-v41-flash-mlx@1b7d0a2c7d33602437dffd44e26a33f39f189661" not in provenance:
        fail("provenance must record historical native source import identity")
    for phrase in ["runtime source dependency: none", "build dependency: none", "test dependency: none"]:
        if phrase not in provenance:
            fail(f"provenance missing dependency statement: {phrase}")

    native_cmake = read(Path("native/CMakeLists.txt"))
    for forbidden in ["context-ladder", "wide-attention", "packed-moe-probe", "benchmark"]:
        if forbidden in native_cmake:
            fail(f"benchmark/diagnostic target leaked into native CMake: {forbidden}")

    manifest_path = ROOT / "artifacts/provenance/legacy-import.json"
    manifest = json.loads(manifest_path.read_text())
    for imp in manifest.get("imports", []):
        dest = imp.get("destination_path")
        mode = imp.get("import_mode")
        if not dest or mode == "not_imported":
            continue
        if not (ROOT / dest).exists():
            fail(f"imported provenance destination missing: {dest}")

    recon_path = ROOT / "artifacts/legacy-evidence-boundary-reconciliation.json"
    if recon_path.exists():
        recon = json.loads(recon_path.read_text())
        for entry in recon.get("local_legacy_import_resolution", {}).get("entries", []):
            local = entry.get("local_path")
            if local and not (ROOT / local).exists():
                fail(f"legacy proof graph local path missing: {local}")

    class_path = ROOT / "docs/doc-classification.json"
    if not class_path.exists():
        fail("missing docs/doc-classification.json")
    cmap = json.loads(class_path.read_text())
    classified = {entry.get("path") for entry in cmap.get("entries", [])}
    for md in ROOT.joinpath("docs").rglob("*.md"):
        rel = str(md.relative_to(ROOT))
        if rel not in classified:
            fail(f"documentation file missing from classification map: {rel}")

    print(
        "PASS repository self-containment: canonical old-repo live dependencies=0; "
        "canonical chronology dependencies=0; links/provenance/archive policy verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
