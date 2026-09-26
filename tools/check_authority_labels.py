#!/usr/bin/env python3
"""Static guard for obvious correctness-authority regressions.

This is intentionally conservative. It scans source/docs for phrases that were
previously used to promote oMLX compatibility evidence into official correctness.
It does not validate model math.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOTS = ["README.md", "docs", "ds41f_mlx", "tools"]

FORBIDDEN = [
    re.compile(r"official\s+oMLX", re.IGNORECASE),
    re.compile(r"oMLX\s+remains\s+authoritative", re.IGNORECASE),
    re.compile(r"oMLX.*official\s+semantics", re.IGNORECASE),
    re.compile(r"official\s+DeepSeek[^\n]*oMLX\s+semantics", re.IGNORECASE),
    re.compile(r"bounded\s+direct/server\s+exactness\s+oracle", re.IGNORECASE),
    re.compile(r"direct/server\s+oracle(?!.*compatibility)", re.IGNORECASE),
    re.compile(r"accepted\s+oMLX[^\n]*(official|qualification|correctness)", re.IGNORECASE),
]

ALLOW_PATHS = {
    "docs/correctness-authority-audit.md",
}


def iter_files(paths: list[str]):
    for item in paths:
        p = Path(item)
        if not p.exists():
            continue
        if p.is_file():
            yield p
        else:
            for child in p.rglob("*"):
                if child.is_file() and child.suffix in {".py", ".md", ".toml"}:
                    yield child


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=DEFAULT_ROOTS)
    args = ap.parse_args()

    failures: list[str] = []
    for path in iter_files(args.paths):
        rel = path.as_posix()
        if rel in ALLOW_PATHS:
            continue
        try:
            text = path.read_text(errors="ignore")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            # Explicit denials/caveats are the desired repaired language.
            if any(marker in lowered for marker in ("not official", "not an official", "no official", "without official", "requires no official", "do not treat")):
                continue
            for pat in FORBIDDEN:
                if pat.search(line):
                    failures.append(f"{rel}:{i}: {line.strip()}")
                    break
    if failures:
        print("authority-label check failed:")
        for f in failures:
            print(f)
        return 1

    source_check = Path(__file__).with_name("check_source_identity_hashes.py")
    if source_check.exists() and Path("artifacts").exists():
        rc = subprocess.run([sys.executable, str(source_check)], check=False).returncode
        if rc != 0:
            return rc
    print("authority-label check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
