"""Build a relocatable local ds41f release bundle."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any

from ds41f_mlx.provenance import ds41f_git, file_sha256
from ds41f_mlx.release import load_release_manifest

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", "target", "build", "build-mlx"}
EXCLUDE_PATTERNS = ["*.pyc", ".DS_Store"]


def copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in EXCLUDE_PATTERNS):
            continue
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def file_manifest(root: Path) -> dict[str, Any]:
    entries = []
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix()
        digest = file_sha256(p) or ""
        entries.append({"path": rel, "sha256": digest, "size": p.stat().st_size})
        h.update(rel.encode()); h.update(b"\0"); h.update(digest.encode()); h.update(b"\0")
    return {"sha256": h.hexdigest(), "file_count": len(entries), "files": entries}


def build_bundle(output_dir: Path, *, skip_rust_build: bool = False) -> dict[str, Any]:
    manifest = load_release_manifest()
    version = manifest["release"]["version"]
    name = f"ds41f-mlx-{version}"
    dist_root = output_dir / name
    if dist_root.exists():
        shutil.rmtree(dist_root)
    dist_root.mkdir(parents=True)

    # Verify manifest/package consistency before copying.
    import tomllib
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if py["project"]["version"] != version:
        raise RuntimeError("pyproject version does not match release manifest")
    cargo_text = (ROOT / "rust/ds41f_api/Cargo.toml").read_text()
    if f'version = "{manifest["rust"]["version"]}"' not in cargo_text:
        raise RuntimeError("ds41f_api Cargo version does not match release manifest")

    copy_tree(ROOT / "ds41f_mlx", dist_root / "ds41f_mlx")
    copy_tree(ROOT / "release", dist_root / "release")
    copy_tree(ROOT / "rust" / "ds41f_api", dist_root / "rust" / "ds41f_api")
    shutil.copy2(ROOT / "pyproject.toml", dist_root / "pyproject.toml")
    shutil.copy2(ROOT / "Cargo.toml", dist_root / "Cargo.toml")
    shutil.copy2(ROOT / "Cargo.lock", dist_root / "Cargo.lock")
    shutil.copy2(ROOT / "README.md", dist_root / "README.md")
    for doc in ("docs/operations.md", "docs/m22-release-packaging.md", "docs/m23-relocatable-release.md"):
        p = ROOT / doc
        if p.exists():
            out = dist_root / doc; out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, out)

    (dist_root / "bin").mkdir(exist_ok=True)
    if not skip_rust_build:
        cargo_target = output_dir / "cargo-target"
        env = os.environ.copy(); env["CARGO_TARGET_DIR"] = str(cargo_target)
        run(["cargo", "build", "--release", "--bin", "m21_real_acceptance"], env=env)
        src_bin = cargo_target / "release" / "m21_real_acceptance"
        if not src_bin.exists():
            raise RuntimeError("expected Rust acceptance binary was not built")
        shutil.copy2(src_bin, dist_root / "bin" / "m21_real_acceptance")

    write_executable(dist_root / "bin" / "ds41f", "#!/usr/bin/env bash\nset -euo pipefail\nDIR=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\nexport PYTHONPATH=\"$DIR${PYTHONPATH:+:$PYTHONPATH}\"\nexec \"${DS41F_PYTHON:-python3}\" -m ds41f_mlx.ops \"$@\"\n")
    write_executable(dist_root / "bin" / "ds41f-accept", "#!/usr/bin/env bash\nset -euo pipefail\nDIR=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\nexport PYTHONPATH=\"$DIR${PYTHONPATH:+:$PYTHONPATH}\"\nexport DS41F_ACCEPTANCE_BIN=\"$DIR/bin/m21_real_acceptance\"\nexec \"${DS41F_PYTHON:-python3}\" -m ds41f_mlx.release_acceptance --installed \"$@\"\n")
    (dist_root / "config").mkdir(exist_ok=True)
    (dist_root / "config" / "ds41f.env.example").write_text("# Copy/edit or source this file before using the bundle.\nexport DS41F_PYTHON=$HOME/.venvs/omlx-0.7.0.release/bin/python\nexport DS41F_CHECKPOINT=/path/to/DeepSeek-V4.1-Flash\nexport DS41F_OMLX_PATH=$HOME/omlx-0.7.0.release\nexport DS41F_RECIPE_PATH=/path/to/deepseek-recipe\nexport DS41F_KV_ROOT=/path/to/ds41f-kv\nexport DS41F_HOST=127.0.0.1\nexport DS41F_PORT=8000\n")

    identity = file_manifest(dist_root)
    record = {
        "schema": "ds41f.release-bundle-record.v1",
        "created_at": time.time(),
        "release_manifest": manifest,
        "source": ds41f_git(),
        "bundle_identity": identity,
        "external_assets": "not bundled; configured with DS41F_* environment variables",
    }
    (dist_root / "release" / "bundle-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    # Recompute after writing bundle record.
    record["bundle_identity"] = file_manifest(dist_root)
    (dist_root / "release" / "bundle-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    tar_path = output_dir / f"{name}.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(dist_root, arcname=name)
    result = {"schema": "ds41f.release-build-result.v1", "status": "PASS", "bundle_dir": str(dist_root), "archive": str(tar_path), "bundle_record": str(dist_root / "release" / "bundle-record.json"), "bundle_identity": record["bundle_identity"]}
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build relocatable ds41f release bundle")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    ap.add_argument("--skip-rust-build", action="store_true", help="copy sources only; for tests")
    args = ap.parse_args(argv)
    result = build_bundle(args.output_dir, skip_rust_build=args.skip_rust_build)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
