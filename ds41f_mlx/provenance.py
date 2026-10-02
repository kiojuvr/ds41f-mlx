"""Cheap runtime provenance inspection for ds41f operational qualification."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import importlib.util
import json
import os
import platform
import subprocess
import sys
from typing import Any

from ds41f_mlx.config import RuntimeConfig, load_runtime_config, validate_runtime_config
from ds41f_mlx.release import load_release_manifest

_RELEASE = load_release_manifest()
PINNED_OMLX_REVISION = _RELEASE["dependencies"]["omlx"]["revision"]
PINNED_RECIPE_REVISION = _RELEASE["dependencies"]["deepseek_recipe"]["revision"]

DS41F_RUNTIME_PATHS = ("ds41f_mlx", "native")
DS41F_RUST_BOUNDARY_PATHS = ("Cargo.toml", "Cargo.lock", "rust/ds41f_api")
DS41F_RELEASE_PATHS = ("release", "pyproject.toml", "Cargo.toml", "Cargo.lock", "rust/ds41f_api/Cargo.toml")
DS41F_RUNTIME_EXCLUDE = {"ds41f_mlx/provenance.py", "ds41f_mlx/qualify.py", "ds41f_mlx/acceptance.py", "ds41f_mlx/release.py", "ds41f_mlx/release_acceptance.py", "ds41f_mlx/ops.py"}
DS41F_RUNTIME_EXCLUDE_PREFIXES = ("ds41f_mlx/web",)
DS41F_QUALIFICATION_PATHS = ("tools", "tests", "ds41f_mlx/provenance.py", "ds41f_mlx/qualify.py", "ds41f_mlx/acceptance.py")
DS41F_NONRUNTIME_PREFIXES = ("artifacts/", "docs/")

# Local oMLX changes present on the target machine during M16/M17. These are
# prompt/processor image-token handling changes. ds41f selected serving uses
# deepseek-recipe rendering plus oMLX model load/GenerationBatch decode, not the
# oMLX API processor. They are recorded exactly so the checkout is identified
# without treating any dirty bit as unexplained runtime drift.
APPROVED_EXTERNAL_PATCHES = {
    "omlx": {
        "omlx/patches/deepseek_v41/encoding.py": {
            "kind": "modified",
            "diff_sha256": "2f4bbfccc21fbd1843cc316caf4d19e3d424dd05b436ba72804e89db655a211f",
            "content_sha256": "889efbdc8abe8aa4d18538596a427ff874b01bf55887ac6981d8c4e8610301e2",
            "production_reachable": False,
            "reason": "oMLX processor literal image-token escape; ds41f production path uses deepseek-recipe text rendering and GenerationBatch decode, not oMLX Processor chat encoding.",
        },
        "omlx/patches/deepseek_v41/processing.py": {
            "kind": "modified",
            "diff_sha256": "ca266be8be962673e3e340caabf941fa1cf4a0dd510f22a7298fab3469e62564",
            "content_sha256": "b920a47364a2af2d628fcd4fa0a62b104b6634a90a0719b67b954031ac4154e7",
            "production_reachable": False,
            "reason": "oMLX Processor wiring for literal image-token escape; not used by ds41f production recipe serving path.",
        },
        "tests/test_deepseek_v41_literal_image_token.py": {
            "kind": "untracked",
            "content_sha256": "bec1010d13827be63168d04071534c4f306d7a58a5688c4e616e2f69f2617f72",
            "production_reachable": False,
            "reason": "oMLX local test file only; not imported by ds41f runtime.",
        },
    },
    "deepseek-recipe": {},
}


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_dirty(path: Path) -> bool | None:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def file_sha256(path: Path, *, max_bytes: int | None = None) -> str | None:
    try:
        h = sha256()
        with path.open("rb") as f:
            remaining = max_bytes
            while True:
                size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
                if size <= 0:
                    break
                b = f.read(size)
                if not b:
                    break
                h.update(b)
                if remaining is not None:
                    remaining -= len(b)
        return h.hexdigest()
    except Exception:
        return None


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists() or not path.is_dir():
        return out
    for name in ("model.safetensors.index.json", "config.json", "tokenizer.json"):
        p = path / name
        if p.exists():
            out[name] = {"size": p.stat().st_size, "sha256": file_sha256(p)}
    # Hugging Face snapshots often have refs; include cheap revision hint when present.
    for name in ("revision", ".git/HEAD"):
        p = path / name
        if p.exists() and p.is_file():
            try:
                out[name.replace("/", "_")] = p.read_text(errors="ignore").strip()[:200]
            except Exception:
                pass
    return out


def tree_digest(root: Path, include: tuple[str, ...]) -> dict[str, Any]:
    files: list[Path] = []
    for item in include:
        p = root / item
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
        else:
            for child in p.rglob("*"):
                rel_parts = child.relative_to(root).parts
                if not child.is_file():
                    continue
                if "__pycache__" in rel_parts or child.name.endswith((".pyc", ".o")):
                    continue
                rel = child.relative_to(root).as_posix()
                if rel in DS41F_RUNTIME_EXCLUDE or rel.startswith(DS41F_RUNTIME_EXCLUDE_PREFIXES):
                    continue
                if rel_parts[0] == "native" and len(rel_parts) > 1 and rel_parts[1].startswith("build"):
                    continue
                files.append(child)
    h = sha256()
    entries = []
    for path in sorted(files):
        rel = path.relative_to(root).as_posix()
        digest = file_sha256(path) or ""
        size = path.stat().st_size
        h.update(rel.encode()); h.update(b"\0"); h.update(digest.encode()); h.update(b"\0")
        entries.append({"path": rel, "sha256": digest, "size": size})
    return {"sha256": h.hexdigest(), "file_count": len(entries), "files": entries}


def ds41f_dirty_classification(root: Path) -> dict[str, Any]:
    try:
        out = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {"available": False}
    entries = []
    runtime_affecting = False
    qualification_affecting = False
    for line in out.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        category = "other_nonruntime"
        if path in DS41F_RUNTIME_EXCLUDE:
            category = "release_or_qualification_tooling"; qualification_affecting = True
        elif path.startswith(DS41F_RUNTIME_EXCLUDE_PREFIXES):
            category = "client_nonruntime"
        elif path in {"pyproject.toml", "Cargo.toml", "Cargo.lock"} or path.startswith("release/") or path.startswith("rust/"):
            category = "release_packaging_or_rust_boundary"
        elif path.startswith("ds41f_mlx/") or path.startswith("native/"):
            category = "runtime_source"; runtime_affecting = True
        elif path.startswith("tools/") or path.startswith("tests/"):
            category = "qualification_tooling"; qualification_affecting = True
        elif path.startswith(DS41F_NONRUNTIME_PREFIXES):
            category = "generated_or_documentation"
        entries.append({"status": status.strip(), "path": path, "category": category})
    return {"available": True, "runtime_affecting_dirty": runtime_affecting, "qualification_affecting_dirty": qualification_affecting, "entries": entries}


def ds41f_git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {
        "path": str(root),
        "commit": git_rev(root),
        "dirty": git_dirty(root),
        "dirty_classification": ds41f_dirty_classification(root),
        "runtime_source_identity": tree_digest(root, DS41F_RUNTIME_PATHS),
        "rust_boundary_identity": tree_digest(root, DS41F_RUST_BOUNDARY_PATHS),
        "release_packaging_identity": tree_digest(root, DS41F_RELEASE_PATHS),
        "qualification_tooling_identity": tree_digest(root, DS41F_QUALIFICATION_PATHS),
        "identity_model": "runtime_source_identity excludes generated artifacts/docs, release packaging metadata, Rust client boundary, and the separate ds41f_mlx.web local client. Runtime-affecting Python/native changes invalidate inherited model evidence; Rust/release packaging changes require boundary/acceptance qualification unless they alter a reachable model runtime path.",
    }


def import_check(module: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return {"module": module, "status": "FAIL", "origin": None, "error": "module spec not found"}
    try:
        imported = __import__(module)
        return {"module": module, "status": "PASS", "origin": getattr(imported, "__file__", spec.origin)}
    except Exception as exc:
        return {"module": module, "status": "FAIL", "origin": spec.origin, "error": repr(exc)}


def external_worktree_identity(name: str, path: Path) -> dict[str, Any]:
    approved = APPROVED_EXTERNAL_PATCHES.get(name, {})
    try:
        raw = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc), "entries": []}
    entries = []
    statuses = []
    h = sha256()
    for line in raw.splitlines():
        if not line:
            continue
        status_code = line[:2]
        rel = line[3:]
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        full = path / rel
        entry: dict[str, Any] = {"path": rel, "git_status": status_code.strip()}
        if status_code.startswith("??"):
            entry["kind"] = "untracked"
            entry["content_sha256"] = file_sha256(full) if full.is_file() else None
        else:
            diff = subprocess.check_output(["git", "-C", str(path), "diff", "--", rel])
            entry["kind"] = "modified"
            entry["diff_sha256"] = sha256(diff).hexdigest()
            entry["content_sha256"] = file_sha256(full) if full.exists() and full.is_file() else None
        spec = approved.get(rel)
        if spec is None:
            entry["approval"] = "UNKNOWN"
            entry["status"] = "WARNING"
        else:
            mismatches = []
            for key in ("kind", "diff_sha256", "content_sha256"):
                if key in spec and entry.get(key) != spec[key]:
                    mismatches.append(key)
            entry["approval"] = "APPROVED" if not mismatches else "MISMATCH"
            entry["production_reachable"] = bool(spec.get("production_reachable"))
            entry["reason"] = spec.get("reason")
            entry["status"] = "PASS" if not mismatches else "WARNING"
            if mismatches:
                entry["mismatches"] = mismatches
        statuses.append(entry["status"])
        h.update(rel.encode()); h.update(b"\0"); h.update((entry.get("content_sha256") or entry.get("diff_sha256") or "").encode()); h.update(b"\0")
        entries.append(entry)
    final = "WARNING" if "WARNING" in statuses else "PASS"
    return {
        "status": final,
        "base_revision": git_rev(path),
        "dirty": bool(entries),
        "dirty_entries": entries,
        "local_identity_sha256": h.hexdigest(),
        "identity_model": "base revision plus exact approved local content/diff digests; unknown or mismatched executable differences warn and require review/requalification.",
    }


def omlx_decode_native_identity(root: Path) -> dict[str, Any]:
    """Identify ignored V4.1 GLM binaries independently of checkout pathname."""
    entries = []
    for path in sorted((root / "omlx/custom_kernels/glm_moe_dsa").glob("*")):
        if path.is_file() and path.suffix in {".so", ".dylib", ".metallib"}:
            entries.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)})
    return {"sha256": sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(), "files": entries}


def package_version_checks() -> list[dict[str, Any]]:
    deps = _RELEASE["dependencies"]
    checks = []
    for package, dep_key in (("mlx", "mlx"), ("mlx-lm", "mlx_lm"), ("deepseek-recipe", "deepseek_recipe"), ("omlx", "omlx")):
        expected = deps[dep_key].get("version")
        actual = package_version(package)
        checks.append({"component": package, "check": "package_version", "expected": expected, "actual": actual, "status": "PASS" if actual == expected else "FAIL"})
    return checks


def release_manifest_checks() -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    checks: list[dict[str, Any]] = []
    try:
        import tomllib
        py = tomllib.loads((root / "pyproject.toml").read_text())
        checks.append({"component": "pyproject", "check": "project.version", "expected": _RELEASE["release"]["version"], "actual": py["project"]["version"], "status": "PASS" if py["project"]["version"] == _RELEASE["release"]["version"] else "FAIL"})
        checks.append({"component": "pyproject", "check": "tool.ds41f.omlx_upstream_baseline", "expected": PINNED_OMLX_REVISION, "actual": py["tool"]["ds41f"].get("omlx_upstream_baseline"), "status": "PASS" if py["tool"]["ds41f"].get("omlx_upstream_baseline") == PINNED_OMLX_REVISION else "FAIL"})
    except Exception as exc:
        checks.append({"component": "pyproject", "check": "parse", "status": "FAIL", "error": repr(exc)})
    cargo = root / "rust" / "ds41f_api" / "Cargo.toml"
    text = cargo.read_text() if cargo.exists() else ""
    expected = _RELEASE["rust"]["version"]
    actual = None
    for line in text.splitlines():
        if line.startswith("version = "):
            actual = line.split("=", 1)[1].strip().strip('"')
            break
    checks.append({"component": "ds41f_api", "check": "crate.version", "expected": expected, "actual": actual, "status": "PASS" if actual == expected else "FAIL"})
    return checks


def inspect_runtime(config: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = config or load_runtime_config()
    cfg.apply_import_paths()
    validation = validate_runtime_config(cfg)
    omlx_rev = git_rev(cfg.omlx_path)
    recipe_rev = git_rev(cfg.recipe_path)
    omlx_identity = external_worktree_identity("omlx", cfg.omlx_path)
    recipe_identity = external_worktree_identity("deepseek-recipe", cfg.recipe_path)
    revision_checks = [
        {"component": "oMLX", "check": "revision", "expected": PINNED_OMLX_REVISION, "actual": omlx_rev, "status": "PASS" if omlx_rev == PINNED_OMLX_REVISION else "WARNING"},
        {"component": "oMLX", "check": "local_differences", "expected": "clean or approved exact patches", "actual": omlx_identity["status"], "status": omlx_identity["status"]},
        {"component": "deepseek-recipe", "check": "revision", "expected": PINNED_RECIPE_REVISION, "actual": recipe_rev, "status": "PASS" if recipe_rev == PINNED_RECIPE_REVISION else "WARNING"},
        {"component": "deepseek-recipe", "check": "local_differences", "expected": "clean or approved exact patches", "actual": recipe_identity["status"], "status": recipe_identity["status"]},
    ]
    import_checks = [import_check("omlx"), import_check("deepseek_recipe"), import_check("mlx"), import_check("mlx_lm")]
    version_checks = package_version_checks()
    manifest_checks = release_manifest_checks()
    statuses = [x["status"] for x in validation] + [x["status"] for x in revision_checks] + [x["status"] for x in import_checks] + [x["status"] for x in version_checks] + [x["status"] for x in manifest_checks]
    final = "FAIL" if "FAIL" in statuses else ("WARNING" if "WARNING" in statuses else "PASS")
    return {
        "schema": "ds41f.runtime-provenance.v2",
        "status": final,
        "ds41f": ds41f_git(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "mac_ver": platform.mac_ver()},
        "release_manifest": _RELEASE,
        "packages": {"mlx": package_version("mlx"), "mlx-lm": package_version("mlx-lm"), "deepseek-recipe": package_version("deepseek-recipe"), "omlx": package_version("omlx")},
        "import_checks": import_checks,
        "package_version_checks": version_checks,
        "release_manifest_checks": manifest_checks,
        "config": cfg.to_json(),
        "validation": validation,
        "checkpoint_fingerprint": checkpoint_fingerprint(cfg.checkpoint_path),
        "omlx": {"path": str(cfg.omlx_path), "revision": omlx_rev, **omlx_identity,
                 "decode_native_identity": omlx_decode_native_identity(cfg.omlx_path)},
        "deepseek_recipe": {"path": str(cfg.recipe_path), "revision": recipe_rev, **recipe_identity},
        "revision_checks": revision_checks,
        "production": {"prefill_selector": cfg.production_prefill_selector, "mtp": cfg.mtp, "dspark": cfg.dspark, "speculative_decode": cfg.speculative_decode},
    }


def main() -> int:
    report = inspect_runtime()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
