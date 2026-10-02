"""Cheap runtime provenance inspection for ds41f operational qualification."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import importlib.util
import json
import platform
import subprocess
import sys
from typing import Any

from ds41f_mlx.config import RuntimeConfig, load_runtime_config, validate_runtime_config

PINNED_OMLX_REVISION = "b390b31e0c6831225fed0f24d278eb1db7fcb68b"
PINNED_RECIPE_REVISION = "8cadfede7063c896b944e7bae05daa3549ae97ea"


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


def ds41f_git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {"path": str(root), "commit": git_rev(root), "dirty": git_dirty(root)}


def import_check(module: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return {"module": module, "status": "FAIL", "origin": None, "error": "module spec not found"}
    try:
        imported = __import__(module)
        return {"module": module, "status": "PASS", "origin": getattr(imported, "__file__", spec.origin)}
    except Exception as exc:
        return {"module": module, "status": "FAIL", "origin": spec.origin, "error": repr(exc)}


def inspect_runtime(config: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = config or load_runtime_config()
    cfg.apply_import_paths()
    validation = validate_runtime_config(cfg)
    omlx_rev = git_rev(cfg.omlx_path)
    recipe_rev = git_rev(cfg.recipe_path)
    omlx_dirty = git_dirty(cfg.omlx_path)
    recipe_dirty = git_dirty(cfg.recipe_path)
    revision_checks = [
        {"component": "oMLX", "check": "revision", "expected": PINNED_OMLX_REVISION, "actual": omlx_rev, "status": "PASS" if omlx_rev == PINNED_OMLX_REVISION else "WARNING"},
        {"component": "oMLX", "check": "dirty", "expected": False, "actual": omlx_dirty, "status": "PASS" if omlx_dirty is False else "WARNING"},
        {"component": "deepseek-recipe", "check": "revision", "expected": PINNED_RECIPE_REVISION, "actual": recipe_rev, "status": "PASS" if recipe_rev == PINNED_RECIPE_REVISION else "WARNING"},
        {"component": "deepseek-recipe", "check": "dirty", "expected": False, "actual": recipe_dirty, "status": "PASS" if recipe_dirty is False else "WARNING"},
    ]
    import_checks = [import_check("omlx"), import_check("deepseek_recipe"), import_check("mlx"), import_check("mlx_lm")]
    statuses = [x["status"] for x in validation] + [x["status"] for x in revision_checks] + [x["status"] for x in import_checks]
    final = "FAIL" if "FAIL" in statuses else ("WARNING" if "WARNING" in statuses else "PASS")
    return {
        "schema": "ds41f.runtime-provenance.v1",
        "status": final,
        "ds41f": ds41f_git(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "mac_ver": platform.mac_ver()},
        "packages": {"mlx": package_version("mlx"), "mlx-lm": package_version("mlx-lm"), "deepseek-recipe": package_version("deepseek-recipe")},
        "import_checks": import_checks,
        "config": cfg.to_json(),
        "validation": validation,
        "checkpoint_fingerprint": checkpoint_fingerprint(cfg.checkpoint_path),
        "omlx": {"path": str(cfg.omlx_path), "revision": omlx_rev, "dirty": omlx_dirty},
        "deepseek_recipe": {"path": str(cfg.recipe_path), "revision": recipe_rev, "dirty": recipe_dirty},
        "revision_checks": revision_checks,
        "production": {"prefill_selector": cfg.production_prefill_selector, "mtp": cfg.mtp, "dspark": cfg.dspark, "speculative_decode": cfg.speculative_decode},
    }


def main() -> int:
    report = inspect_runtime()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
