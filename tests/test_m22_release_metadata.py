from __future__ import annotations

import tomllib
from pathlib import Path

from ds41f_mlx.provenance import inspect_runtime, PINNED_OMLX_REVISION
from ds41f_mlx.release import load_release_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_release_manifest_versions_match_package_metadata():
    manifest = load_release_manifest()
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert py["project"]["version"] == manifest["release"]["version"]
    assert py["tool"]["ds41f"]["omlx_upstream_baseline"] == manifest["dependencies"]["omlx"]["revision"]
    cargo = (ROOT / "rust" / "ds41f_api" / "Cargo.toml").read_text()
    assert f'version = "{manifest["rust"]["version"]}"' in cargo


def test_active_omlx_pin_is_m20_release_not_dev2():
    assert PINNED_OMLX_REVISION == "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"
    assert PINNED_OMLX_REVISION != "b390b31e0c6831225fed0f24d278eb1db7fcb68b"


def test_provenance_reports_release_manifest_identity_without_model_load():
    report = inspect_runtime()
    assert report["schema"] == "ds41f.runtime-provenance.v2"
    assert report["release_manifest"]["release"]["version"] == load_release_manifest()["release"]["version"]
    checks = {(c["component"], c["check"]): c for c in report["release_manifest_checks"]}
    assert checks[("pyproject", "project.version")]["status"] == "PASS"
    assert checks[("ds41f_api", "crate.version")]["status"] == "PASS"
