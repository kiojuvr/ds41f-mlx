"""Historical bundle packaging must not silently lose relocated reports."""
from pathlib import Path
import tarfile

from ds41f_mlx import build_release


def test_bundle_keeps_archived_packaging_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ds41f_git", lambda: {"test": True})
    result = build_release.build_bundle(tmp_path, skip_rust_build=True)
    bundle = Path(result["bundle_dir"])
    paths = (
        "docs/operations.md",
        "docs/archive/development/m22-release-packaging.md",
        "docs/archive/development/m23-relocatable-release.md",
    )
    with tarfile.open(result["archive"]) as archive:
        names = set(archive.getnames())
    for relative in paths:
        assert (bundle / relative).read_bytes() == (build_release.ROOT / relative).read_bytes()
        assert f"{bundle.name}/{relative}" in names
    assert not (bundle / "docs/m22-release-packaging.md").exists()
    assert not (bundle / "docs/m23-relocatable-release.md").exists()
