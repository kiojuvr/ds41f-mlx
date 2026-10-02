"""Release identity helpers for the current ds41f operational release."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST_PATH = ROOT / "release" / "ds41f-release.json"


def load_release_manifest(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or RELEASE_MANIFEST_PATH).read_text())


def release_version() -> str:
    return str(load_release_manifest()["release"]["version"])


def dependency_pin(component: str, key: str) -> str | None:
    value = load_release_manifest().get("dependencies", {}).get(component, {}).get(key)
    return None if value is None else str(value)
