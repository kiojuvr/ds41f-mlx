"""M9 persistence for the M8 idle DeepseekV41Cache authority.

Artifacts are dormant storage, not a second live cache authority.  The only
supported boundary is an M8 idle state: ``DeepseekV41Cache[40]`` plus exact
``all_tokens`` after ``extract_final_state()`` and before another append begins.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

from ds41f_mlx.prefill_fp8_mlx.handoff import _validate_live_cache_structure
from ds41f_mlx.runtime.omlx_core import DEFAULT_OMLX
from ds41f_mlx.config import DEFAULT_KV_ROOT

SCHEMA = "ds41f.m9.deepseek-v41-kv-artifact.v1"


class M9PersistenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class M9ArtifactInfo:
    path: Path
    frontier: int
    token_count: int
    tensor_count: int
    tensor_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {"path": str(self.path), "frontier": self.frontier, "token_count": self.token_count, "tensor_count": self.tensor_count, "tensor_sha256": self.tensor_sha256}


def _git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_fingerprint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    out: dict[str, Any] = {"path": str(checkpoint.resolve() if checkpoint.exists() else checkpoint)}
    if index.exists():
        st = index.stat()
        out.update({"index_size": st.st_size, "index_mtime_ns": st.st_mtime_ns, "index_sha256": _sha256(index)})
    return out


def _cache_class(omlx_path: Path = DEFAULT_OMLX):
    root = str(omlx_path)
    if root not in sys.path:
        sys.path.insert(0, root)
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
    return DeepseekV41Cache


def _tensor_name(layer: int, slot: int) -> str:
    return f"layer_{layer:02d}.slot_{slot}"


def cache_inventory(cache: list[Any]) -> list[dict[str, Any]]:
    inv = []
    for layer, item in enumerate(cache):
        slots = []
        for slot, value in enumerate(getattr(item, "cache", [])):
            if value is None:
                slots.append({"slot": slot, "present": False})
            else:
                slots.append({"slot": slot, "present": True, "shape": [int(x) for x in value.shape], "dtype": str(value.dtype).rsplit(".", 1)[-1], "tensor": _tensor_name(layer, slot)})
        inv.append({"layer": layer, "class_name": type(item).__name__, "compress_ratio": getattr(item, "compress_ratio", None), "offset": int(item.size()), "slots": slots})
    return inv


def save_m8_idle_state(*, artifact_root: Path = DEFAULT_KV_ROOT, model: Any, live_cache: list[Any], all_tokens: Sequence[int], checkpoint: Path, omlx_path: Path = DEFAULT_OMLX, diagnostics: dict[str, Any] | None = None) -> M9ArtifactInfo:
    """Atomically persist an M8 idle cache and token history."""
    import mlx.core as mx

    lm = getattr(model, "language_model", model)
    tokens = [int(t) for t in all_tokens]
    if not tokens:
        raise M9PersistenceError("cannot persist empty token history")
    frontier = len(tokens)
    _validate_live_cache_structure(live_cache, lm._config, frontier)
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    final = artifact_root / f"m9-{int(time.time())}-{uuid.uuid4().hex[:12]}"
    tmp = artifact_root / (final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tensors: dict[str, Any] = {}
    for layer, item in enumerate(live_cache):
        for slot, value in enumerate(item.cache):
            if value is None:
                raise M9PersistenceError(f"layer {layer} slot {slot} is None at M8 idle boundary")
            tensors[_tensor_name(layer, slot)] = value
    tensor_path = tmp / "cache.safetensors"
    mx.save_safetensors(str(tensor_path), tensors, metadata={"schema": SCHEMA})
    mx.synchronize()
    tensor_hash = _sha256(tensor_path)
    manifest = {
        "schema": SCHEMA,
        "created_at": time.time(),
        "frontier": frontier,
        "all_tokens": tokens,
        "layer_count": len(live_cache),
        "slot_count": 7,
        "tensor_file": "cache.safetensors",
        "tensor_sha256": tensor_hash,
        "tensor_count": len(tensors),
        "checkpoint": _checkpoint_fingerprint(Path(checkpoint)),
        "omlx_path": str(Path(omlx_path)),
        "omlx_revision": _git_rev(Path(omlx_path)),
        "python": sys.version,
        "cache_inventory": cache_inventory(live_cache),
        "diagnostics": diagnostics or {},
        "commit_marker": "complete",
    }
    (tmp / "manifest.json.tmp").write_text(json.dumps(manifest, indent=2))
    os.replace(tmp / "manifest.json.tmp", tmp / "manifest.json")
    (tmp / "COMMITTED").write_text(tensor_hash + "\n")
    os.replace(tmp, final)
    return M9ArtifactInfo(final, frontier, frontier, len(tensors), tensor_hash)


def restore_m8_idle_state(*, artifact_path: Path, model: Any, checkpoint: Path, omlx_path: Path = DEFAULT_OMLX) -> tuple[list[Any], list[int], dict[str, Any]]:
    """Validate and restore one live request-local DeepseekV41Cache[40]."""
    import mlx.core as mx

    path = Path(artifact_path)
    manifest_path = path / "manifest.json"
    tensor_path = path / "cache.safetensors"
    if not (manifest_path.exists() and tensor_path.exists() and (path / "COMMITTED").exists()):
        raise M9PersistenceError("artifact is incomplete")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("commit_marker") != "complete":
        raise M9PersistenceError("artifact schema/commit marker mismatch")
    if _sha256(tensor_path) != manifest.get("tensor_sha256"):
        raise M9PersistenceError("artifact tensor sha256 mismatch")
    expected_ckpt = _checkpoint_fingerprint(Path(checkpoint))
    recorded = manifest.get("checkpoint", {})
    if recorded.get("index_sha256") and expected_ckpt.get("index_sha256") != recorded.get("index_sha256"):
        raise M9PersistenceError("checkpoint fingerprint mismatch")
    tokens = [int(t) for t in manifest.get("all_tokens", [])]
    frontier = int(manifest.get("frontier", -1))
    if frontier <= 0 or frontier != len(tokens):
        raise M9PersistenceError("artifact token/frontier mismatch")
    arrays = mx.load(str(tensor_path))
    Cache = _cache_class(Path(omlx_path))
    restored = []
    inv = manifest.get("cache_inventory", [])
    if len(inv) != 40:
        raise M9PersistenceError("artifact does not contain 40 cache layers")
    for layer_meta in inv:
        layer = int(layer_meta["layer"])
        item = Cache(int(layer_meta.get("compress_ratio") or 0))
        slots = []
        for slot_meta in layer_meta["slots"]:
            slot = int(slot_meta["slot"])
            name = slot_meta.get("tensor")
            if not slot_meta.get("present") or name not in arrays:
                raise M9PersistenceError(f"missing layer {layer} slot {slot}")
            value = arrays[name]
            if [int(x) for x in value.shape] != list(slot_meta["shape"]):
                raise M9PersistenceError(f"shape mismatch for {name}")
            if str(value.dtype).rsplit(".", 1)[-1] != slot_meta["dtype"]:
                raise M9PersistenceError(f"dtype mismatch for {name}")
            slots.append(value)
        item.cache = slots
        restored.append(item)
    lm = getattr(model, "language_model", model)
    _validate_live_cache_structure(restored, lm._config, frontier)
    return restored, tokens, manifest
