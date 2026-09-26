"""M2 DeepSeek-V4.1 explicit state-publication skeleton.

This module is historical oMLX-compatibility instrumentation only.  It does not
introduce new math, kernels, checkpoint formats, scheduling, or performance
policy.  The objects here adapt the existing oMLX per-layer ``shared`` mapping
into an explicit producer-frontier publication contract so tiny fixtures can
show that state currently passed implicitly through a dict can be named, hashed,
and compared at layer boundaries.  Those fixtures are not official DeepSeek
correctness evidence.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np


STATE_KEYS = ("kv", "index_k", "idx", "candidates")


@dataclass(frozen=True)
class PublicationFrontier:
    """State names expected to be published at/after a producer layer."""

    layer: int
    publishes: tuple[str, ...]


@dataclass
class PublicationRecord:
    """Digest snapshot at one layer publication frontier."""

    layer: int
    frontier: tuple[str, ...]
    shared_keys: tuple[str, ...]
    shared: dict[str, dict[str, Any] | None]
    cache: dict[str, Any]
    engram: dict[str, Any] | None = None


class ExplicitStatePublication(MutableMapping[str, Any]):
    """Dict-compatible explicit shared-state publication object.

    Existing oMLX layer code consumes a mapping named ``shared``.  This adapter
    intentionally preserves that interface while making producer frontiers and
    shared values observable.  It is layer/frontier oriented: callers publish a
    frontier after an existing layer call, and one frontier may contain several
    state types.
    """

    def __init__(self, frontiers: dict[int, PublicationFrontier]):
        self._data: dict[str, Any] = {}
        self.frontiers = frontiers
        self.records: list[PublicationRecord] = []

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def publish(self, layer: int, cache: Any, *, has_engram: bool = False) -> None:
        frontier = self.frontiers.get(layer)
        if frontier is None and not has_engram:
            return
        names = frontier.publishes if frontier is not None else ()
        self.records.append(
            PublicationRecord(
                layer=layer,
                frontier=names,
                shared_keys=tuple(sorted(self._data)),
                shared={name: tensor_digest(self._data.get(name)) for name in names},
                cache=cache_digest(cache),
                engram=engram_digest(cache) if has_engram else None,
            )
        )


def build_publication_frontiers(config: Any) -> dict[int, PublicationFrontier]:
    """Build producer-layer frontiers from the oMLX DeepSeek-V4.1 config."""

    by_layer: dict[int, set[str]] = {}
    for layer in getattr(config, "engram_layer_ids", []) or []:
        by_layer.setdefault(int(layer), set()).add("engram")
    for layer in getattr(config, "kv_source_layers", []) or []:
        by_layer.setdefault(int(layer), set()).add("kv")
    for layer in getattr(config, "index_source_layers", []) or []:
        by_layer.setdefault(int(layer), set()).update(("index_k", "idx"))
    candidate_layer = getattr(config, "candidate_source_layer", None)
    if candidate_layer is not None and int(candidate_layer) >= 0:
        by_layer.setdefault(int(candidate_layer), set()).add("candidates")
    return {
        layer: PublicationFrontier(layer=layer, publishes=tuple(sorted(names)))
        for layer, names in sorted(by_layer.items())
    }


def tensor_digest(x: Any) -> dict[str, Any] | None:
    """Return a small exact byte digest for an MLX/NumPy-like tensor."""

    if x is None:
        return None
    try:
        # MLX arrays are lazy; force only the tiny historical compatibility fixture tensors.
        import mlx.core as mx  # type: ignore

        mx.eval(x)
    except Exception:
        pass
    logical_shape = list(getattr(x, "shape", []))
    logical_dtype = str(getattr(x, "dtype", ""))
    try:
        arr = np.asarray(x)
        payload = np.ascontiguousarray(arr).tobytes()
        if not logical_shape:
            logical_shape = list(arr.shape)
        if not logical_dtype:
            logical_dtype = str(arr.dtype)
    except ValueError:
        # NumPy cannot consume MLX bfloat16 via PEP-3118.  Hash the raw bytes
        # through an MLX uint8 view while preserving the logical dtype/shape in
        # the digest metadata and preimage.
        import mlx.core as mx  # type: ignore

        raw = np.asarray(x.view(mx.uint8))
        payload = np.ascontiguousarray(raw).tobytes()
    h = sha256()
    h.update(logical_dtype.encode())
    h.update(str(tuple(logical_shape)).encode())
    h.update(payload)
    return {
        "type": type(x).__name__,
        "shape": logical_shape,
        "dtype": logical_dtype,
        "sha256": h.hexdigest(),
    }


def cache_digest(cache: Any) -> dict[str, Any]:
    slots = list(getattr(cache, "cache", []))
    return {
        "type": type(cache).__name__,
        "compress_ratio": getattr(cache, "compress_ratio", None),
        "size": _cache_size(cache),
        "slots": [tensor_digest(slot) for slot in slots],
    }


def engram_digest(cache: Any) -> dict[str, Any] | None:
    slots = list(getattr(cache, "cache", []))
    if len(slots) <= 6:
        return None
    return {"history_slot_6": tensor_digest(slots[6])}


def _cache_size(cache: Any) -> int | None:
    try:
        return int(cache.size())
    except Exception:
        return None


def records_to_json(records: list[PublicationRecord]) -> list[dict[str, Any]]:
    return [
        {
            "layer": r.layer,
            "frontier": list(r.frontier),
            "shared_keys": list(r.shared_keys),
            "shared": r.shared,
            "cache": r.cache,
            "engram": r.engram,
        }
        for r in records
    ]
