"""P8 evidence package for DwarfStar FP8/MLX prefill.

This module is qualification/debug instrumentation only.  It does not select a
production backend, compile stateful layers, force MLX evaluation, or replace any
pinned-oMLX/custom-kernel route.  All records are scalar metadata; tensors and
module objects are never retained.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import importlib
import os
import time
from typing import Any, Callable, Iterator
from unittest.mock import patch


P8_ENV_FLAG = "DS41F_P8_OPTIMIZER"


def _shape(value: Any) -> tuple[int, ...] | None:
    s = getattr(value, "shape", None)
    if s is None:
        return None
    try:
        return tuple(int(x) for x in s)
    except Exception:
        return tuple(s)


def _rows(value: Any) -> int | None:
    s = _shape(value)
    if s is None:
        return None
    if len(s) >= 2:
        return int(s[1])
    if s:
        return int(s[0])
    return None


def _dtype(value: Any) -> str | None:
    d = getattr(value, "dtype", None)
    return None if d is None else str(d)


def _device(value: Any) -> str | None:
    d = getattr(value, "device", None)
    if d is None:
        return None
    return str(d)


def _hidden_dim(value: Any) -> int | None:
    s = _shape(value)
    return int(s[-1]) if s else None


def _classify_rows(rows: int | None, *, phase: str | None = None, command_kind: str | None = None) -> str:
    if command_kind == "P6_SOURCE_COMPLETE_AND_DETACH_CONE":
        return "p6_detach"
    if rows == 8192:
        return "encoder8192"
    if rows == 2048:
        return "engram2048_or_short_encoder"
    if rows == 1:
        return "ordinary_1_token_tail"
    if phase == "DECODER_SUFFIX" or (rows is not None and rows in {1 + (39 - l) * 127 for l in range(20, 40)}):
        return "decoder_suffix_Q(L)"
    if rows is not None and rows < 8192:
        return "encoder_short_final"
    if rows == 16384:
        return "encoder16384_request/two_8192_commands_expected"
    return "unknown"


@dataclass(frozen=True)
class ShapeSignature:
    region: str
    layer: int | None
    phase: str | None
    rows: int | None
    dtype: str | None
    device: str | None
    hidden_dim: int | None
    hc_multiplicity: int | None
    absolute_start_class: str
    cache_source_ratio: str | None
    command_kind: str | None
    shape_identity: str
    layer_module_identity: str
    weight_identity_class: str
    side_effect_class: str

    def as_key(self) -> tuple[Any, ...]:
        return tuple(getattr(self, f.name) for f in self.__dataclass_fields__.values())

    def to_json(self) -> dict[str, Any]:
        return dict(zip(self.__dataclass_fields__, self.as_key()))


@dataclass
class ShapeClassEntry:
    signature: ShapeSignature
    call_count: int = 0
    layers_observed: set[int] = field(default_factory=set)
    row_counts: set[int] = field(default_factory=set)

    def to_json(self) -> dict[str, Any]:
        return {
            "signature": self.signature.to_json(),
            "call_count": self.call_count,
            "layers_observed": sorted(self.layers_observed),
            "row_counts": sorted(self.row_counts),
        }


class ShapeClassRegistry:
    """Collapse actual production-call shape signatures without retaining tensors."""

    def __init__(self) -> None:
        self._entries: dict[tuple[Any, ...], ShapeClassEntry] = {}
        self.write_rows: list[dict[str, Any]] = []
        self.microtile_concat: list[dict[str, Any]] = []

    def record_command(self, *, region: str, command: Any, value: Any = None, pre: Any = None, layer_obj: Any = None,
                       cache: Any = None, shared: Any = None, absolute_start: int | None = None, hc_multiplicity: int | None = None) -> None:
        rows = int(getattr(command, "rows", 0)) if command is not None else _rows(value)
        phase = getattr(getattr(command, "phase", None), "name", None) or str(getattr(command, "phase", "")) or None
        kind = getattr(getattr(command, "kind", None), "value", None) or str(getattr(command, "kind", "")) or None
        layer = getattr(command, "layer", None)
        if layer is not None:
            layer = int(layer)
        shape = _shape(value)
        abs_cls = "none" if absolute_start is None else ("zero" if int(absolute_start) == 0 else "nonzero_dynamic")
        ratio = None
        if cache is not None and absolute_start is not None:
            try:
                c0 = cache[0]
                frontier = int(c0.item()) if hasattr(c0, "item") else int(c0[0]) if hasattr(c0, "__getitem__") else int(c0)
                ratio = f"{frontier}:{int(absolute_start)}"
            except Exception:
                ratio = "unknown"
        side = "cache+publication" if cache is not None or shared is not None else "pure_or_unknown"
        module_id = "none" if layer_obj is None else f"{type(layer_obj).__module__}.{type(layer_obj).__qualname__}:layer{layer}"
        weight_id = _weight_identity_class(layer_obj)
        sig = ShapeSignature(
            region=region,
            layer=layer,
            phase=phase,
            rows=rows,
            dtype=_dtype(value) or _dtype(pre),
            device=_device(value) or _device(pre),
            hidden_dim=_hidden_dim(value),
            hc_multiplicity=hc_multiplicity,
            absolute_start_class=abs_cls,
            cache_source_ratio=ratio,
            command_kind=kind,
            shape_identity=f"{_classify_rows(rows, phase=phase, command_kind=kind)}:{shape}:{_dtype(value)}",
            layer_module_identity=module_id,
            weight_identity_class=weight_id,
            side_effect_class=side,
        )
        entry = self._entries.setdefault(sig.as_key(), ShapeClassEntry(sig))
        entry.call_count += 1
        if layer is not None:
            entry.layers_observed.add(layer)
        if rows is not None:
            entry.row_counts.add(int(rows))

    def record_write_rows(self, *, role: str, offset: int, rows: int, base: Any, update: Any, branch: str) -> None:
        self.write_rows.append({
            "role": str(role), "offset": int(offset), "rows": int(rows),
            "base_shape": _shape(base), "update_shape": _shape(update), "implementation_branch": branch,
        })

    def record_microtile_concat(self, *, outputs: list[Any], result: Any, references_released: bool | None) -> None:
        self.microtile_concat.append({
            "number_of_microtile_outputs": len(outputs),
            "input_shapes": [_shape(v) for v in outputs],
            "concat_output_shape": _shape(result),
            "microtile_python_references_released_after_reassembly": references_released,
        })

    def report(self) -> list[dict[str, Any]]:
        return [e.to_json() for e in sorted(self._entries.values(), key=lambda x: (x.signature.region, x.signature.layer or -1, x.signature.rows or -1))]

    def to_json(self) -> dict[str, Any]:
        return {"shape_classes": self.report(), "write_rows": list(self.write_rows), "microtile_concat": list(self.microtile_concat), "tensor_free": True}


def _weight_identity_class(obj: Any) -> str:
    if obj is None:
        return "none"
    names = []
    for name, val in getattr(obj, "__dict__", {}).items():
        if name.startswith("_"):
            continue
        if hasattr(val, "weight") or _shape(val) is not None:
            names.append(name)
    return "module_weights_present:" + ",".join(sorted(names)[:8]) if names else "module_no_observed_weights"


@dataclass
class FastPathCall:
    component: str
    path: str
    function: str
    rows: int | None
    layer: str | None
    dtype: str | None
    shape_class: str
    observation: str

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ExistingKernelFastPathVerifier:
    """Qualification wrappers for actual donor Python seams.

    Wrappers are installed only inside ``instrument()`` and return the original
    function result unchanged.  Missing symbols are reported as UNAVAILABLE.
    """

    TARGETS = (
        ("omlx.patches.deepseek_v41.language", "fused_hc_projection", "HC", "fast", "DIRECT_CALL_OBSERVATION"),
        ("omlx.patches.deepseek_v41.language", "fused_hc_pre_norm", "HC", "fast", "DIRECT_CALL_OBSERVATION"),
        ("omlx.patches.deepseek_v41.language", "fused_hc_post", "HC", "fast", "DIRECT_CALL_OBSERVATION"),
        ("omlx.patches.deepseek_v41.language", "packed_sparse_attention", "attention", "fallback", "DIRECT_CALL_OBSERVATION"),
        ("omlx.patches.deepseek_v41.language", "packed_index_topk", "index", "fast", "DIRECT_CALL_OBSERVATION"),
        ("omlx.patches.deepseek_v41.language", "packed_index_scores", "index", "fast", "DIRECT_CALL_OBSERVATION"),
        ("omlx.patches.deepseek_v41.quantization", "quantize_activation", "activation_quant", "fast_or_fallback", "CALLER_BRANCH_OBSERVATION"),
        ("omlx.patches.deepseek_v41.quantization", "deepseek_mxfp4_gather_qmm_pair_concat_blocks", "moe", "fast", "DIRECT_CALL_OBSERVATION"),
        ("omlx.patches.deepseek_v41.quantization", "deepseek_affine_gather_qmm_pair_concat_blocks", "moe", "fast", "DIRECT_CALL_OBSERVATION"),
        ("omlx.patches.deepseek_v41.language", "combine_sorted_experts", "moe", "fast", "DIRECT_CALL_OBSERVATION"),
    )

    def __init__(self) -> None:
        self.calls: list[FastPathCall] = []
        self.symbol_status: dict[str, str] = {}

    def record(self, component: str, path: str, function: str, args: tuple[Any, ...], *, observation: str) -> None:
        first = next((a for a in args if _shape(a) is not None), None)
        self.calls.append(FastPathCall(component, path, function, _rows(first), _recover_layer(args), _dtype(first), _classify_rows(_rows(first)), observation))

    @contextmanager
    def instrument(self) -> Iterator["ExistingKernelFastPathVerifier"]:
        with ExitStack() as stack:
            for modname, symbol, component, path, observation in self.TARGETS:
                try:
                    mod = importlib.import_module(modname)
                    fn = getattr(mod, symbol)
                except Exception:
                    self.symbol_status[f"{modname}.{symbol}"] = "UNAVAILABLE"
                    continue
                if not callable(fn):
                    self.symbol_status[f"{modname}.{symbol}"] = "UNAVAILABLE"
                    continue
                self.symbol_status[f"{modname}.{symbol}"] = observation
                def make_wrapper(orig: Callable[..., Any], comp: str, pth: str, name: str, obs: str) -> Callable[..., Any]:
                    def wrapper(*args: Any, **kwargs: Any) -> Any:
                        self.record(comp, pth, name, args, observation=obs)
                        return orig(*args, **kwargs)
                    return wrapper
                stack.enter_context(patch.object(mod, symbol, make_wrapper(fn, component, path, symbol, observation)))
            # GLM native symbols live behind an extension module; wrap when importable without changing callers.
            for modname, symbol, component in (
                ("mlx._glm", "deepseek_v41_packed_attention", "attention"),
                ("mlx._glm", "deepseek_v41_grouped_expert", "moe"),
                ("mlx.core", "quantized_matmul", "quantized_projection"),
            ):
                try:
                    mod = importlib.import_module(modname); fn = getattr(mod, symbol)
                except Exception:
                    self.symbol_status[f"{modname}.{symbol}"] = "UNAVAILABLE"
                    continue
                self.symbol_status[f"{modname}.{symbol}"] = "DIRECT_CALL_OBSERVATION"
                def make_wrapper(orig: Callable[..., Any], comp: str, name: str) -> Callable[..., Any]:
                    def wrapper(*args: Any, **kwargs: Any) -> Any:
                        self.record(comp, "fast", name, args, observation="DIRECT_CALL_OBSERVATION")
                        return orig(*args, **kwargs)
                    return wrapper
                stack.enter_context(patch.object(mod, symbol, make_wrapper(fn, component, symbol)))
            yield self

    def report(self) -> list[dict[str, Any]]:
        agg: dict[tuple[str, str, str], dict[str, Any]] = {}
        for c in self.calls:
            k = (c.component, c.path, c.function)
            e = agg.setdefault(k, {"component": c.component, "path": c.path, "function": c.function, "calls": 0, "shape_classes": defaultdict(int), "rows": defaultdict(int), "layers": defaultdict(int), "observation": c.observation})
            e["calls"] += 1
            e["shape_classes"][c.shape_class] += 1
            if c.rows is not None: e["rows"][str(c.rows)] += 1
            if c.layer is not None: e["layers"][str(c.layer)] += 1
        out = []
        for e in agg.values():
            e = dict(e); e["shape_classes"] = dict(e["shape_classes"]); e["rows"] = dict(e["rows"]); e["layers"] = dict(e["layers"]); out.append(e)
        return sorted(out, key=lambda x: (x["component"], x["path"], x["function"]))

    def required_table(self) -> list[dict[str, Any]]:
        by_component: dict[str, dict[str, Any]] = {}
        for row in self.report():
            comp = row["component"]
            e = by_component.setdefault(comp, {"component": comp, "fast path": [], "fast calls": 0, "fallback path": [], "fallback calls": 0, "shape classes using each": {}})
            if row["path"] == "fallback":
                e["fallback path"].append(row["function"]); e["fallback calls"] += row["calls"]
            else:
                e["fast path"].append(row["function"]); e["fast calls"] += row["calls"]
            e["shape classes using each"][row["function"]] = row["shape_classes"]
        return list(by_component.values())

    def to_json(self) -> dict[str, Any]:
        return {"calls": [c.to_json() for c in self.calls], "summary": self.report(), "component_table": self.required_table(), "symbol_status": dict(self.symbol_status)}


def _recover_layer(args: tuple[Any, ...]) -> str | None:
    for a in args:
        layer = getattr(a, "_layer", None)
        if layer is not None:
            return str(layer)
    return None


@dataclass
class TimingEvent:
    name: str
    elapsed_s: float
    nesting: tuple[str, ...] = ()
    inclusive_overlapping: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "elapsed_s": self.elapsed_s, "nesting": list(self.nesting), "inclusive_overlapping": self.inclusive_overlapping, "metadata": dict(self.metadata)}


class PerformanceTelemetry:
    """Hierarchical timing and memory telemetry around existing boundaries only."""

    def __init__(self, mx: Any | None = None) -> None:
        self.mx = mx
        self.events: list[TimingEvent] = []
        self.materialization_boundaries: list[dict[str, Any]] = []
        self.graph_retention_proxies: list[dict[str, Any]] = []

    @contextmanager
    def time_region(self, name: str, *, nesting: tuple[str, ...] = (), inclusive_overlapping: bool = True, **metadata: Any) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.events.append(TimingEvent(name, time.perf_counter() - t0, nesting, inclusive_overlapping, metadata))

    def memory_snapshot(self) -> dict[str, Any]:
        mx = self.mx
        if mx is None:
            try: mx = importlib.import_module("mlx.core")
            except Exception: mx = None
        def call(name: str) -> int | None:
            f = getattr(mx, name, None) if mx is not None else None
            if f is None: return None
            try: return int(f())
            except Exception: return None
        return {"active_memory": call("get_active_memory"), "cache_memory": call("get_cache_memory"), "peak_memory": call("get_peak_memory")}

    def record_materialization_boundary(self, name: str, before: dict[str, Any], after: dict[str, Any], elapsed_s: float, **metadata: Any) -> None:
        self.materialization_boundaries.append({"boundary": name, "before": dict(before), "after": dict(after), "elapsed_s": float(elapsed_s), **metadata})

    def record_graph_retention_proxy(self, point: str, *, live: dict[str, bool] | None = None) -> None:
        self.graph_retention_proxies.append({"point": point, **self.memory_snapshot(), "logical_live_references": dict(live or {})})

    def to_json(self) -> dict[str, Any]:
        return {"timing_events": [e.to_json() for e in self.events], "materialization_boundaries": list(self.materialization_boundaries), "graph_retention_proxies": list(self.graph_retention_proxies), "overlap_warning": "ENCODE_ROWS and Engram micro-pipeline are inclusive/overlapping; do not sum percentages."}


class GraphReusePolicy:
    implemented = False
    reason = "planned interface only; no graph compilation or production work in this P8 evidence package"


class MaterializationOptimizer:
    implemented = False
    reason = "planned interface only; telemetry records existing boundaries but does not alter materialization"


class P8ExecutionOptimizer:
    """Coherent opt-in P8 instrumentation package."""

    def __init__(self, *, enabled: bool = False, mx: Any | None = None) -> None:
        self.enabled = bool(enabled)
        self.shape_registry = ShapeClassRegistry()
        self.fast_path_verifier = ExistingKernelFastPathVerifier()
        self.telemetry = PerformanceTelemetry(mx=mx)
        self.graph_reuse_policy = GraphReusePolicy()
        self.materialization_optimizer = MaterializationOptimizer()

    @classmethod
    def from_env(cls, *, mx: Any | None = None) -> "P8ExecutionOptimizer | None":
        return cls(enabled=True, mx=mx) if os.environ.get(P8_ENV_FLAG) in {"1", "true", "TRUE", "yes"} else None

    @contextmanager
    def verification_context(self) -> Iterator["P8ExecutionOptimizer"]:
        if not self.enabled:
            yield self
            return
        with self.fast_path_verifier.instrument():
            yield self

    def to_json(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "shape_registry": self.shape_registry.to_json(), "fast_path_verifier": self.fast_path_verifier.to_json(), "performance_telemetry": self.telemetry.to_json(), "graph_reuse_policy": {"implemented": False, "reason": GraphReusePolicy.reason}, "materialization_optimizer": {"implemented": False, "reason": MaterializationOptimizer.reason}}
