"""Runtime configuration contract for the ds41f scoped text release."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import importlib.util
import os
import sys
from typing import Any

DEFAULT_CHECKPOINT = Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash")
DEFAULT_OMLX = Path.home() / "omlx-0.7.0.release"
DEFAULT_RECIPE = Path("/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe")
DEFAULT_KV_ROOT = Path("/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODEL_ID = "deepseek-v4.1-flash"

ENV_VARS = {
    "checkpoint_path": "DS41F_CHECKPOINT",
    "omlx_path": "DS41F_OMLX_PATH",
    "recipe_path": "DS41F_RECIPE_PATH",
    "kv_root": "DS41F_KV_ROOT",
    "host": "DS41F_HOST",
    "port": "DS41F_PORT",
    "max_live_sessions": "DS41F_MAX_LIVE_SESSIONS",
    "trace_history_limit": "DS41F_TRACE_HISTORY_LIMIT",
    "enable_diagnostics": "DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS",
    "model_id": "DS41F_MODEL_ID",
}


@dataclass(frozen=True)
class RuntimeConfig:
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    omlx_path: Path = DEFAULT_OMLX
    recipe_path: Path = DEFAULT_RECIPE
    kv_root: Path = DEFAULT_KV_ROOT
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_live_sessions: int = 4
    trace_history_limit: int = 32
    enable_diagnostics: bool = False
    model_id: str = DEFAULT_MODEL_ID
    production_prefill_selector: str = "DENSE_P0_P7"
    mtp: str = "OFF"
    dspark: str = "OFF"
    speculative_decode: str = "OFF"

    def __post_init__(self) -> None:
        for key in ("checkpoint_path", "omlx_path", "recipe_path", "kv_root"):
            value = getattr(self, key)
            if not isinstance(value, Path):
                object.__setattr__(self, key, Path(value).expanduser())

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("checkpoint_path", "omlx_path", "recipe_path", "kv_root"):
            out[key] = str(out[key])
        return out

    def apply_environment(self) -> None:
        """Publish settings consumed by older seams that still read env vars."""
        os.environ[ENV_VARS["max_live_sessions"]] = str(self.max_live_sessions)
        os.environ[ENV_VARS["trace_history_limit"]] = str(self.trace_history_limit)
        os.environ[ENV_VARS["enable_diagnostics"]] = "1" if self.enable_diagnostics else "0"

    def apply_import_paths(self) -> None:
        candidates = [self.omlx_path]
        # Prefer an installed deepseek-recipe wheel/native extension when present.
        # A source checkout's pure Python directory is only added as a fallback.
        if importlib.util.find_spec("deepseek_recipe") is None:
            candidates.append(self.recipe_path / "deepseek-recipe-python" / "python")
        for path in candidates:
            s = str(path)
            if path.exists() and s not in sys.path:
                sys.path.insert(0, s)


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(ENV_VARS[name], str(default))).expanduser()


def _int(name: str, default: int) -> int:
    raw = os.environ.get(ENV_VARS[name])
    return default if raw is None or raw == "" else int(raw)


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(ENV_VARS[name])
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        checkpoint_path=_path("checkpoint_path", DEFAULT_CHECKPOINT),
        omlx_path=_path("omlx_path", DEFAULT_OMLX),
        recipe_path=_path("recipe_path", DEFAULT_RECIPE),
        kv_root=_path("kv_root", DEFAULT_KV_ROOT),
        host=os.environ.get(ENV_VARS["host"], DEFAULT_HOST),
        port=_int("port", DEFAULT_PORT),
        max_live_sessions=_int("max_live_sessions", 4),
        trace_history_limit=_int("trace_history_limit", 32),
        enable_diagnostics=_bool("enable_diagnostics", False),
        model_id=os.environ.get(ENV_VARS["model_id"], DEFAULT_MODEL_ID),
    )


def validate_runtime_config(config: RuntimeConfig, *, require_checkpoint: bool = True, require_recipe: bool = True, require_omlx: bool = True, require_kv_parent: bool = False) -> list[dict[str, str]]:
    """Return validation findings with status PASS/WARNING/FAIL."""
    findings: list[dict[str, str]] = []

    def add(status: str, key: str, message: str) -> None:
        findings.append({"status": status, "key": key, "message": message})

    if require_checkpoint:
        if not config.checkpoint_path.exists():
            add("FAIL", "checkpoint_path", f"checkpoint path does not exist: {config.checkpoint_path}")
        elif not config.checkpoint_path.is_dir():
            add("FAIL", "checkpoint_path", f"checkpoint path is not a directory: {config.checkpoint_path}")
        else:
            add("PASS", "checkpoint_path", str(config.checkpoint_path))

    if require_omlx:
        if not config.omlx_path.exists():
            add("FAIL", "omlx_path", f"oMLX path does not exist: {config.omlx_path}")
        else:
            add("PASS", "omlx_path", str(config.omlx_path))

    if require_recipe:
        tokenizer = config.recipe_path / "static" / "tokenizers" / "v41" / "tokenizer.json"
        if not config.recipe_path.exists():
            add("FAIL", "recipe_path", f"deepseek-recipe path does not exist: {config.recipe_path}")
        elif not tokenizer.exists():
            add("FAIL", "recipe_tokenizer", f"DeepSeek V4.1 tokenizer not found: {tokenizer}")
        else:
            add("PASS", "recipe_path", str(config.recipe_path))

    if require_kv_parent and not config.kv_root.parent.exists():
        add("FAIL", "kv_root", f"KV root parent does not exist: {config.kv_root.parent}")
    else:
        add("PASS" if config.kv_root.exists() else "WARNING", "kv_root", str(config.kv_root))

    if config.production_prefill_selector != "DENSE_P0_P7":
        add("FAIL", "production_prefill_selector", config.production_prefill_selector)
    else:
        add("PASS", "production_prefill_selector", config.production_prefill_selector)
    for key in ("mtp", "dspark", "speculative_decode"):
        value = getattr(config, key)
        add("PASS" if value == "OFF" else "FAIL", key, value)
    return findings
