#!/usr/bin/env python3
"""Line-oriented worker for the M0 oMLX bridge.

Protocol: one JSON request per stdin line, one JSON response per stdout line.
The worker owns the model process so the API server can keep tensor execution out
of the request handler.  It is deliberately narrow and conservative for M0.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .omlx_core import OmlxRuntime, OmlxRuntimeConfig


class Worker:
    def __init__(self) -> None:
        self.runtime = OmlxRuntime(
            OmlxRuntimeConfig(
                omlx_path=Path(os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2"))),
                checkpoint_path=Path(os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash")),
                engram_ssd_offload=os.environ.get("DS41F_ENGRAM_SSD_OFFLOAD", "1") != "0",
                preserve_mtp=None,
            )
        )
        self.loaded = False
        self.load_seconds: float | None = None

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        t0 = time.perf_counter()
        self.runtime.load_model()
        self.load_seconds = time.perf_counter() - t0
        self.loaded = True

    def complete(self, req: dict[str, Any]) -> dict[str, Any]:
        self.ensure_loaded()
        processor = self.runtime.processor
        model = self.runtime.model
        if processor is None or model is None:
            raise RuntimeError("oMLX runtime did not return model and processor")
        tokenizer = processor.tokenizer
        messages = req["messages"]
        max_tokens = int(req.get("max_tokens") or 16)
        temperature = float(req.get("temperature") or 0.0)
        enable_thinking = True
        thinking = req.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            enable_thinking = False
        prompt_ids = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            reasoning_effort=req.get("reasoning_effort"),
            tools=req.get("tools"),
        )

        import mlx.core as mx  # type: ignore
        from mlx_lm.generate import generate_step  # type: ignore

        generated: list[int] = []
        t0 = time.perf_counter()
        # M0 uses greedy direct generation. Nonzero temperature is preserved in
        # provenance but not yet sampled here; broader sampling is a later gate.
        del temperature
        for token, _logprobs in generate_step(mx.array(prompt_ids), model.language_model, max_tokens=max_tokens):
            value = int(token.item() if hasattr(token, "item") else token)
            generated.append(value)
            if len(generated) >= max_tokens:
                break
        mx.synchronize()
        gen_seconds = time.perf_counter() - t0
        return {
            "ok": True,
            "prompt_ids": prompt_ids,
            "generated_ids": generated,
            "text": tokenizer.decode(generated),
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(prompt_ids) + len(generated),
            },
            "finish_reason": "length" if len(generated) >= max_tokens else "stop",
            "timing": {
                "load_seconds": self.load_seconds,
                "generation_seconds": gen_seconds,
            },
            "memory": {
                "active": mx.get_active_memory(),
                "peak": mx.get_peak_memory(),
                "cache": mx.get_cache_memory(),
            },
        }


def main() -> int:
    worker = Worker()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "shutdown":
                print(json.dumps({"ok": True, "shutdown": True}), flush=True)
                return 0
            if req.get("op") != "chat.completions":
                raise ValueError("unsupported op")
            resp = worker.complete(req)
        except BaseException as exc:
            resp = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(resp, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
