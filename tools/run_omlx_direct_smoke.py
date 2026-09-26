#!/usr/bin/env python3
"""Direct oMLX smoke runner for M0.

This script intentionally loads the full model only when invoked.  It is not run
as part of ordinary unit checks.  The generation implementation is conservative
and may need adjustment to the exact mlx-lm version installed with oMLX; failures
are recorded as M0 integration evidence rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Hello", help="raw prompt text unless --chat is set; with --chat this is the user message content")
    ap.add_argument("--chat", action="store_true", help="apply the DeepSeek-V4.1 chat template to --prompt before generation")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--out", default="artifacts/m0/omlx-direct-smoke/result.json")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    ap.add_argument("--engram-resident", action="store_true", help="load Engram embeddings into Metal/RAM; default is SSD-backed offload for M3 Ultra 512GB")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rt = OmlxRuntime(OmlxRuntimeConfig(omlx_path=Path(args.omlx), checkpoint_path=Path(args.checkpoint), engram_ssd_offload=not args.engram_resident))
    record: dict = {
        "schema": "ds41f.m0.omlx-direct-smoke.v1",
        "prompt": args.prompt,
        "chat": args.chat,
        "max_tokens": args.max_tokens,
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
        "engram_ssd_offload": not args.engram_resident,
    }
    try:
        model, processor = rt.load_model()
        record["load_seconds"] = time.perf_counter() - started
        tokenizer = processor.tokenizer
        if args.chat:
            messages = [{"role": "user", "content": args.prompt}]
            prompt_ids = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            record["messages"] = messages
        else:
            prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        record["prompt_ids"] = prompt_ids

        import mlx.core as mx  # type: ignore
        from mlx_lm.generate import generate_step  # type: ignore

        tokens = []
        step_started = time.perf_counter()
        for token, _logprobs in generate_step(mx.array(prompt_ids), model.language_model, max_tokens=args.max_tokens):
            value = int(token.item() if hasattr(token, "item") else token)
            tokens.append(value)
            if len(tokens) >= args.max_tokens:
                break
        mx.synchronize()
        record.update(
            ok=True,
            generated_ids=tokens,
            text=tokenizer.decode(tokens),
            generation_seconds=time.perf_counter() - step_started,
            active_memory=mx.get_active_memory(),
            peak_memory=mx.get_peak_memory(),
            cache_memory=mx.get_cache_memory(),
        )
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        rt.close()
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
