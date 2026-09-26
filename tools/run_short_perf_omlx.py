#!/usr/bin/env python3
"""M0.5 diagnostic short-context performance runner for direct oMLX.

This is intentionally bounded and is not a long-context qualification tool.
It loads the official checkpoint through the local known-good oMLX runtime with
SSD-backed Engram by default, then records prefill/decode timing for one prompt.
This direct generate_step path is not automatically equivalent to the oMLX
production benchmark/generation topology; treat mismatches as diagnostic input.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig


def rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB. This project is macOS-targeted.
        return int(value)
    except Exception:
        return None


def _size_to_bytes(value: str, unit: str) -> int:
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}.get(unit.upper(), 1)
    return int(float(value) * scale)


def system_memory_snapshot() -> dict:
    """Best-effort macOS system swap/compressor snapshot."""
    snapshot: dict = {}
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        # total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)
        for name in ("total", "used", "free"):
            m = re.search(rf"{name}\s*=\s*([0-9.]+)([KMGT])", out)
            if m:
                snapshot[f"swap_{name}_bytes"] = _size_to_bytes(m.group(1), m.group(2))
        snapshot["swapusage_raw"] = out.strip()
    except Exception as exc:
        snapshot["swapusage_error"] = repr(exc)
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page_size_match = re.search(r"page size of (\d+) bytes", out)
        page_size = int(page_size_match.group(1)) if page_size_match else None
        snapshot["vm_stat_page_size"] = page_size
        fields = {
            "Pages stored in compressor": "compressor_pages_stored",
            "Pages occupied by compressor": "compressor_pages_occupied",
            "Compressions": "compressions",
            "Decompressions": "decompressions",
            "Swapins": "swapins",
            "Swapouts": "swapouts",
            "Pageins": "pageins",
            "Pageouts": "pageouts",
        }
        for label, key in fields.items():
            m = re.search(rf"{re.escape(label)}:\s*([0-9]+)\.", out)
            if m:
                snapshot[key] = int(m.group(1))
    except Exception as exc:
        snapshot["vm_stat_error"] = repr(exc)
    return snapshot


def delta(after: dict, before: dict, key: str) -> int | None:
    if key in after and key in before and isinstance(after[key], int) and isinstance(before[key], int):
        return after[key] - before[key]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ping")
    ap.add_argument("--repeat", type=int, default=1, help="repeat prompt text to form a bounded synthetic prompt")
    ap.add_argument("--target-prompt-tokens", type=int, default=0, help="expand the prompt until tokenized length reaches this bounded target")
    ap.add_argument("--chat", action="store_true", default=True)
    ap.add_argument("--raw", dest="chat", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--omlx", default=os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2")))
    ap.add_argument("--engram-resident", action="store_true")
    args = ap.parse_args()

    out = Path(args.out or f"artifacts/m0_5/short-perf-omlx/run-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out.with_suffix(out.suffix + ".progress.jsonl")
    progress_file = progress_path.open("w", buffering=1)

    prompt = "\n".join([args.prompt] * max(1, args.repeat))
    system_before = system_memory_snapshot()
    record: dict = {
        "schema": "ds41f.m0_5.short-perf-omlx.v1",
        "checkpoint": args.checkpoint,
        "omlx": args.omlx,
        "engram_ssd_offload": not args.engram_resident,
        "chat": args.chat,
        "prompt_repeat": args.repeat,
        "target_prompt_tokens": args.target_prompt_tokens,
        "max_tokens": args.max_tokens,
    }

    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    rt = OmlxRuntime(
        OmlxRuntimeConfig(
            omlx_path=Path(args.omlx),
            checkpoint_path=Path(args.checkpoint),
            engram_ssd_offload=not args.engram_resident,
        )
    )
    try:
        load_start = time.perf_counter()
        model, processor = rt.load_model()
        load_seconds = time.perf_counter() - load_start
        tokenizer = processor.tokenizer
        def encode(candidate: str):
            if args.chat:
                msgs = [{"role": "user", "content": candidate}]
                return processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True), msgs
            return tokenizer.encode(candidate, add_special_tokens=False), None

        prompt_ids, messages = encode(prompt)
        if args.target_prompt_tokens > 0:
            parts = [prompt]
            while len(prompt_ids) < args.target_prompt_tokens:
                parts.append(args.prompt)
                prompt = "\n".join(parts)
                prompt_ids, messages = encode(prompt)
                if len(parts) > args.target_prompt_tokens * 4:
                    raise RuntimeError("target prompt expansion did not converge")
        if messages is not None:
            record["messages"] = messages

        import mlx.core as mx  # type: ignore
        from mlx_lm.generate import generate_step  # type: ignore

        progress_events = []
        gen_start = None
        prefill_done = None

        def progress(done: int, total: int) -> None:
            nonlocal prefill_done
            now = time.perf_counter()
            event = {"kind": "prefill_progress", "done": done, "total": total, "seconds_from_start": None if gen_start is None else now - gen_start}
            progress_events.append(event)
            progress_file.write(json.dumps(event, sort_keys=True) + "\n")
            if gen_start is not None and done == total and prefill_done is None:
                prefill_done = now

        generated: list[int] = []
        token_times: list[float] = []
        gen_start = time.perf_counter()
        iterator = generate_step(
            mx.array(prompt_ids),
            model.language_model,
            max_tokens=args.max_tokens,
            prompt_progress_callback=progress,
        )
        for token, _logprobs in iterator:
            value = int(token.item() if hasattr(token, "item") else token)
            now = time.perf_counter()
            generated.append(value)
            token_times.append(now - gen_start)
            progress_file.write(json.dumps({"kind": "token", "index": len(generated), "token": value, "seconds_from_start": token_times[-1]}, sort_keys=True) + "\n")
            if stop_requested or len(generated) >= args.max_tokens:
                break
        mx.synchronize()
        gen_end = time.perf_counter()

        prefill_seconds = (prefill_done - gen_start) if prefill_done is not None else None
        decode_start = prefill_done or (token_times[0] + gen_start if token_times else gen_start)
        decode_seconds = gen_end - decode_start
        prompt_tokens = len(prompt_ids)
        completion_tokens = len(generated)
        system_after = system_memory_snapshot()
        swap_used_before = system_before.get("swap_used_bytes")
        swap_used_after = system_after.get("swap_used_bytes")
        swap_delta_bytes = (
            swap_used_after - swap_used_before
            if isinstance(swap_used_after, int) and isinstance(swap_used_before, int)
            else None
        )
        record.update(
            ok=True,
            load_seconds=load_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            generated_ids=generated,
            text=tokenizer.decode(generated),
            prefill_seconds=prefill_seconds,
            prefill_tokens_per_second=(prompt_tokens / prefill_seconds) if prefill_seconds and prefill_seconds > 0 else None,
            decode_seconds=decode_seconds,
            decode_tokens_per_second=(completion_tokens / decode_seconds) if decode_seconds > 0 else None,
            decode_tpt_seconds=(decode_seconds / completion_tokens) if completion_tokens else None,
            generation_seconds=gen_end - gen_start,
            token_times_seconds=token_times,
            progress_events=progress_events,
            progress_path=str(progress_path),
            stopped_early=stop_requested,
            active_memory_bytes=mx.get_active_memory(),
            peak_memory_bytes=mx.get_peak_memory(),
            cache_memory_bytes=mx.get_cache_memory(),
            max_rss_bytes=rss_bytes(),
            system_memory_before=system_before,
            system_memory_after=system_after,
            swap_delta_bytes=swap_delta_bytes,
            swapins_delta=delta(system_after, system_before, "swapins"),
            swapouts_delta=delta(system_after, system_before, "swapouts"),
            compressions_delta=delta(system_after, system_before, "compressions"),
            decompressions_delta=delta(system_after, system_before, "decompressions"),
        )
    except BaseException as exc:
        record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        progress_file.close()
        rt.close()

    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
