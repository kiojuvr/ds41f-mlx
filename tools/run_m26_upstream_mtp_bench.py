#!/usr/bin/env python3
"""M26 diagnostic-only stock oMLX DeepSeek V4.1 DSpark/MTP benchmark.

This script intentionally avoids ds41f serving/P7/persistence paths.  It loads a
stock oMLX checkout in an isolated PYTHONPATH slot, uses normal upstream prompt
prefill through BatchGenerator, and records speculative telemetry sufficient to
compare against the M25 terminal-only priming probe.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def make_prompt(tokenizer: Any, target_tokens: int) -> list[int]:
    code = """
# Python performance diagnostic: implement a streaming tokenizer and parser.
from dataclasses import dataclass
from typing import Iterable, Iterator

@dataclass
class Token:
    kind: str
    value: str
    line: int
    column: int

def tokenize(source: str) -> Iterator[Token]:
    line = column = 1
    buffer: list[str] = []
    for ch in source:
        if ch.isidentifier() or ch.isdigit() or ch == '_':
            buffer.append(ch)
            column += 1
            continue
        if buffer:
            yield Token('name', ''.join(buffer), line, column - len(buffer))
            buffer.clear()
        if ch == '\n':
            line += 1
            column = 1
        elif not ch.isspace():
            yield Token('punct', ch, line, column)
            column += 1
        else:
            column += 1

def parse_assignments(tokens: Iterable[Token]) -> dict[str, str]:
    out: dict[str, str] = {}
    pending: str | None = None
    for tok in tokens:
        if tok.kind == 'name' and pending is None:
            pending = tok.value
        elif tok.value == '=' and pending is not None:
            out[pending] = ''
        elif pending is not None and pending in out:
            out[pending] += tok.value
        elif tok.value == ';':
            pending = None
    return out
""".strip()
    text = (code + "\n\n# Continue with careful edge cases and optimized implementation.\n")
    # Overshoot, then truncate to the requested context length.
    ids: list[int] = []
    while len(ids) < target_tokens:
        ids.extend(tokenizer.encode(text))
    return ids[:target_tokens]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omlx-path", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contexts", default="4096,65536")
    ap.add_argument("--mtp", choices=("ON", "OFF"), required=True)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup-context", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sampling", choices=("categorical", "greedy"), default="categorical")
    ap.add_argument("--engram", choices=("ssd", "ram"), default="ssd")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    head = git(args.omlx_path, "rev-parse", "HEAD")
    dirty = git(args.omlx_path, "status", "--short")
    if dirty and not args.allow_dirty:
        raise SystemExit(f"oMLX checkout is dirty; use --allow-dirty to record diagnostic run:\n{dirty}")
    sys.path.insert(0, str(args.omlx_path))

    import mlx.core as mx
    import psutil
    import omlx.scheduler  # noqa: F401 - installs normal scheduler patches
    from mlx_lm.generate import BatchGenerator, generation_stream
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.mlx_lm_mtp import batch_generator as mtp, cache_rollback

    cache_rollback.apply(); mtp.apply()

    def memory() -> dict[str, Any]:
        return {
            "rss_bytes": psutil.Process().memory_info().rss,
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
            "mlx_cache_bytes": mx.get_cache_memory(),
        }

    result: dict[str, Any] = {
        "schema": "ds41f.m26.upstream-mtp-bench.v1",
        "qualification": "DIAGNOSTIC_ONLY_NOT_PRODUCTION",
        "ds41f_commit": git(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        "omlx_path": str(args.omlx_path),
        "omlx_revision": head,
        "omlx_dirty_status": dirty.splitlines(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {n: md.version(n) for n in ("omlx", "mlx", "mlx-lm")},
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("OMLX_", "MLX_", "DS41F_"))},
        "settings": {
            "mtp": args.mtp,
            "depth": args.depth,
            "max_tokens": args.max_tokens,
            "sampling": args.sampling,
            "temperature": args.temperature,
            "top_p": 1,
            "top_k": 0,
            "prefix_cache": False,
            "engram_ssd_offload": args.engram == "ssd",
            "priming_mode": "stock upstream BatchGenerator prompt prefill; full native DSpark prompt capture when MTP ON",
            "warmup_context": args.warmup_context,
        },
        "source_sha256": {},
        "load": None,
        "warmup": None,
        "runs": [],
        "status": "RUNNING",
    }
    for rel in ("omlx/patches/deepseek_v41/mtp.py", "omlx/patches/deepseek_v41/dspark.py", "omlx/patches/deepseek_v41/language.py", "omlx/patches/deepseek_v41/loading.py"):
        result["source_sha256"][rel] = sha256(args.omlx_path / rel)

    def save() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    def sampler(logits):
        if args.sampling == "greedy" or args.temperature == 0:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits / args.temperature)

    model = None
    try:
        t0 = time.perf_counter()
        model, processor = load(args.checkpoint, preserve_mtp=args.mtp == "ON", engram_ssd_offload=args.engram == "ssd")
        mx.synchronize()
        lm = model.language_model
        lm.configure_mtp(args.mtp == "ON", args.depth)
        result["load"] = {"seconds": time.perf_counter() - t0, **memory()}
        result["model_config"] = {k: getattr(lm._config, k, None) for k in ("preserve_mtp", "dspark_block_size", "dspark_target_layer_ids", "n_mtp_layers")}
        tokenizer = processor.tokenizer

        def one_run(context: int, label: str) -> dict[str, Any]:
            mx.random.seed(12345 + context)
            lm.configure_mtp(args.mtp == "ON", args.depth)
            ids = make_prompt(tokenizer, context)
            bg = BatchGenerator(lm, max_tokens=args.max_tokens, sampler=sampler, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=2048, stream=generation_stream)
            rec: dict[str, Any] = {"label": label, "context": context, "prompt_tokens": len(ids), "steps": [], "memory_before": memory()}
            try:
                with mx.stream(generation_stream):
                    t = time.perf_counter()
                    uid = bg.insert([ids], max_tokens=[args.max_tokens])[0]
                    prefill_done = False
                    while not prefill_done:
                        pr, gr = bg.next(); mx.synchronize(generation_stream)
                        rec.setdefault("prefill_responses", 0)
                        rec["prefill_responses"] += len(pr)
                        prefill_done = any(getattr(r, "end_of_prompt", False) for r in pr)
                        if gr:
                            raise RuntimeError("generation before prompt completion")
                    rec["prefill_seconds_including_scheduler"] = time.perf_counter() - t
                    total = 0
                    tdecode = time.perf_counter()
                    while total < args.max_tokens:
                        ts = time.perf_counter()
                        _, gr = bg.next(); mx.synchronize(generation_stream)
                        sec = time.perf_counter() - ts
                        if len(gr) != 1:
                            raise RuntimeError(f"expected singleton response, got {len(gr)}")
                        r = gr[0]
                        total += 1
                        state = getattr(bg._generation_batch, "_omlx_mtp_state", None)
                        step = {"seconds": sec, "token": int(r.token), "finish_reason": r.finish_reason}
                        if state is not None:
                            step.update({
                                "queue_len": len(state.queue),
                                "depth": state.depth,
                                "adaptive_depth": getattr(state.controller, "cur", None),
                                "target_offsets": [c.size() for c in bg._generation_batch.prompt_cache],
                            })
                            rec["last_stats"] = dataclasses.asdict(state.stats)
                        rec["steps"].append(step)
                        if r.finish_reason is not None:
                            break
                    rec["decode_seconds"] = time.perf_counter() - tdecode
                    rec["generated_tokens"] = total
                    rec["decode_tok_s"] = total / rec["decode_seconds"]
                    rec["first_token_seconds"] = rec["steps"][0]["seconds"] if rec["steps"] else None
                    rec["memory_after"] = memory()
                    stats = rec.get("last_stats") or {}
                    drafted = sum(stats.get("depth_drafted", [])) if stats else 0
                    rec["acceptance_rate"] = (stats.get("accepts", 0) / drafted) if drafted else None
                    rec["accepted_per_cycle"] = (stats.get("accepts", 0) / stats.get("cycles", 0)) if stats.get("cycles") else None
                    rec["primed_context_present_after_prefill"] = getattr(lm, "_omlx_mtp_prime_ctx", None) is not None
                    rec["uid"] = int(uid)
            finally:
                try:
                    bg.remove(list(bg._generation_batch.uids))
                finally:
                    bg.close()
            return rec

        if args.warmup_context:
            result["warmup"] = one_run(args.warmup_context, "warmup")
            save()
        for c in [int(x) for x in args.contexts.split(",") if x.strip()]:
            rec = one_run(c, "measure")
            result["runs"].append(rec)
            save()
            print(json.dumps({"context": c, "mtp": args.mtp, "tok_s": rec.get("decode_tok_s"), "acceptance": rec.get("acceptance_rate")}))
        result["status"] = "COMPLETED_DIAGNOSTIC"
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = repr(exc)
        raise
    finally:
        if model is not None:
            model.close()
        save()


if __name__ == "__main__":
    main()
