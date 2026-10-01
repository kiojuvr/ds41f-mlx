#!/usr/bin/env python3
"""Qualification-only subprocess watchdog for real MLX long-run workers.

The supervisor is outside model semantics: workers emit JSONL heartbeat/progress
records on stdout; the parent terminates a stuck worker instead of waiting for a
running segment to return.
"""
from __future__ import annotations

import argparse, json, selectors, subprocess, sys, time
from dataclasses import dataclass
from typing import Any


@dataclass
class SupervisorResult:
    status: str
    reason: str | None
    last_progress: dict[str, Any] | None
    elapsed_wall_s: float
    returncode: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "last_progress": self.last_progress,
            "elapsed_wall_s": self.elapsed_wall_s,
            "returncode": self.returncode,
        }


def _is_progress(record: dict[str, Any]) -> bool:
    return record.get("event") in {"progress", "heartbeat", "segment", "command"} or "segment" in record


def supervise(argv: list[str], *, no_progress_timeout_s: float = 180.0, min_tokens_per_s: float = 10.0, baselines: dict[str, float] | None = None) -> SupervisorResult:
    start = time.monotonic(); last_progress_time = start; last_progress = None
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    sel = selectors.DefaultSelector(); sel.register(proc.stdout, selectors.EVENT_READ)
    try:
        while True:
            ready = sel.select(timeout=0.1)
            line = proc.stdout.readline() if ready else ""
            now = time.monotonic()
            if line:
                print(line, end="")
                try:
                    rec = json.loads(line)
                except Exception:
                    rec = None
                if isinstance(rec, dict) and _is_progress(rec):
                    payload = rec.get("segment", rec)
                    if not isinstance(payload, dict):
                        payload = rec
                    last_progress = payload; last_progress_time = now
                    elapsed = payload.get("elapsed_wall_s")
                    count = payload.get("count")
                    mode = payload.get("mode")
                    if elapsed and count and count >= 4096 and count / elapsed < min_tokens_per_s:
                        proc.terminate()
                        return SupervisorResult("ABORT", "below_min_tokens_per_second", last_progress, now - start, proc.wait(timeout=10))
                    if baselines and elapsed and count and mode in baselines and elapsed > 4.0 * baselines[mode] * (count / 16384.0):
                        proc.terminate()
                        return SupervisorResult("ABORT", "exceeded_4x_normalized_baseline", last_progress, now - start, proc.wait(timeout=10))
            elif proc.poll() is not None:
                status = "PASS" if proc.returncode == 0 else "FAIL"
                return SupervisorResult(status, None if status == "PASS" else "worker_failed", last_progress, now - start, proc.returncode)
            if now - last_progress_time > no_progress_timeout_s:
                proc.terminate()
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); rc = proc.wait()
                return SupervisorResult("ABORT", "no_semantic_progress_timeout", last_progress, now - start, rc)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


def _selftest_worker(kind: str):
    if kind == "ok":
        for i in range(3):
            print(json.dumps({"event":"progress","case":"selftest","segment":0,"command_kind":"noop","layer":i,"absolute_start":i,"C":0,"E":i,"D":0,"T":3,"monotonic":time.monotonic()}), flush=True)
            time.sleep(0.05)
        return 0
    if kind == "stall":
        print(json.dumps({"event":"progress","case":"selftest","segment":0,"command_kind":"begin","layer":0,"absolute_start":0,"C":0,"E":0,"D":0,"T":1,"monotonic":time.monotonic()}), flush=True)
        time.sleep(2.0)
        return 0
    raise SystemExit(2)


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--selftest", choices=("ok","stall"))
    ap.add_argument("--worker-selftest", choices=("ok","stall"))
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args=ap.parse_args(argv)
    if args.worker_selftest:
        return _selftest_worker(args.worker_selftest)
    if args.selftest:
        cmd=[sys.executable, __file__, "--worker-selftest", args.selftest]
        res=supervise(cmd, no_progress_timeout_s=args.timeout)
        print(json.dumps({"supervisor_selftest":res.to_json()}, indent=2))
        return 0 if (args.selftest == "ok" and res.status == "PASS") or (args.selftest == "stall" and res.status == "ABORT") else 1
    if not args.cmd:
        ap.error("provide --selftest or worker command")
    res=supervise(args.cmd, no_progress_timeout_s=args.timeout)
    print(json.dumps({"supervisor":res.to_json()}, indent=2))
    return 0 if res.status == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
