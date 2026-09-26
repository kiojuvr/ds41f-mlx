#!/usr/bin/env python3
"""Compare short-context M0.5 performance records.

This does not run inference. It consumes JSON records from oMLX baseline and the
new runtime and produces an explicit pass/fail comparison before any long-context
qualification is allowed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def metric(record: dict[str, Any], *names: str) -> float | None:
    cur: Any = record
    for name in names:
        if not isinstance(cur, dict) or name not in cur:
            return None
        cur = cur[name]
    return float(cur) if isinstance(cur, (int, float)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--out", default="artifacts/m0_5/comparison.json")
    ap.add_argument("--max-prefill-regression", type=float, default=0.15, help="fractional slowdown allowed before blocking long-context work")
    ap.add_argument("--max-decode-regression", type=float, default=0.15)
    args = ap.parse_args()

    b = load(args.baseline)
    c = load(args.candidate)
    checks = []

    def add(name: str, b_value: float | None, c_value: float | None, larger_better: bool, max_regression: float) -> None:
        if b_value is None or c_value is None or c_value < 0 or b_value < 0:
            checks.append({"name": name, "status": "missing", "pass": False, "baseline": b_value, "candidate": c_value})
            return
        if b_value == 0:
            if larger_better:
                passed = c_value >= 0
            else:
                passed = c_value <= max_regression
            checks.append({"name": name, "baseline": b_value, "candidate": c_value, "ratio": None, "regression": c_value, "pass": passed})
            return
        ratio = c_value / b_value
        if larger_better:
            passed = ratio >= 1.0 - max_regression
            regression = 1.0 - ratio
        else:
            passed = ratio <= 1.0 + max_regression
            regression = ratio - 1.0
        checks.append({"name": name, "baseline": b_value, "candidate": c_value, "ratio": ratio, "regression": regression, "pass": passed})

    add("prefill_tokens_per_second", metric(b, "prefill_tokens_per_second"), metric(c, "prefill_tokens_per_second"), True, args.max_prefill_regression)
    add("decode_tokens_per_second", metric(b, "decode_tokens_per_second"), metric(c, "decode_tokens_per_second"), True, args.max_decode_regression)
    add("decode_tpt_seconds", metric(b, "decode_tpt_seconds"), metric(c, "decode_tpt_seconds"), False, args.max_decode_regression)
    add("peak_memory_bytes", metric(b, "peak_memory_bytes"), metric(c, "peak_memory_bytes"), False, 0.10)
    add("swap_delta_bytes", metric(b, "swap_delta_bytes"), metric(c, "swap_delta_bytes"), False, 0.00)

    passed = all(item.get("pass") for item in checks)
    result = {
        "schema": "ds41f.m0_5.short-perf-comparison.v1",
        "passed": passed,
        "decision": "long-context qualification may proceed only if passed is true" if passed else "block long-context qualification; investigate short-context performance gap first",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "checks": checks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
