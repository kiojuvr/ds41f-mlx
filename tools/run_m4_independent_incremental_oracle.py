#!/usr/bin/env python3
"""Bounded M4 independent incremental-oracle wrapper.

Authority source: the historical `deepseek-v41-flash-mlx` native/reference full
backbone test.  That runtime is not production topology for ds41f; here it is
used only as a qualification oracle because it executes token-serial/chunk,
continuation, fork, reset, state, Engram, compressor/index lifecycle checks on
real official-checkpoint tokens.

Current limitation: the retained historical CLI reports PASS/FAIL and lifecycle
coverage but does not export the final logits bytes/digest.  Therefore this tool
records usable independent lifecycle evidence for [0,3,15] and deliberately
leaves the M4 full-logits digest gate incomplete until a digest-exporting oracle
entry point is added.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def sha(path: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--historical", default="/Volumes/SDXC-512/deepseek-v41-flash-mlx")
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--out", default="artifacts/m4/independent-incremental-oracle/result.json")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()
    hist = Path(args.historical)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema": "ds41f.m4.independent-incremental-oracle.v1",
        "fixture": {"prefill_tokens": [0, 3], "incremental_input": [15], "full_token_sequence_for_historical_cli": [0, 3, 15]},
        "authority_source": {
            "repository": str(hist),
            "role": "qualification_oracle_only_not_production_topology",
            "runner": "tools/benchmark/run_text_backbone_reference.sh with TOKENS_FILE='0 3 15'",
            "binary_sha256": sha(hist / "build-mlx/dsv41-text-backbone-test"),
            "test_source_sha256": sha(hist / "tests/attention/test_text_backbone.cpp"),
        },
        "production_digest_to_qualify": "bbc86311483405ba433a11c3b77eddc9161aed5f74bc3667a3b67929c701c7ce",
        "oracle_logits_digest": None,
        "digest_gate_complete": False,
        "match": None,
        "limitation": "historical retained CLI does not export final logits bytes/digest; it only proves independent token-serial/chunk continuation/state exactness for the supplied token sequence",
    }
    if not args.skip_run:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("0 3 15\n")
            token_file = f.name
        env = os.environ.copy()
        env.update({"TOKENS_FILE": token_file, "CHECKPOINT": args.checkpoint})
        start = time.perf_counter()
        proc = subprocess.run(
            ["bash", "tools/benchmark/run_text_backbone_reference.sh"],
            cwd=hist,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
        rec["historical_run"] = {
            "returncode": proc.returncode,
            "elapsed_s": time.perf_counter() - start,
            "stdout_tail": proc.stdout[-4000:],
            "pass_lifecycle": proc.returncode == 0 and "PASS: 3 token IDs -> encoder 0..19 -> decoder 20..39 -> logits" in proc.stdout,
        }
        # Extract run dir from first line: Logs: artifacts/text-backbone/run-...
        for line in proc.stdout.splitlines():
            if line.startswith("Logs: "):
                rec["historical_run"]["artifact_dir"] = str(hist / line.split("Logs: ", 1)[1])
                break
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if rec.get("historical_run", {}).get("returncode", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
