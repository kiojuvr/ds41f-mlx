#!/usr/bin/env python3
"""Bounded M4 independent incremental-oracle wrapper.

Authority source: the historical `deepseek-v41-flash-mlx` native/reference full
backbone test.  That runtime is not production topology for ds41f; here it is
used only as a qualification oracle because it executes token-serial/chunk,
continuation, fork, reset, state, Engram, compressor/index lifecycle checks on
real official-checkpoint tokens.

The historical test has an opt-in `DSV41_ORACLE_LOGITS_OUT` export hook for
this qualification task.  Normal historical test behavior is unchanged unless
that environment variable is set.
"""

from __future__ import annotations

import argparse
import hashlib
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
        "oracle_export_policy": "opt-in DSV41_ORACLE_LOGITS_OUT binary float32 logits export; historical runtime remains qualification-only",
    }
    if not args.skip_run:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            token_path = tmpdir / "tokens.txt"
            logits_path = tmpdir / "logits.bin"
            token_path.write_text("0 3 15\n")
            env = os.environ.copy()
            env.update({"TOKENS_FILE": str(token_path), "CHECKPOINT": args.checkpoint, "DSV41_ORACLE_LOGITS_OUT": str(logits_path)})
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
            logits_bytes = logits_path.read_bytes() if logits_path.exists() else b""
            rec["historical_run"] = {
                "returncode": proc.returncode,
                "elapsed_s": time.perf_counter() - start,
                "stdout_tail": proc.stdout[-4000:],
                "pass_lifecycle": proc.returncode == 0 and "PASS: 3 token IDs -> encoder 0..19 -> decoder 20..39 -> logits" in proc.stdout,
            }
            if logits_bytes:
                import numpy as np
                arr = np.frombuffer(logits_bytes, dtype=np.float32)
                rec["oracle_logits_digest"] = hashlib.sha256(logits_bytes).hexdigest()
                rec["oracle_logits"] = {
                    "shape": [1, int(arr.size)],
                    "dtype": "float32",
                    "bytes": len(logits_bytes),
                    "argmax": int(arr.argmax()),
                    "max": float(arr.max()),
                    "anchors": {str(i): float(arr[i]) for i in [0, 1, 15, 266, 11992] if i < arr.size},
                    "export_source": "DSV41_ORACLE_LOGITS_OUT opt-in instrumentation in historical tests/attention/test_text_backbone.cpp",
                }
                rec["digest_gate_complete"] = True
                rec["match"] = rec["oracle_logits_digest"] == rec["production_digest_to_qualify"]
            # Extract run dir from first line: Logs: artifacts/text-backbone/run-...
            for line in proc.stdout.splitlines():
                if line.startswith("Logs: "):
                    rec["historical_run"]["artifact_dir"] = str(hist / line.split("Logs: ", 1)[1])
                    break
            out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
            print(out)
            return 0 if proc.returncode == 0 else 1
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if rec.get("historical_run", {}).get("returncode", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
