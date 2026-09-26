#!/usr/bin/env python3
"""Compare M0 direct oMLX smoke with API server smoke for the same request.

This is an oMLX compatibility comparison, not official DeepSeek correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("direct")
    ap.add_argument("server")
    ap.add_argument("--out", default="artifacts/m0/oracle-compare/direct-vs-server.json")
    args = ap.parse_args()

    direct = json.loads(Path(args.direct).read_text())
    server = json.loads(Path(args.server).read_text())
    body = server.get("body", {})
    runtime = body.get("ds41f_runtime", {})
    direct_prompt_hash = hashlib.sha256(json.dumps(direct.get("prompt_ids", []), separators=(",", ":")).encode()).hexdigest()
    checks = [
        {"name": "direct_ok", "pass": direct.get("ok") is True},
        {"name": "server_status_200", "pass": server.get("status") == 200},
        {"name": "generated_ids", "pass": direct.get("generated_ids") == runtime.get("generated_ids"), "direct": direct.get("generated_ids"), "server": runtime.get("generated_ids")},
        {"name": "text", "pass": direct.get("text") == body.get("choices", [{}])[0].get("message", {}).get("content"), "direct": direct.get("text"), "server": body.get("choices", [{}])[0].get("message", {}).get("content")},
        {"name": "prompt_ids_sha256", "pass": direct_prompt_hash == runtime.get("prompt_ids_sha256"), "direct": direct_prompt_hash, "server": runtime.get("prompt_ids_sha256")},
        {"name": "usage_completion_tokens", "pass": len(direct.get("generated_ids", [])) == body.get("usage", {}).get("completion_tokens")},
    ]
    result = {
        "schema": "ds41f.m0.direct-server-omlx-compatibility.v2",
        "classification": "omlx_compatibility_reference_not_official_qualification",
        "passed": all(c["pass"] for c in checks),
        "direct": args.direct,
        "server": args.server,
        "checks": checks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
