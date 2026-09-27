#!/usr/bin/env python3
"""Run the first integrated DwarfStar-derived prefill executor slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.dwarfstar_prefill_slice import DwarfStarPrefillVerticalSliceExecutor

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/vertical-slice-layer0-7-source2-reuse.json")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--tokens", default="0,3")
    args = ap.parse_args()

    tokens = [int(x) for x in args.tokens.split(",") if x]
    executor = DwarfStarPrefillVerticalSliceExecutor.with_compiled_native(Path(args.checkpoint), Path(args.native_out_dir))
    result = executor.run(tokens=tokens, layers=args.layers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.artifact, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
