#!/usr/bin/env python3
"""Run the first integrated DwarfStar-derived prefill executor slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ds41f_mlx.dwarfstar_prefill_slice import DwarfStarPrefillVerticalSliceExecutor

DEFAULT_CHECKPOINT = "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DS41F_CHECKPOINT", DEFAULT_CHECKPOINT))
    ap.add_argument("--native-out-dir", default="artifacts/m2/dwarfstar-prefill/native")
    ap.add_argument("--out", default="artifacts/m2/dwarfstar-prefill/vertical-slice-layer0-27-candidate20-indexrefresh24.json")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--tokens", default="0,3")
    args = ap.parse_args()

    tokens = [int(x) for x in args.tokens.split(",") if x]
    executor = DwarfStarPrefillVerticalSliceExecutor.with_compiled_native(Path(args.checkpoint), Path(args.native_out_dir))
    result = executor.run(tokens=tokens, layers=args.layers)

    def clean(value):
        if isinstance(value, np.ndarray):
            if value.size <= 256:
                return value.tolist()
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean(result.artifact), indent=2, sort_keys=True) + "\n")
    print(out)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
