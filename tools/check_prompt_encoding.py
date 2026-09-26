#!/usr/bin/env python3
"""Record M0 prompt-encoding fixtures using the local known-good oMLX patch.

This does not load model weights. It checks tokenizer/config/processor behavior,
including the DeepSeek-V4.1 literal image-token parser fix.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("DS41F_PROMPT_FIXTURE_OUT", "artifacts/m0/prompt-encoding")) / time.strftime("fixture-%Y%m%d-%H%M%S.json")
PYTHON = os.environ.get("DS41F_PYTHON", str(Path.home() / ".venvs" / "omlx-0.7.0.dev2" / "bin" / "python"))
OMLX = os.environ.get("DS41F_OMLX", str(Path.home() / "omlx-0.7.0.dev2"))
CHECKPOINT = os.environ.get("DS41F_CHECKPOINT", "/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash")

PROBE = r'''
import hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["DS41F_OMLX"])
from transformers import PreTrainedTokenizerFast
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.processing import Processor, format_messages

checkpoint = Path(os.environ["DS41F_CHECKPOINT"])
raw = json.loads((checkpoint / "config.json").read_text())
config = ModelConfig.from_dict(raw)
tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
processor = Processor(tokenizer, config)
image_token = config.image_token_id
cases = [
    ("simple", [{"role":"user","content":"ping"}], 0),
    ("literal_image_token", [{"role":"user","content":"literal <｜deepseek_image｜> text"}], 0),
    ("structured_image_placeholder", [{"role":"user","content":[{"type":"text","text":"see"},{"type":"image","url":"omlx-prepared-image"}]}], 1),
]
records = []
for name, messages, image_count in cases:
    formatted, ranges = format_messages(messages, image_count)
    text = processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
    ids = processor.apply_chat_template(formatted, tokenize=True, add_generation_prompt=True)
    records.append({
        "name": name,
        "formatted": formatted,
        "image_ranges": ranges,
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_count": len(ids),
        "image_token_count": ids.count(image_token),
        "first_tokens": ids[:32],
        "last_tokens": ids[-32:],
    })
print(json.dumps({
    "ok": True,
    "checkpoint": str(checkpoint),
    "image_token_id": image_token,
    "cases": records,
}, ensure_ascii=False, sort_keys=True))
'''


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DS41F_OMLX"] = OMLX
    env["DS41F_CHECKPOINT"] = CHECKPOINT
    proc = subprocess.run([PYTHON, "-c", PROBE], env=env, text=True, capture_output=True)
    record = {
        "schema": "ds41f.m0.prompt-encoding-fixture.v1",
        "python": PYTHON,
        "omlx": OMLX,
        "checkpoint": CHECKPOINT,
        "returncode": proc.returncode,
        "stderr": proc.stderr,
    }
    if proc.stdout.strip():
        try:
            record["result"] = json.loads(proc.stdout)
        except Exception:
            record["stdout_raw"] = proc.stdout
    else:
        record["stdout_raw"] = proc.stdout
    OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(OUT)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
