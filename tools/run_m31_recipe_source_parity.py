#!/usr/bin/env python3
"""Compare canonical Rust parsing against the exact pinned base, no model load."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PIN = "8cadfede7063c896b944e7bae05daa3549ae97ea"


def fixture_cases():
    from tools.run_m30_recipe_preview_audit import tool
    single = '<｜DSML｜ calls>\n' + tool() + '</｜DSML｜ calls>'
    multiple = '<｜DSML｜ calls>\n' + tool() + tool("store", '{"count":2,"ok":true}', False) + '</｜DSML｜ calls>'
    cases = [
        ("ordinary", "Ordinary café 🙂 text.", {}),
        ("reasoning", "Consider café🙂.\n</think>\nAnswer.", {"reasoning": True}),
        ("one_call", single, {}),
        ("multiple_calls_string_and_json", multiple, {}),
        ("thinking_to_tool", "Think.\n</think>\n" + single, {"reasoning": True}),
        ("json_fence", "before ```json\n{\"ok\":true}\n``` after", {"json": True}),
        ("json_raw", '[1,true,"🙂"]', {"json": True}),
        ("json_stop", '```json\n{"text":"HALT NOW tail"}\n```', {"json": True, "stops": ["HALT NOW"]}),
        ("backend_stop", "text", {}),
        ("backend_length", "text", {"finish": "length"}),
        ("backend_eof", "text", {"finish": "eof"}),
        ("eos_literal_backend_stop", "text<｜end▁of▁sentence｜>", {}),
        ("one_token_stop", "abcSTOPtail", {"stops": ["STOP"]}),
        ("multi_token_stop", "alpha HALT NOW omega", {"stops": ["HALT NOW"]}),
        ("shared_prefix_stop", "HALT NOT yet HALT NOWtail", {"stops": ["HALT NOW", "HALT NEVER"]}),
        ("unicode_stop", "café🙂HALT NOWtail", {"stops": ["HALT NOW"]}),
        ("unicode_stop_itself", "café🙂tail", {"stops": ["🙂"]}),
        ("stop_in_reasoning_excluded", "STOP thought</think>answer STOP tail", {"reasoning": True, "stops": ["STOP"]}),
        ("incomplete_dsml", '<｜DSML｜ calls>\n' + tool(), {}),
        ("complete_dsml_invalid_tail", single + "INVALID TAIL", {}),
        ("multi_tool_length", multiple, {"finish": "length"}),
    ]
    result = [{"name": name, "text": text, "stops": [], **options} for name, text, options in cases]
    historical = json.loads((ROOT / "artifacts/m11/tool-boundary-qualification.json").read_text())
    result.append({"name": "historical_actual_m11_off_tokens", "tokens": historical["flows"][0]["turn0"]["generated_tokens"], "stops": [],
                   "origin": "historical, not fresh M31 generation"})
    return result


def run_driver(recipe, fixtures, target, driver="m31_recipe_parity_driver.rs", release=False):
    with tempfile.TemporaryDirectory(prefix="m31-parity-driver-") as directory:
        path = Path(directory)
        source = ROOT / "tools" / driver
        manifest = ('[package]\nname="m31-parity-driver"\nversion="0.0.0"\nedition="2024"\n'
                    '[dependencies]\n'
                    f'deepseek-recipe={{path={json.dumps(str(recipe / "deepseek-recipe"))}}}\n'
                    'serde_json="=1.0.151"\ntokenizers="=0.23.2"\ntokio-stream="=0.1.19"\n'
                    'tokio={version="=1.53.1",features=["macros","rt-multi-thread"]}\n'
                    '[[bin]]\nname="m31-parity-driver"\n'
                    f'path={json.dumps(str(source))}\n')
        (path / "Cargo.toml").write_text(manifest)
        env = dict(os.environ, CARGO_TARGET_DIR=str(target))
        run = subprocess.run(["cargo", "run", "--offline", "--quiet", *(["--release"] if release else []), "--manifest-path", str(path / "Cargo.toml"), "--",
                              str(recipe / "static/tokenizers/v41/tokenizer.json")],
                             input=json.dumps(fixtures), text=True, capture_output=True, env=env)
        if run.returncode:
            raise RuntimeError(run.stderr)
        return json.loads(run.stdout)


def main(base, patched, out, target):
    revision = subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise RuntimeError("wrong parity authority")
    fixtures = fixture_cases()
    baseline = run_driver(base, fixtures, target)
    candidate = run_driver(patched, fixtures, target)
    out.mkdir(parents=True, exist_ok=True)
    for name, record in [("parity-fixtures", fixtures), ("canonical-base", baseline), ("canonical-patched", candidate)]:
        (out / f"{name}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    comparisons = [{"name": a["name"], "input_mode": a["input_mode"], "equal": a == b} for a, b in zip(baseline, candidate)]
    result = {"schema": "ds41f.m31.canonical-source-parity.v1", "base_revision": revision,
              "scope": "Rust raw OutputChunk and chat protocol events; NOT Python native binding or live model",
              "clock_normalization": "chat event created=0; all other fields compared exactly",
              "comparisons": comparisons, "case_count": len(fixtures), "comparison_count": len(comparisons),
              "equal": len(baseline) == len(candidate) and baseline == candidate,
              "driver_sha256": hashlib.sha256((ROOT / "tools/m31_recipe_parity_driver.rs").read_bytes()).hexdigest()}
    (out / "canonical-parity.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["equal"]:
        mapping = run_driver(patched, fixtures, target, driver="m31_recipe_preview_driver.rs", release=True)
        (out / "source-preview-mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
    if not result["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/tmp/ds41f-m31-base"))
    p.add_argument("--patched", type=Path, default=Path("/tmp/ds41f-m31-recipe"))
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/m31")
    p.add_argument("--target", type=Path, default=Path("/tmp/ds41f-m31-cargo"))
    args = p.parse_args()
    main(args.base, args.patched, args.out, args.target)
