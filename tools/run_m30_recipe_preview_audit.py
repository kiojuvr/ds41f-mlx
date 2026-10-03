#!/usr/bin/env python3
"""Blocked-gate diagnostics, NOT a speculative preview or runtime adapter.

Uses the pinned Rust state machine directly (rustc path module, no grammar copy)
and the existing release Python binding. Never loads or forwards the model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = "8cadfede7063c896b944e7bae05daa3549ae97ea"
DEFAULT_RECIPE = Path("/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def tool(name="lookup_weather", value="Paris", string=True):
    return (f'<｜DSML｜ invoke name="{name}">\n'
            f'<｜DSML｜ parameter name="city" string="{str(string).lower()}">{value}'
            '</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n')


def make_processor(d, tokenizer, stops=(), reasoning=False):
    options = d.ParsingOptions()
    options.stop_sequences = list(stops)
    if reasoning:
        options.reasoning_initial_stage = d.ReasoningStage.Reasoning
    gen = d.ChatCompletionChunkGenerator("m30", "deepseek-v4.1-flash", False, reasoning)
    return d.StreamProcessor(gen, options, tokenizer), d.ChatCompletionResponse("m30", "deepseek-v4.1-flash", 0, 0, 0)


def parse_tokens(d, tokenizer, tokens, stops=(), reasoning=False):
    p, response = make_processor(d, tokenizer, stops, reasoning)
    events = []
    finished_at = None
    start = time.perf_counter_ns()
    for index, token in enumerate(tokens):
        for out in p.push(d.InferenceChunk.token(int(token))):
            events.append(json.loads(out.to_json()))
            response.append(out)
        if p.finished:
            finished_at = index
            break
    before_finish = p.finished
    if not p.finished:
        for out in p.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
            events.append(json.loads(out.to_json()))
            response.append(out)
    elapsed = time.perf_counter_ns() - start
    data = json.loads(response.to_json())
    p.close()
    return {"response": data, "events": events, "finished_before_backend_finish": before_finish,
            "finished_candidate_index": finished_at, "diagnostic_parse_ns": elapsed}


def decoder_trace(tok, tokens):
    """Diagnostic observation of decoder.rs; not a serving implementation.

    Rust StreamDecoder decodes pending IDs, skip_special_tokens=false, holds
    empty/U+FFFD-ending output and clears the pending IDs on a successful flush.
    This trace is cross-checked against the Python parser, not used for a clamp.
    """
    pending, trace = [], []
    for index, token in enumerate(tokens):
        pending.append(int(token))
        text = tok.decode(pending, skip_special_tokens=False)
        held = not text or text.endswith("\ufffd")
        trace.append({"candidate_index": index, "pending_ids": list(pending),
                      "decoded": text, "held": held,
                      "parser_input": None if held else text})
        if not held:
            pending.clear()
    return trace


def source_probe(recipe, cases):
    """Compile the unmodified pinned state machine directly; no dependency fork."""
    source = recipe / "deepseek-recipe/src/stream/state_machine.rs"
    with tempfile.TemporaryDirectory(prefix="m30-source-probe-") as directory:
        path = Path(directory)
        wrapper = path / "main.rs"
        wrapper.write_text(
            '#[path = ' + json.dumps(str(source)) + '] mod authority;\n'
            'use std::io::{self, Read};\n'
            'fn main() {\n'
            ' let mut input = String::new(); io::stdin().read_to_string(&mut input).unwrap();\n'
            ' let mut options = authority::ParsingOptions::default();\n'
            ' options.stop_sequences = std::env::args().skip(1).collect();\n'
            ' let mut parser = authority::StateMachine::new(options);\n'
            ' let mut offset = 0;\n'
            ' for ch in input.chars() { let text = ch.to_string();\n'
            '  offset += text.len();\n'
            '  for action in parser.feed(&text) { println!("{} {:?}", offset, action); }\n'
            ' }\n'
            '}\n')
        subprocess.run(["rustc", "--edition=2024", "-Awarnings", str(wrapper), "-o", str(path / "probe")], check=True)
        results = []
        for name, text, stops in cases:
            run = subprocess.run([str(path / "probe"), *stops], input=text, text=True, capture_output=True, check=True)
            results.append({"name": name, "input": text, "stops": list(stops),
                            "actions": run.stdout.splitlines()})
        return results


def audit(recipe, out):
    import deepseek_recipe as d
    import deepseek_recipe._native as native
    from tokenizers import Tokenizer

    revision = git(recipe, "rev-parse", "HEAD")
    if revision != PIN or git(recipe, "status", "--porcelain"):
        raise RuntimeError("audit requires exact clean pinned recipe source")
    tokenizer_path = recipe / "static/tokenizers/v41/tokenizer.json"
    tokenizer = d.Tokenizer.from_file(str(tokenizer_path))
    raw_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    out.mkdir(parents=True, exist_ok=True)

    p, _ = make_processor(d, tokenizer)
    copies = {}
    for name, operation in [("copy", copy.copy), ("deepcopy", copy.deepcopy), ("pickle", pickle.dumps)]:
        try:
            operation(p)
            copies[name] = {"supported": True}
        except Exception as exc:
            copies[name] = {"supported": False, "error": f"{type(exc).__name__}: {exc}"}
    p.close()

    block = '<｜DSML｜ calls>\n' + tool() + '</｜DSML｜ calls>'
    multi = '<｜DSML｜ calls>\n' + tool() + tool("store", '{"count":2,"ok":true}', False) + '</｜DSML｜ calls>'
    cases = [
        ("ordinary", "Ordinary café 🙂 text.", (), False),
        ("one_call", block, (), False),
        ("multiple_calls", multi, (), False),
        ("thinking_to_dsml", "Let me check.\n</think>\n" + block, (), True),
        ("invoke_closed_block_unclosed", '<｜DSML｜ calls>\n' + tool(), (), False),
        ("stop_one_token", "abcSTOPtail", ("STOP",), False),
        ("stop_multiple_tokens_unicode", "café🙂alpha HALT NOW omega", ("HALT NOW",), False),
        ("stop_shared_prefix", "HALT NOT yet HALT NOW tail", ("HALT NOW",), False),
    ]
    diagnostics = []
    for name, text, stops, reasoning in cases:
        tokens = tokenizer.encode(text)
        trace = decoder_trace(raw_tokenizer, tokens)
        parsed = parse_tokens(d, tokenizer, tokens, stops, reasoning)
        diagnostics.append({"name": name, "origin": "synthetic parser diagnostic; NOT model qualification",
                            "input": text, "tokens": tokens, "stop_sequences": list(stops),
                            "decoder_trace": trace, **parsed,
                            "target_frontier": None, "dspark_frontier": None,
                            "m_recipe_semantic": None, "commit_clamp": "unavailable"})
    (out / "parser-diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n")

    # Actual historical model tokens: preserve provenance, do NOT label fresh M30.
    historical_path = ROOT / "artifacts/m11/tool-boundary-qualification.json"
    historical = json.loads(historical_path.read_text())
    turn = historical["flows"][0]["turn0"]
    tokens = turn["generated_tokens"]
    reference = {"origin": "historical actual MTP-OFF M11 model generation, NOT fresh M30",
                 "source": str(historical_path.relative_to(ROOT)), "source_sha256": sha(historical_path),
                 "source_recipe_revision": historical["recipe_revision"],
                 "tokens": tokens, "original_turn": turn,
                 "decoded_pending_stream": decoder_trace(raw_tokenizer, tokens),
                 "m30_parser_reparse": parse_tokens(d, tokenizer, tokens)}
    (out / "off-reference.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")

    actions = source_probe(recipe, [("complete_dsml_then_tail", block + "TAIL", ()),
                                   ("incomplete_block_then_tail", '<｜DSML｜ calls>\n' + tool() + "TAIL", ()),
                                   ("unicode_stop_then_tail", "café🙂HALT NOWTAIL", ("HALT NOW",))])
    (out / "authoritative-source-actions.json").write_text(json.dumps(actions, indent=2, ensure_ascii=False) + "\n")

    sys.path.insert(0, str(Path.home() / "omlx-0.7.0.release"))
    import omlx.api.tool_calling as tool_calling  # real jsonschema-dependent import
    files = ["deepseek-recipe/src/stream/state_machine.rs", "deepseek-recipe/src/stream/decoder.rs",
             "deepseek-recipe/src/stream/processor.rs", "deepseek-recipe-python/src/response.rs"]
    packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}
    identity = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                "packages": packages, "recipe_revision": revision,
                "recipe_worktree_clean": True, "source_hashes": {f: sha(recipe / f) for f in files},
                "tokenizer_sha256": sha(tokenizer_path), "native_module": native.__file__,
                "native_sha256": sha(native.__file__), "native_version": d.__version__,
                "native_origin": "byte-for-byte copy of existing release environment distribution; not rebuilt in M30",
                "native_build_revision_independently_attested": False,
                "tool_calling_import": tool_calling.__file__, "tool_calling_import_ok": True,
                "jsonschema_required_by": "pinned oMLX pyproject.toml jsonschema>=4.0.0",
                "recipe_patch": None, "ds41f_dependency_changes": []}
    (out / "runtime-identities.json").write_text(json.dumps(identity, indent=2) + "\n")
    record = {"schema": "ds41f.m30.recipe-preview-audit.v1", "base_commit": "e19eb817c33e848237758332c552e52e3461a9f4",
              "decision": "RECIPE_PREVIEW_API_REQUIRED", "protocol_gate_passed": False,
              "production_mtp": "OFF", "copy_attempts": copies,
              "python_public_parser_api": [name for name in dir(d.StreamProcessor) if not name.startswith("_")],
              "preview_implemented": False, "semantic_clamp_installed": False,
              "model_executions": 0, "model_history_replay": 0, "full_target_cache_repack": 0,
              "verify_during_quiesce": 0, "proposal_during_quiesce": 0,
              "zero_counter_scope": "this audit did not execute model or quiescence; NOT live-MTP proof",
              "parser_history_replays": 1, "parser_history_replay_scope": "one historical OFF artifact diagnostic, not per cycle",
              "preview_cycle_overhead_ns": None, "tool_result_reentry": "not run: protocol gate blocked",
              "fresh_actual_model_artifacts": [], "off_on_parity": "not qualified",
              "diagnostic_cases": [case[0] for case in cases],
              "required_live_cases": {name: {"status": "BLOCKED_PREVIEW_API", "target_frontier": None,
                  "dspark_frontier": None, "parser_frontier": None, "acceptance_topology": None}
                  for name in ["ordinary_completion", "EOS", "max_tokens", "one_token_stop",
                      "multi_token_stop", "stop_cross_verify_cycles", "one_valid_DSML_call",
                      "multiple_calls_one_DSML_block", "DSML_close_cross_candidate_blocks",
                      "thinking_to_DSML", "full_acceptance", "partial_acceptance", "rejection",
                      "semantic_boundary_before_rejection", "rejection_before_semantic_boundary",
                      "interruption_before_tool_completion", "interruption_safe_undelivered_prefix",
                      "quiescence_after_tool_completion", "tool_result_reentry"]}}
    (out / "qualification.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/m30")
    args = parser.parse_args()
    print(json.dumps(audit(args.recipe, args.out), indent=2))
