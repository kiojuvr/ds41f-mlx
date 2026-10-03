"""Parser diagnostics only. These tests do not certify an MTP protocol gate."""
import copy
import json
from pathlib import Path

import pytest

from tools.run_m30_recipe_preview_audit import (
    DEFAULT_RECIPE, decoder_trace, make_processor, parse_tokens, tool,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recipe():
    d = pytest.importorskip("deepseek_recipe")
    from tokenizers import Tokenizer
    path = DEFAULT_RECIPE / "static/tokenizers/v41/tokenizer.json"
    if not path.exists():
        pytest.skip("pinned tokenizer not installed")
    return d, d.Tokenizer.from_file(str(path)), Tokenizer.from_file(str(path))


def parse_text_chunks(d, tokenizer, chunks, stops=(), reasoning=False):
    p, response = make_processor(d, tokenizer, stops, reasoning)
    for text in chunks:
        if p.finished:
            break
        for out in p.push(d.InferenceChunk.text(text, 1)):
            response.append(out)
    if not p.finished:
        for out in p.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
            response.append(out)
    p.close()
    return json.loads(response.to_json())["choices"][0]


@pytest.mark.parametrize("text,stops,reasoning", [
    ("café🙂alpha HALT NOW omega", ("HALT NOW",), False),
    ("HALT NOT yet HALT NOWtail", ("HALT NOW",), False),
    ('<｜DSML｜ calls>\n' + tool() + '</｜DSML｜ calls>', (), False),
    ('<｜DSML｜ calls>\n' + tool() + tool("store", '[1,true]', False) + '</｜DSML｜ calls>', (), False),
    ('reason\n</think>\n<｜DSML｜ calls>\n' + tool() + '</｜DSML｜ calls>', (), True),
    ("Ordinary café 🙂 text.", (), False),
])
def test_pending_decoder_trace_matches_token_parser(recipe, text, stops, reasoning):
    d, tokenizer, raw = recipe
    tokens = tokenizer.encode(text)
    trace = decoder_trace(raw, tokens)
    # Feed each exact successful decoder flush, retaining special markers.
    chunks = [entry["parser_input"] for entry in trace if not entry["held"]]
    reference = parse_tokens(d, tokenizer, tokens, stops, reasoning)["response"]["choices"][0]
    assert parse_text_chunks(d, tokenizer, chunks, stops, reasoning) == reference
    # Character chunks exercise marker/stop prefixes held across push boundaries.
    assert parse_text_chunks(d, tokenizer, list(text), stops, reasoning) == reference


def test_utf8_pending_is_not_independent_token_decode(recipe):
    d, tokenizer, raw = recipe
    tokens = tokenizer.encode("🙂")
    trace = decoder_trace(raw, tokens)
    assert len(tokens) > 1
    assert any(entry["held"] for entry in trace)
    assert "".join(entry["parser_input"] or "" for entry in trace) == "🙂"
    assert "".join(raw.decode([t], skip_special_tokens=False) for t in tokens) != "🙂"


def test_dsml_close_is_not_python_stream_finished(recipe):
    d, tokenizer, _ = recipe
    text = '<｜DSML｜ calls>\n' + tool() + '</｜DSML｜ calls>'
    result = parse_tokens(d, tokenizer, tokenizer.encode(text))
    assert not result["finished_before_backend_finish"]
    assert result["response"]["choices"][0]["finish_reason"] == "tool_calls"


def test_eof_tool_probe_does_not_prove_closed_dsml(recipe):
    d, tokenizer, _ = recipe
    text = '<｜DSML｜ calls>\n' + tool()
    result = parse_tokens(d, tokenizer, tokenizer.encode(text))
    assert '</｜DSML｜ calls>' not in text
    assert result["response"]["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(result["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}


def test_canonical_parser_copy_fails_without_mutating_it(recipe):
    d, tokenizer, _ = recipe
    p, response = make_processor(d, tokenizer)
    for out in p.push(d.InferenceChunk.text("safe", 1)):
        response.append(out)
    with pytest.raises(TypeError):
        copy.copy(p)
    for out in p.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
        response.append(out)
    assert json.loads(response.to_json())["choices"][0]["message"]["content"] == "safe"
    p.close()


def test_source_actions_distinguish_closed_block_not_eof_heuristic():
    records = json.loads((ROOT / "artifacts/m30/authoritative-source-actions.json").read_text())
    complete, incomplete, stop = records
    assert any("ContentAfterFinished" in action for action in complete["actions"])
    assert not any("ContentAfterFinished" in action for action in incomplete["actions"])
    assert any("StopSequence" in action for action in stop["actions"])


def test_blocked_evidence_does_not_claim_live_qualification():
    record = json.loads((ROOT / "artifacts/m30/qualification.json").read_text())
    assert record["decision"] == "RECIPE_PREVIEW_API_REQUIRED"
    assert record["production_mtp"] == "OFF"
    assert not record["protocol_gate_passed"]
    assert not record["semantic_clamp_installed"]
    assert record["preview_cycle_overhead_ns"] is None
    assert record["fresh_actual_model_artifacts"] == []
