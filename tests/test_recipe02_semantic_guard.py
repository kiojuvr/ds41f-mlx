"""Real patched Recipe 0.2 extension; controlled tokens, no model claims."""
import json
from pathlib import Path
import re
import sys

import deepseek_recipe as d
import pytest

from ds41f_mlx.runtime.recipe_semantic_guard import RecipeSemanticGuard

RECIPE = Path(sys.prefix) / 'share/ds41f-mtp/recipe'
TOKENIZER = RECIPE / 'static/tokenizers/v41/tokenizer.json'
PREFIX = '<｜DSML｜tool_calls><｜DSML｜invoke name="f"><｜DSML｜parameter name="x" string="true">'
ARG_END = '</｜DSML｜parameter></｜DSML｜'
TAIL = 'invoke></｜DSML｜tool_calls>'


def owner(stop_sequences=None):
    tok = d.Tokenizer.from_file(str(TOKENIZER))
    converted = d.ChatCompletionRequest({'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': 'hi'}], 'reasoning_effort': 'none',
        'tools': [{'type': 'function', 'function': {'name': 'f'}},
                  {'type': 'function', 'function': {'name': 'g'}}]}).convert(d.ConversionOptions())
    generator = d.ChatCompletionRequest.chunk_generator(converted, 'response', 'model')
    options = converted.parsing_options
    if stop_sequences is not None:
        options.stop_sequences = stop_sequences
    session = d.ChatCompletionSemanticSession(generator, options, tok)
    guard = RecipeSemanticGuard(session, frontier=17)
    guard._push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(prompt_tokens=17)))
    return tok, guard


def encode(tok, text):
    added = {t['content']: t['id'] for t in json.loads(TOKENIZER.read_text())['added_tokens']}
    pattern = '(' + '|'.join(re.escape(s) for s in sorted(added, key=len, reverse=True)) + ')'
    return [i for p in re.split(pattern, text) if p for i in ([added[p]] if p in added else tok.encode(p))]


def feed(guard, ids):
    prediction = guard.preview(ids)
    for n, token in enumerate(ids):
        expected = prediction.identity if prediction is not None and n == prediction.index else None
        reason = guard.observe_canonical_emit(token, expected)
        if reason:
            return n + 1
    return len(ids)


@pytest.mark.parametrize('multiple', [False, True])
def test_arguments_are_not_turn_terminal_and_exact_stream(multiple):
    tok, guard = owner()
    ids = encode(tok, PREFIX + 'Tokyo🤭' + ARG_END)
    before = guard.processor.status
    assert guard.preview(ids) is None
    assert guard.preview(ids) is None
    assert guard.processor.status == before
    feed(guard, ids)
    assert len(guard.call_certificates) == 1
    original = guard.call_certificates[0].call
    assert not guard.finished and not guard.tool_complete
    assert guard.preview(encode(tok, 'inv')) is None
    feed(guard, encode(tok, 'invoke>'))
    assert not guard.finished
    if multiple:
        feed(guard, encode(tok, '<｜DSML｜invoke name="g"></｜DSML｜invoke>'))
        assert len(guard.call_certificates) == 2
        assert guard.call_certificates[0].call == original
    # Closing marker fragments cannot terminate until the final parser byte.
    feed(guard, encode(tok, '</｜DSML｜tool_calls'))
    assert not guard.finished
    assert feed(guard, encode(tok, '>after')) == 1
    assert guard.finished and guard.tool_complete
    body = json.loads(guard.response.to_json())
    assert body['choices'][0]['finish_reason'] == 'tool_calls'
    assert guard.call_certificates[0].call == original
    accumulated = d.ChatCompletionResponse('guard', 'guard', 0, 0, 0)
    for event in guard.events:
        accumulated.accumulate(event)
    assert accumulated.to_json() == guard.response.to_json()
    guard.processor.close()


def test_rejected_candidates_rollback_and_ids():
    tok, guard = owner()
    before = guard.processor.status, guard.response.to_json()
    rejected = encode(tok, '<｜DSML｜tool_calls><｜DSML｜invoke name="reject"></｜DSML｜invoke></｜DSML｜tool_calls>')
    a = guard.preview(rejected)
    b = guard.preview(rejected)
    assert a.index == b.index and a.identity[1].outcome == b.identity[1].outcome
    assert (guard.processor.status, guard.response.to_json()) == before
    ids = encode(tok, PREFIX + 'ok' + ARG_END + TAIL)
    prediction = guard.preview(ids)
    outcome = prediction.identity[1].outcome
    # New candidate supersedes rejected proofs without spending canonical RNG.
    feed(guard, ids)
    assert guard.response.to_json() == outcome


@pytest.mark.parametrize('fault', ['foreign', 'stale', 'divergent', 'response'])
def test_proofs_fail_closed(fault):
    tok, guard = owner()
    ids = encode(tok, PREFIX + 'ok' + ARG_END + TAIL)
    pred = guard.preview(ids)
    if fault == 'foreign':
        _, foreign = owner()
        other = foreign.preview(ids)
        for token in ids[:pred.index]:
            guard.observe_canonical_emit(token, None)
        action = lambda: guard.observe_canonical_emit(ids[pred.index], other.identity)
    elif fault == 'stale':
        guard.processor.push(d.InferenceChunk.tokens([ids[0]]))
        action = lambda: guard.observe_canonical_emit(ids[0], None)
    elif fault == 'response':
        guard.response = d.ChatCompletionResponse('divergent', 'guard', 0, 0, 0)
        action = lambda: guard.observe_canonical_emit(ids[0], None)
    else:
        action = lambda: guard.observe_canonical_emit(tok.encode('different')[0], None)
    with pytest.raises(RuntimeError):
        action()
    assert guard.failed and guard.processor.status[3]
    assert not guard.call_certificates and not guard.tool_complete


@pytest.mark.parametrize('reason', ['stop', 'length'])
def test_backend_finish_not_full_block(reason):
    tok, guard = owner()
    feed(guard, encode(tok, PREFIX + 'ok' + ARG_END))
    assert len(guard.call_certificates) == 1
    guard.finish_backend(reason)
    assert guard.finished and not guard.tool_complete
    assert json.loads(guard.response.to_json())['choices'][0]['finish_reason'] == (
        'tool_calls' if reason == 'stop' else 'length')
    guard.processor.close()


def test_cancel_and_incomplete_marker_cleanup():
    tok, guard = owner()
    feed(guard, encode(tok, PREFIX + 'partial</'))
    assert not guard.finished and not guard.call_certificates
    guard.close()
    assert guard.processor.status[3] and guard._eof_proof is None
    with pytest.raises(RuntimeError):
        guard.preview(tok.encode('next'))
    assert guard.failed


def test_user_stop_owns_exact_token_prefix():
    tok, guard = owner(['HALT'])
    feed(guard, tok.encode('answer HA'))
    assert not guard.finished
    ids = tok.encode('LT after')
    pred = guard.preview(ids)
    assert pred is not None and pred.kind == 'RECIPE_STOP'
    consumed = feed(guard, ids)
    assert consumed == pred.index+1 and consumed < len(ids)
    assert guard.finished and not guard.tool_complete
    assert json.loads(guard.response.to_json())['choices'][0]['message']['content'] == 'answer '
    guard.close()


def test_backend_control_suppression_and_candidate_identity():
    tok, guard = owner()
    eos = tok.encode(d.SpecialToken.Eos.default_str())[0]
    guard.control_token_ids = frozenset([eos])
    ids = tok.encode('hello') + [eos]
    assert guard.preview(ids) is None
    feed(guard, ids)
    assert guard.emitted == len(ids)
    assert guard.processor.status[1] == len(ids)-1
    guard.finish_backend('stop')
    assert guard.finished and not guard.tool_complete
    assert json.loads(guard.response.to_json())['choices'][0]['message']['content'] == 'hello'
    guard.close()


def test_actual_eof_is_not_backend_stop_tool_completion():
    tok, guard = owner()
    feed(guard, encode(tok, PREFIX + 'ok' + ARG_END))
    assert len(guard.call_certificates) == 1
    guard.finish_eof()
    assert guard.finished and not guard.tool_complete
    assert json.loads(guard.response.to_json())['choices'][0]['finish_reason'] == 'stop'
    guard.close()


@pytest.mark.parametrize('backend_eof', [False, True])
def test_settlement_drains_before_eof_and_cancel_does_not_invent_finish(backend_eof):
    # Actual Recipe/MLX adapter, controlled physical-quiescence metadata only.
    # This is not evidence of numerical/native cache execution.
    from types import SimpleNamespace as NS
    from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend, QualificationSession
    tok, guard = owner()
    initial = tok.encode('first')
    feed(guard, initial)
    pending = tok.encode(' after')
    guard.preview(pending)
    rec = QualificationSession('controlled')
    rec.guard, rec.processor = guard, guard.processor
    canonical = [1] * 17 + initial
    generated = list(initial)
    size = len(canonical)+len(pending)
    target = [NS(size=lambda: size) for _ in range(40)]
    rings = [NS(offset=size) for _ in range(3)]
    def quiesce():
        assert not guard.finished
        feed(guard, pending)
        canonical.extend(pending)
        generated.extend(pending)
        return NS(target_cache=target, dspark_context=NS(caches=rings), canonical_tokens=canonical,
                  counters=NS(require_m28_zeroes=lambda: None), to_json=lambda: {})
    rec.owner = NS(_bg=NS(_generation_batch=NS()), quiesce=quiesce, close=lambda: None,
        history=NS(canonical_generated_tokens=generated, canonical_frontier=size), connection=NS(lifetime='physical', revision=1,
        disposition='settled', consumed_tokens=canonical, queue_ahead=(), pending_prediction=None, rng_draws=0))
    backend = NS(_qualify_settled_protocol=InternalMTPQualificationBackend._qualify_settled_protocol,
                 _finalize_application_certificate=lambda *args: None)
    trace = {'backend_eof': backend_eof}
    InternalMTPQualificationBackend._settle(backend, rec, trace)
    assert trace['response']['choices'][0]['message']['content'] == 'first after'
    assert trace['response']['choices'][0]['finish_reason'] == ('stop' if backend_eof else None)
    assert guard.finished == backend_eof
    assert guard.failed != backend_eof
    assert not guard.tool_complete and guard.processor.status[3]
    assert guard._eof_proof is None


def test_partial_accept_then_new_candidate_preserves_call_rng():
    tok, guard = owner()
    rejected = encode(tok, PREFIX + 'rejected' + ARG_END + TAIL)
    guard.preview(rejected)
    prefix = encode(tok, PREFIX)
    for token in prefix:
        guard.observe_canonical_emit(token, None)
    remaining = encode(tok, 'accepted' + ARG_END + TAIL)
    proof = guard.preview(remaining).identity[1]
    feed(guard, remaining)
    assert guard.response.to_json() == proof.outcome
    assert json.loads(guard.call_certificates[0].call)['function']['arguments'] == '{"x": "accepted"}'
    guard.close()
