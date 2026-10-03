"""M36 fail-closed protocol settlement regression; no model fixture claims."""
from types import SimpleNamespace

from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend as Backend, QualificationSession


def fixture(finished, arguments):
    rec = SimpleNamespace(guard=SimpleNamespace(finished=finished), poisoned=False)
    trace = {'terminal_matches': [{'identity': ['DSML_TOOL_CALL_BLOCK_END', 0, 1]}] if finished else [],
             'response': {'choices': [{'message': {'tool_calls': [
        {'function': {'name': 'lookup_weather', 'arguments': arguments}}
    ]}, 'finish_reason': 'tool_calls' if finished else None}]}}
    return rec, trace


def test_incomplete_canonical_tool_is_not_idle_recoverable():
    rec, trace = fixture(False, '{"city"')
    Backend._qualify_settled_protocol(rec, trace)
    assert rec.poisoned
    assert 'unfinished canonical tool' in trace['recovery_error']


def test_valid_json_without_canonical_terminal_is_not_execution_permission():
    rec, trace = fixture(False, '{"city":"Berlin"}')
    Backend._qualify_settled_protocol(rec, trace)
    assert rec.poisoned  # No JSON parser substitutes for the recipe terminal.


def test_backend_finish_without_dsml_completion_is_not_execution_permission():
    rec, trace = fixture(False, '{"city":"Berlin"}')
    rec.guard.finished = True  # Length/EOS can finish the processor, not the block.
    trace['response']['choices'][0]['finish_reason'] = 'length'
    Backend._qualify_settled_protocol(rec, trace)
    assert rec.poisoned


def test_complete_recipe_tool_remains_recoverable():
    rec, trace = fixture(True, '{"city":"Berlin"}')
    Backend._qualify_settled_protocol(rec, trace)
    assert not rec.poisoned
    assert 'recovery_error' not in trace


def test_poisoned_cleanup_remains_busy_until_lease_release():
    rec = QualificationSession('one', busy=True, poisoned=True)
    assert rec.to_json()['state'] == 'busy'
    rec.busy = False
    assert rec.to_json()['state'] == 'poisoned'


def test_text_transport_prefix_is_not_partial_tool_protocol():
    rec = SimpleNamespace(guard=SimpleNamespace(finished=False), poisoned=False)
    trace = {'response': {'choices': [{'message': {'content': 'café 🙂 '}}]}}
    Backend._qualify_settled_protocol(rec, trace)
    assert not rec.poisoned
