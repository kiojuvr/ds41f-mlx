"""Internal M36R official-recipe reconstruction, not a second parser."""
from copy import deepcopy
import hashlib
import json


def reconstruction_certificate(body, response, canonical, *, tokenizer, recipe_path,
                               completed_tool_block=False, options=None, checkpoint=None):
    from deepseek_recipe import ConversionError
    from .server import prepare_request, RequestError
    choices = response.get('choices', [])
    result = dict(version=1, frontier=len(canonical),
                  canonical_sha256=hashlib.sha256(json.dumps(canonical).encode()).hexdigest(),
                  representable=False, executable_tools=False)
    if len(choices) != 1:
        return dict(result, error='requires exactly one authoritative choice')
    message = deepcopy(choices[0]['message'])
    calls = message.get('tool_calls', [])
    complete = not calls or (completed_tool_block and choices[0].get('finish_reason') == 'tool_calls')
    witness = deepcopy(body)
    witness['stream'] = False
    witness.pop('stream_options', None)
    witness['max_tokens'] = 1
    witness['messages'].append(message)
    for call in calls:
        witness['messages'].append(dict(role='tool', tool_call_id=call['id'],
                                        content='NOT EXECUTED: reconstruction witness'))
    witness['messages'].append(dict(role='user', content='Continue.'))
    if witness.get('tools'):
        witness['tool_choice'] = 'auto'
    try:
        prepared = prepare_request('chat_completions', json.dumps(witness).encode(),
                                   tokenizer=tokenizer, recipe_path=recipe_path, options=options, checkpoint=checkpoint)
        ids = prepared.token_ids
        exact = len(ids) > len(canonical) and ids[:len(canonical)] == canonical
        result.update(exact_prefix=exact, semantic_complete=complete,
                      first_mismatch=next((i for i in range(len(canonical))
                                           if i >= len(ids) or ids[i] != canonical[i]), None),
                      witness=witness, encoded_length=len(ids), representable=exact and complete,
                      executable_tools=bool(calls) and exact and complete)
    except (ConversionError, RequestError, ValueError) as exc:
        result.update(error=str(exc), semantic_complete=complete)
    return result


class LocalToolLedger:
    """One living local client. Reserve before effect; no crash recovery claim."""
    def __init__(self):
        self.entries = {}

    def execute(self, session_id, sequence, outcome, execute):
        if outcome.get('outcome_state') != 'recoverable' or outcome.get('sequence') != sequence:
            raise ValueError('requires settled recoverable outcome with matching request identity')
        certificate = outcome['certificate']
        if not certificate['representable'] or not certificate['executable_tools']:
            raise ValueError('no executable canonical tool certificate')
        calls = outcome['response']['choices'][0]['message']['tool_calls']
        results = []
        for index, call in enumerate(calls):
            key = (session_id, sequence, index, call['id'])
            if key not in self.entries:
                self.entries[key] = None  # a failed/ambiguous effect is never retried
                self.entries[key] = execute(call)
            if self.entries[key] is None:
                raise RuntimeError('tool effect ambiguous; local automatic retry forbidden')
            results.append(dict(role='tool', tool_call_id=call['id'], content=self.entries[key]))
        return results
