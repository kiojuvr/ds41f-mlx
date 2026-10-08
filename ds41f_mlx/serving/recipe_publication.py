"""One protected standard application publication for JSON and SSE.

The pre-existing reconstruction certificate and Web effect ledger remain the
sole re-entry/effect authorities. This function never runs tools or generation.
"""
import json
from time import time

from .recovery_certificate import reconstruction_certificate


def publish_recipe_turn(backend, rec, prepared, tokenizer, body, options, turn,
                        *, cancelled=False, request_id, application_id=None):
    record = turn.to_json()
    record.update(request_id=request_id, application_request_id=application_id, cancelled=cancelled,
                  capacity=None if prepared.capacity is None else {**prepared.capacity, 'binding': True},
                  termination_reason=('user_stop' if cancelled else 'context_capacity'
                    if turn.finish_reason == 'length' and prepared.capacity and prepared.capacity['output_mode'] == 'auto'
                    else 'output_limit' if turn.finish_reason == 'length' else turn.finish_reason))
    if cancelled or turn.tool_calls:
        if body is None:
            raise RuntimeError('canonical tool/cancellation publication requires original request body')
        certificate = reconstruction_certificate(
            json.loads(body), turn.response_json, rec.m11.m8.token_history,
            tokenizer=tokenizer, recipe_path=backend.recipe_path,
            options=options, checkpoint=backend.checkpoint,
            completed_tool_block=turn.finish_reason == 'tool_calls')
        certificate.pop('witness', None)  # no second transcript/image owner
        record['reconstruction'] = certificate
        if not certificate['representable']:
            # Settled output can be delivered/retried; no rollback/rebuild or
            # external effect is licensed by a non-representable outcome.
            rec.recovery_state = 'unrecoverable'
            rec.m11.m8.close()
        else:
            rec.recovery_state = 'ready'
    rec.last_turn = record
    rec.request_count += 1
    rec.updated_at = time()
    rec.last_error = None
    return record
