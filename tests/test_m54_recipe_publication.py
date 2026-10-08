"""Shared JSON/SSE publication regression; real recipe, no checkpoint generation."""
import json
from types import SimpleNamespace
from uuid import uuid4
import pytest

from test_m54_native_eof_preview import fixture
from ds41f_mlx.runtime.tool_boundary_session import M11AssistantTurn, _extract_tool_calls, _make_processor_and_response
from ds41f_mlx.serving.server import prepare_request
from tools.run_m11_tool_boundary_qualification import initial_body
from ds41f_mlx.serving.deepseek_recipe_backend import StatefulSessionRecord
from ds41f_mlx.serving.recipe_publication import publish_recipe_turn
from ds41f_mlx.serving.response_reservation import reserve


def test_shared_publication_preserves_standard_effect_certificate_and_identity():
    cfg, d, tokenizer, cursor = fixture()
    cursor.processor.close()
    payload = initial_body('Paris'); payload['max_tokens'] = 64
    body = json.dumps(payload).encode()
    prepared = prepare_request('chat_completions',body,tokenizer=tokenizer,recipe_path=cfg.recipe_path)
    p, response = _make_processor_and_response(d,prepared,'m54-publish',cfg.model_id,tokenizer)
    cursor = SimpleNamespace(processor=p,response=response,prepared=prepared)
    for chunk in p.push(d.InferenceChunk.ready()):
        response.append(chunk)
    # Recorded first official-checkpoint M54 native EOF turn (28 inputs).
    # Only protocol reconstruction is replayed here, never target execution.
    ids = [30,128825,34756,2329,1281,12533,1425,65,50219,3816,30,128825,10767,2329,1281,
           37399,4,3418,1281,11476,3320,51119,1718,128825,10767,1018,1718,128825]
    try:
        for token in ids:
            for chunk in cursor.processor.push(d.InferenceChunk.token(token)):
                cursor.response.append(chunk)
        for chunk in cursor.processor.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
            cursor.response.append(chunk)
        response = json.loads(cursor.response.to_json())
        history = cursor.prepared.token_ids + ids
        turn = M11AssistantTurn(finish_reason='tool_calls', generated_tokens=tuple(ids),
            frontier_after_commit=len(history), tool_calls=tuple(_extract_tool_calls('chat_completions',response,[])),
            response_json=response)
        certificates = []
        for media in ('application/json', 'text/event-stream'):
            rec = StatefulSessionRecord(session_id=str(uuid4()))
            closed = []
            rec.m11 = SimpleNamespace(m8=SimpleNamespace(token_history=history,close=lambda:closed.append(True)))
            slot = reserve(rec,1,body,media)
            request_id = str(uuid4()); slot.request_id = request_id
            record = publish_recipe_turn(SimpleNamespace(recipe_path=cfg.recipe_path,checkpoint=cfg.checkpoint_path),
                rec,cursor.prepared,tokenizer,body,None,turn,request_id=request_id)
            assert rec.request_count == 1 and rec.last_turn is record
            assert record['request_id'] == slot.request_id
            assert record['reconstruction']['executable_tools']
            assert record['reconstruction']['representable'] and not closed
            assert slot.state == 'active' and not slot.chunks  # publication precedes serialization/effects
            certificates.append(record['reconstruction'])
        assert certificates[0] == certificates[1]
    finally:
        cursor.processor.close()


def test_missing_original_body_cannot_publish_tool_certificate():
    rec = StatefulSessionRecord(session_id='s')
    turn = SimpleNamespace(to_json=lambda:{}, tool_calls=[object()], finish_reason='tool_calls')
    with pytest.raises(RuntimeError, match='original request body'):
        publish_recipe_turn(None,rec,SimpleNamespace(capacity=None),None,None,None,turn,request_id='r')
    assert rec.request_count == 0 and rec.last_turn is None
