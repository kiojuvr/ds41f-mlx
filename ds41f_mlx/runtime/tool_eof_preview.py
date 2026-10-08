"""M11 EOF decision over a native fork of the *consuming* recipe state.

No tokenizer replay, grammar, protocol generator or effect authority lives here.
Only the existing M11 event/response predicates interpret native projections.
"""
from dataclasses import dataclass
import json

from .tool_boundary_session import (
    _recent_tool_arguments_look_complete, _protocol_finish_reason, _extract_tool_calls,
)


@dataclass(frozen=True)
class EOFProvenance:
    processor: object
    revision: int
    ordinal: int
    response_snapshot: str
    recent_events: tuple[str, ...]
    candidate_ids: tuple[int, ...]
    observed_count: int
    includes_eof: bool
    boundary: str


class ToolEOFPreview:
    def __init__(self, turn):
        if turn.prepared.protocol != 'chat_completions':
            raise RuntimeError('native EOF projection is qualified only for Chat Completions')
        self.turn = turn
        self.processor = turn.processor
        self.last_boundary = None

    def preview(self, ids, ordinal):
        ids = tuple(map(int, ids))
        p = self.processor
        revision = p.preview_revision
        snapshot = p.semantic_snapshot()
        if not p.preview_tokens(list(ids)).mapping_exact:
            raise RuntimeError('inexact native decoder provenance')
        response_snapshot = self.turn.response.to_json()
        recent_events = tuple(json.dumps(e, sort_keys=True, separators=(',', ':')) for e in self.turn.events[-4:])
        response = self.turn.response.fork()
        events = list(self.turn.events[-4:])
        actual_revision, actual_ids, rows = p.preview_eof_tokens(list(ids))
        if (revision != actual_revision or tuple(actual_ids) != ids
                or revision != p.preview_revision or snapshot != p.semantic_snapshot()
                or len(rows) != max(1, len(ids))):
            raise RuntimeError('inexact native EOF provenance')
        for index, (tokens, eof) in enumerate(rows):
            for chunk in tokens:
                events.append(json.loads(chunk.to_json()))
                del events[:-4]
                response.append(chunk)
            # EOF must fork the accumulated response as well as the native
            # generator/parser. Never feed EOF into the next-prefix branch.
            finished = response.fork()
            for chunk in eof:
                finished.append(chunk)
            data = json.loads(finished.to_json())
            complete = (_recent_tool_arguments_look_complete(self.turn.prepared.protocol, events)
                        and _protocol_finish_reason(self.turn.prepared.protocol, data, [])
                        in {'tool_calls', 'tool_use', 'completed'}
                        and bool(_extract_tool_calls(self.turn.prepared.protocol, data, [])))
            if complete:
                count = index + 1 if ids else 0
                return EOFProvenance(p, revision, ordinal, response_snapshot, recent_events, ids, count, True, 'M11_TOOL_EOF')
        return EOFProvenance(p, revision, ordinal, response_snapshot, recent_events, ids, len(ids), True, 'CONTINUE')

    def validate(self, proof, ids, ordinal):
        if (proof.processor is not self.processor or proof.revision != self.processor.preview_revision
                or proof.ordinal != ordinal or proof.candidate_ids != tuple(ids)
                or proof.response_snapshot != self.turn.response.to_json()
                or proof.recent_events != tuple(json.dumps(e, sort_keys=True, separators=(',', ':')) for e in self.turn.events[-4:])
                or self.processor.finished or not proof.includes_eof):
            raise RuntimeError('stale native EOF authorization')

    def completed(self, ordinal):
        proof = self.preview((), ordinal)
        if proof.boundary == 'M11_TOOL_EOF':
            self.last_boundary = proof
            return True
        return False
