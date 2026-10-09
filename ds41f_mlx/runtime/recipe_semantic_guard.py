"""Internal opt-in recipe adapter for the isolated oMLX semantic horizon.

Never selected by public serving or MTP-OFF. Canonical protocol input is fed exactly once per emitted non-control token;
preview is the pinned Rust fork, never history reparsing.
"""
from collections import deque
from time import perf_counter_ns
from dataclasses import dataclass
import json
from uuid import uuid4


@dataclass(frozen=True)
class ConsumingEOFProof:
    processor: object
    lifetime: str
    revision: int
    ordinal: int
    frontier: int
    candidate_ids: tuple[int, ...]
    response_snapshot: str
    prefix_responses: tuple[str, ...]
    outcome: str
    consuming_outcomes: tuple[str | None, ...]


@dataclass(frozen=True)
class CallCertificate:
    processor: object
    lifetime: str
    revision: int
    frontier: int
    choice: int
    index: int
    call: str
    candidate_ids: tuple[int, ...]
    response_snapshot: str
    consuming_outcome: str



class RecipeSemanticGuard:
    def __init__(self, processor, *, diagnostic=False, control_token_ids=(), response=None, frontier=0):
        import deepseek_recipe as d
        if not hasattr(processor, 'preview_certified_eof_tokens'):
            raise RuntimeError('qualification consuming EOF native recipe required')
        self.processor = processor
        self.response = response or d.ChatCompletionResponse('guard', 'guard', 0, 0, 0)
        self.lifetime = uuid4().hex
        self.frontier = frontier
        self.tool_complete = False
        self._eof_proof = None
        self.call_certificates = ()
        # Backend control tokens (e.g. its proven EOS IDs) are not protocol
        # text. Arbitrary recipe stop strings NEVER use this ID-based seam.
        self.control_token_ids = frozenset(map(int, control_token_ids))
        self.diagnostic = diagnostic
        self.emitted = 0
        self.finished = False
        self.events = []
        self.calls = 0
        self.candidate_ids = 0
        self.latencies_ns = deque(maxlen=4096)
        self.pending_sizes = deque(maxlen=4096)
        self.matches = deque(maxlen=1)
        self.preview_rows = deque(maxlen=64)

    def preview(self, candidate_ids):
        from omlx.patches.mlx_lm_mtp.semantic_horizon import Prediction, SemanticGuardError
        if self.finished:
            raise SemanticGuardError('recipe turn already terminated')
        ids = list(map(int, candidate_ids))
        # Mirror the official backend's canonical EOS suppression. Printable
        # control spellings must not create user-stop predictions. Matcher
        # narrowing must make a control ID final, never an interior candidate.
        positions = [i for i, token in enumerate(ids) if token not in self.control_token_ids]
        controls = [i for i, token in enumerate(ids) if token in self.control_token_ids]
        if controls and controls != [len(ids)-1]:
            raise SemanticGuardError('candidate sequence crosses a backend control terminal')
        recipe_ids = [ids[i] for i in positions]
        before = self.processor.semantic_snapshot() if self.diagnostic else None
        start = perf_counter_ns()
        result = self.processor.preview_tokens(recipe_ids)
        self.latencies_ns.append(perf_counter_ns() - start)
        self.calls += 1
        self.candidate_ids += len(candidate_ids)
        self.pending_sizes.append(result.max_pending_ids)
        if self.diagnostic and self.processor.semantic_snapshot() != before:
            raise SemanticGuardError('native preview mutated canonical state')
        if not result.mapping_exact:
            raise SemanticGuardError('recipe preview mapping is inexact')
        if self.diagnostic:
            self.preview_rows.append(dict(ids=ids, recipe_input_ids=recipe_ids, source_bytes_before=before[3],
                                          pending_ids_before=before[1], safe=result.safe_token_count,
                                          terminal_kind=result.terminal_kind, index=result.completing_token_index,
                                          start=result.source_start, end=result.source_end))
        # Fork the actual consuming parser/stashing/generator AND accumulated
        # response. Native completion is an official ToolCallArgumentsEnd action,
        # not a spelling, raw terminal, or validity guess over argument JSON.
        revision = self.processor.preview_revision
        snapshot = self.processor.semantic_snapshot()
        response_snapshot = self.response.to_json()
        projected = self.response.fork()
        actual_revision, actual_ids, rows = self.processor.preview_certified_eof_tokens(recipe_ids)
        if (revision != actual_revision or actual_ids != recipe_ids or
                revision != self.processor.preview_revision or
                snapshot != self.processor.semantic_snapshot()):
            raise SemanticGuardError('foreign or mutating consuming EOF projection')
        prefix_responses = []
        consuming_outcomes = []
        for i, (tokens, eof, completed) in enumerate(rows):
            for chunk in tokens:
                projected.append(chunk)
            prefix_responses.append(projected.to_json())
            finished = projected.fork()
            for chunk in eof:
                finished.append(chunk)
            data = json.loads(finished.to_json())
            choices = data.get('choices', [])
            if completed and choices and all(c.get('finish_reason') == 'tool_calls' and
                    c.get('message', {}).get('tool_calls') for c in choices):
                consuming_outcomes.append(finished.to_json())
            else:
                consuming_outcomes.append(None)
        # Consuming completion certifies calls, not the assistant turn. Keep
        # the canonical parser/generator live across additional invocations.
        proof = ConsumingEOFProof(self.processor, self.lifetime, revision,
            self.emitted, self.frontier + self.emitted, tuple(ids),
            response_snapshot, tuple(prefix_responses),
            consuming_outcomes[-1] if consuming_outcomes else None,
            tuple(consuming_outcomes))
        self._eof_proof = proof
        if result.terminal_kind == 'DSML_TOOL_CALL_BLOCK_END':
            index = result.completing_token_index
            if index >= len(consuming_outcomes) or consuming_outcomes[index] is None:
                raise SemanticGuardError('raw tool terminal lacks consuming completion')
            # This recipe parser transition is a physical/full-response boundary
            # only. Tool authority was already acquired on earlier consuming
            # prefixes; closing still requires the official finished outcome.
            index = positions[index]
            return Prediction('CONSUMING_TOOL_EOF', index, index, ids[index],
                              ('CONSUMING_TOOL_EOF', proof))
        if result.terminal_kind is None:
            if result.safe_token_count != len(recipe_ids):
                raise SemanticGuardError('nonterminal preview omitted candidate provenance')
            return None
        index = positions[result.completing_token_index]
        identity = (result.terminal_kind, result.source_start, result.source_end)
        return Prediction(result.terminal_kind, index, result.safe_token_count,
                          int(ids[index]), identity)

    def observe_canonical_emit(self, token_id, expected):
        import deepseek_recipe as d
        from omlx.patches.mlx_lm_mtp.semantic_horizon import SemanticGuardError
        if self.finished:
            raise SemanticGuardError('canonical feed after recipe terminal')
        if int(token_id) in self.control_token_ids:
            if expected is not None:
                raise SemanticGuardError('recipe prediction attempted to own a suppressed backend control token')
            self.emitted += 1
            return None
        proof = self._eof_proof
        if proof is not None:
            offset = self.emitted - proof.ordinal
            prior = proof.response_snapshot if offset == 0 else proof.prefix_responses[offset-1] if 0 < offset < len(proof.prefix_responses) else None
            terminal = offset == len(proof.prefix_responses)-1
            if (proof.processor is not self.processor or proof.lifetime != self.lifetime or
                    proof.frontier != self.frontier + proof.ordinal or
                    self.processor.preview_revision != proof.revision + offset or
                    not 0 <= offset < len(proof.prefix_responses) or
                    proof.candidate_ids[offset] != int(token_id) or
                    self.response.to_json() != prior or
                    (expected is not None and expected[0] == 'CONSUMING_TOOL_EOF' and
                     (not terminal or expected[1] is not proof))):
                raise SemanticGuardError('stale/foreign consuming EOF ownership')
        elif expected is not None and expected[0] == 'CONSUMING_TOOL_EOF':
            raise SemanticGuardError('foreign consuming EOF proof')
        self._push(d.InferenceChunk.token(int(token_id)))
        if proof is not None and self.response.to_json() != proof.prefix_responses[offset]:
            raise SemanticGuardError('consuming projection diverged from canonical response')
        if proof is not None:
            outcome = proof.consuming_outcomes[offset]
            if outcome is not None:
                self._certify_calls(outcome, proof.revision + offset + 1,
                                    proof.frontier + offset + 1, proof=proof,
                                    offset=offset)
        if expected is not None and expected[0] == 'CONSUMING_TOOL_EOF':
            self._push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop))
            if self.response.to_json() != proof.outcome:
                raise SemanticGuardError('consuming EOF outcome mismatch')
            self.emitted += 1
            boundary = self.processor.semantic_terminal
            self.matches.append(dict(ordinal=self.emitted-1, token_id=int(token_id),
                identity=(boundary.kind, boundary.source_start, boundary.source_end),
                consuming_revision=proof.revision, consuming_frontier=proof.frontier + offset))
            self.tool_complete = self.finished = True
            self._eof_proof = None
            return 'stop'
        if proof is not None and terminal:
            self._eof_proof = None
        actual = self.processor.semantic_terminal
        identity = None if actual is None else (actual.kind, actual.source_start, actual.source_end)
        if identity != expected:
            raise SemanticGuardError(f'canonical recipe observation mismatch: {identity!r} != {expected!r}')
        self.emitted += 1
        if actual is None:
            return None
        self.matches.append(dict(ordinal=self.emitted-1, token_id=int(token_id), identity=identity))
        # DSML completion is NOT StreamProcessor.finished. Ordinary backend Stop
        # drives the original protocol generator to authoritative tool_calls.
        if not self.processor.finished:
            self._push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop))
        self.finished = True
        return 'stop'

    def _certify_calls(self, outcome, revision, frontier, *, proof=None, offset=0):
        from omlx.patches.mlx_lm_mtp.semantic_horizon import SemanticGuardError
        calls = [(c['index'], i, json.dumps(call, sort_keys=True))
                 for c in json.loads(outcome).get('choices', [])
                 for i, call in enumerate(c.get('message', {}).get('tool_calls', []))]
        prior = self.call_certificates
        if len(calls) < len(prior) or any(
                (cert.choice, cert.index, cert.call) != calls[i]
                for i, cert in enumerate(prior)):
            raise SemanticGuardError('certified call changed or disappeared')
        if proof is None:
            if len(calls) != len(prior):
                raise SemanticGuardError('final response contains uncertified calls')
            return
        self.call_certificates += tuple(CallCertificate(self.processor,
            self.lifetime, revision, frontier, choice, index, call,
            proof.candidate_ids[:offset+1], proof.response_snapshot, outcome)
            for choice, index, call in calls[len(prior):])

    def _push(self, chunk):
        events = self.processor.push(chunk)
        for event in events:
            self.response.append_copy(event)
        self.events.extend(events)

    def finish_backend(self, reason='stop'):
        import deepseek_recipe as d
        if not self.processor.finished:
            self._push(d.InferenceChunk.finish(
                d.InferenceFinishReason.Length if reason == 'length' else d.InferenceFinishReason.Stop))
        if self.call_certificates:
            self._certify_calls(self.response.to_json(), self.processor.preview_revision,
                                self.frontier + self.emitted)
            self.tool_complete = reason == 'stop'
        self.finished = True
