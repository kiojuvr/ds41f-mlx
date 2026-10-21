"""Request-owned Recipe 0.2 semantics for the ordinary native MTP horizon.

Only canonical single-token input spends parser/generator state. Preview forks
that same native session and the official response accumulator, including IDs.
Argument completion certifies a call, never a generation/application terminal.
"""
from collections import deque
from dataclasses import dataclass
from functools import wraps
from time import perf_counter_ns
from uuid import uuid4
import json


@dataclass(frozen=True)
class ConsumingEOFProof:
    processor: object
    lifetime: str
    revision: int
    ordinal: int
    frontier: int
    native_frontier: int
    candidate_ids: tuple[int, ...]
    response_snapshot: str
    prefix_responses: tuple[str, ...]
    chunks: tuple[tuple[str, ...], ...]
    completed: tuple[tuple[tuple[int, int], ...], ...]
    terminals: tuple[tuple[bool, bool], ...]
    outcome: str | None


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


def fail_closed(method):
    @wraps(method)
    def operation(self, *args, **kwargs):
        if self.failed:
            raise RuntimeError('recipe semantic owner failed closed')
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.failed = True
            self._eof_proof = None
            self.call_certificates = ()
            self.tool_complete = False
            self.processor.close()
            raise
    return operation


class RecipeSemanticGuard:
    def __init__(self, processor, *, diagnostic=False, control_token_ids=(), response=None, frontier=0):
        import deepseek_recipe as d
        if not isinstance(processor, d.ChatCompletionSemanticSession):
            raise RuntimeError('Recipe 0.2 native semantic session required')
        self._processor = processor
        self.response = response or d.ChatCompletionResponse('guard', 'guard', 0, 0, 0)
        self.lifetime = uuid4().hex
        self.frontier = frontier
        self.control_token_ids = frozenset(map(int, control_token_ids))
        self.diagnostic = diagnostic
        self.emitted = 0
        self.finished = self.failed = self.tool_complete = False
        self._block_complete = False
        self._eof_proof = None
        self.call_certificates = ()
        self.events = []
        self.calls = self.candidate_ids = 0
        self.latencies_ns = deque(maxlen=4096)
        self.pending_sizes = deque(maxlen=4096)
        self.matches = deque(maxlen=1)
        self.preview_rows = deque(maxlen=64)

    @property
    def processor(self):
        return self._processor

    def close(self):
        """Release request state; cancellation is not a successful terminal."""
        self._eof_proof = None
        if not self.finished:
            self.failed = True
            self.call_certificates = ()
            self.tool_complete = False
        self.processor.close()

    @fail_closed
    def preview(self, candidate_ids):
        import deepseek_recipe as d
        from omlx.patches.mlx_lm_mtp.semantic_horizon import Prediction, SemanticGuardError
        if self.finished:
            raise SemanticGuardError('recipe turn already terminated')
        ids = tuple(map(int, candidate_ids))
        controls = [i for i, t in enumerate(ids) if t in self.control_token_ids]
        if controls and controls != [len(ids)-1]:
            raise SemanticGuardError('candidate sequence crosses a backend control terminal')
        recipe_ids = list(ids[:-1] if controls else ids)
        before = self.processor.status
        if before[2] or before[3]:
            raise SemanticGuardError('native owner already terminated')
        start = perf_counter_ns()
        # Prefix previews stop at the first authoritative parser terminal. The
        # binding deliberately rejects unconsumed suffixes after a user stop.
        # Each preview forks native state; none reparses canonical history.
        rows = []
        for n in range(1, len(recipe_ids)+1):
            revision, native_frontier, exact_ids, projected_rows = self.processor.preview_tokens(recipe_ids[:n])
            if (revision, native_frontier) != before[:2] or exact_ids != recipe_ids[:n]:
                raise SemanticGuardError('foreign preview revision/frontier/candidates')
            rows = projected_rows
            if rows[-1][4] or rows[-1][5]:
                break
        self.latencies_ns.append(perf_counter_ns()-start)
        self.calls += 1
        self.candidate_ids += len(ids)
        if self.processor.status != before:
            raise SemanticGuardError('native preview mutated canonical state')
        projected = self.response.fork()
        snapshots, chunks, completed, terminals = [], [], [], []
        for i, row in enumerate(rows):
            rev, native, output, calls, finished, block = row
            if (rev, native) != (before[0]+i+1, before[1]+i+1):
                raise SemanticGuardError('inexact native token attribution')
            chunks.append(tuple(c.to_json() for c in output))
            for chunk in output:
                projected.accumulate(chunk)
            snapshots.append(projected.to_json())
            completed.append(tuple(calls))
            terminals.append((finished, block))
        terminal = bool(rows and (rows[-1][4] or rows[-1][5]))
        outcome = None
        if terminal:
            if rows[-1][5]:
                stop = d.InferenceChunk.finish(d.InferenceFinishReason.Stop)
                rev, native, exact_ids, stop_rows = self.processor.preview_tokens(recipe_ids[:len(rows)], finish=stop)
                if (rev, native) != before[:2] or exact_ids != recipe_ids[:len(rows)]:
                    raise SemanticGuardError('foreign terminal projection')
                for chunk in stop_rows[-1][2]:
                    projected.accumulate(chunk)
            outcome = projected.to_json()
        proof = ConsumingEOFProof(self.processor, self.lifetime, before[0], self.emitted,
            self.frontier+self.emitted, before[1], ids, self.response.to_json(),
            tuple(snapshots), tuple(chunks), tuple(completed), tuple(terminals), outcome)
        self._eof_proof = proof
        if terminal:
            index = len(rows)-1
            kind = 'CONSUMING_TOOL_EOF' if rows[-1][5] else 'RECIPE_STOP'
            return Prediction(kind, index, index, ids[index], (kind, proof))
        return None

    @fail_closed
    def observe_canonical_emit(self, token_id, expected):
        import deepseek_recipe as d
        from omlx.patches.mlx_lm_mtp.semantic_horizon import SemanticGuardError
        if self.finished:
            raise SemanticGuardError('canonical feed after recipe terminal')
        token_id = int(token_id)
        if token_id in self.control_token_ids:
            if expected is not None:
                raise SemanticGuardError('recipe cannot own suppressed backend control')
            proof = self._eof_proof
            if proof is not None:
                offset = self.emitted-proof.ordinal
                prior = proof.prefix_responses[-1] if proof.prefix_responses else proof.response_snapshot
                if (proof.processor is not self.processor or proof.lifetime != self.lifetime or
                        offset != len(proof.candidate_ids)-1 or proof.candidate_ids[-1] != token_id or
                        self.processor.status != (proof.revision+offset, proof.native_frontier+offset, False, False) or
                        self.response.to_json() != prior):
                    raise SemanticGuardError('stale/foreign/divergent backend control candidate')
            self._eof_proof = None
            self.emitted += 1
            return None
        proof = self._eof_proof
        if proof is None:
            # Canonical drain may have no outstanding speculative proof. Obtain
            # one from this SAME native owner before consuming, not from history.
            prediction = self.preview([token_id])
            if prediction is not None or expected is not None:
                raise SemanticGuardError('unowned terminal in canonical drain')
            proof = self._eof_proof
        offset = self.emitted-proof.ordinal
        prior = proof.response_snapshot if offset == 0 else (
            proof.prefix_responses[offset-1] if 0 < offset < len(proof.prefix_responses) else None)
        if (proof.processor is not self.processor or proof.lifetime != self.lifetime or
                proof.frontier != self.frontier+proof.ordinal or
                self.processor.status[:2] != (proof.revision+offset, proof.native_frontier+offset) or
                not 0 <= offset < len(proof.prefix_responses) or
                proof.candidate_ids[offset] != token_id or self.response.to_json() != prior):
            raise SemanticGuardError('stale/foreign/divergent semantic proof')
        is_terminal = any(proof.terminals[offset])
        terminal_kind = 'CONSUMING_TOOL_EOF' if proof.terminals[offset][1] else 'RECIPE_STOP'
        if is_terminal != (expected is not None) or (expected is not None and
                (expected[0] != terminal_kind or expected[1] is not proof)):
            raise SemanticGuardError('unowned or foreign semantic terminal')
        row = self._push(d.InferenceChunk.tokens([token_id]))
        if (self.response.to_json() != proof.prefix_responses[offset] or
                tuple(row[3]) != proof.completed[offset] or
                (row[4], row[5]) != proof.terminals[offset] or
                self._last_chunks != proof.chunks[offset]):
            raise SemanticGuardError('canonical response/chunks/actions diverged from preview')
        self.emitted += 1
        self._certify_calls(row[3], proof, offset)
        self._block_complete |= row[5]
        if is_terminal:
            if row[5]:
                self._push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop))
            if self.response.to_json() != proof.outcome:
                raise SemanticGuardError('official terminal response diverged')
            self._check_final_calls()
            self.tool_complete = self._block_complete and bool(self.call_certificates)
            self.finished = True
            self.matches.append(dict(ordinal=self.emitted-1, token_id=token_id,
                                     identity=expected[0], consuming_revision=proof.revision,
                                     consuming_frontier=self.frontier+self.emitted))
            self._eof_proof = None
            return 'stop'
        if offset == len(proof.candidate_ids)-1:
            self._eof_proof = None
        return None

    def _certify_calls(self, completed, proof, offset):
        from omlx.patches.mlx_lm_mtp.semantic_horizon import SemanticGuardError
        data = json.loads(self.response.to_json())
        for index, _source_end in completed:
            choices = data.get('choices', [])
            if len(choices) != 1:
                raise SemanticGuardError('call completion lacks unique official choice')
            choice = choices[0]
            calls = choice.get('message', {}).get('tool_calls', [])
            if index != len(self.call_certificates) or index >= len(calls):
                raise SemanticGuardError('noncanonical call completion index')
            self.call_certificates += (CallCertificate(self.processor, self.lifetime,
                proof.revision+offset+1, self.frontier+self.emitted, choice['index'], index,
                json.dumps(calls[index], sort_keys=True), proof.candidate_ids[:offset+1],
                proof.response_snapshot, self.response.to_json()),)

    def _check_final_calls(self):
        from omlx.patches.mlx_lm_mtp.semantic_horizon import SemanticGuardError
        calls = [(c['index'], i, json.dumps(call, sort_keys=True))
                 for c in json.loads(self.response.to_json()).get('choices', [])
                 for i, call in enumerate(c.get('message', {}).get('tool_calls', []))]
        if calls != [(c.choice, c.index, c.call) for c in self.call_certificates]:
            raise SemanticGuardError('final response contains changed/uncertified calls')

    @fail_closed
    def _push(self, chunk):
        row = self.processor.push(chunk)
        return self._accumulate(row)

    def _accumulate(self, row):
        self._last_chunks = tuple(c.to_json() for c in row[2])
        for event in row[2]:
            self.response.accumulate_copy(event)
        self.events.extend(row[2])
        return row

    @fail_closed
    def finish_eof(self):
        """Actual exhausted backend, not cancellation and not backend Stop."""
        if not self.finished:
            self._accumulate(self.processor.finish())
            self.finished = True
            self.tool_complete = False
            self._eof_proof = None

    @fail_closed
    def finish_backend(self, reason='stop'):
        import deepseek_recipe as d
        if self.finished:
            return
        if reason not in ('stop', 'length'):
            raise RuntimeError('unsupported backend settlement reason')
        self._push(d.InferenceChunk.finish(
            d.InferenceFinishReason.Length if reason == 'length' else d.InferenceFinishReason.Stop))
        # Stop can end an incomplete block. Do not turn completed arguments or
        # a generator's tool_calls finish reason into a full-block certificate.
        if self._block_complete:
            self._check_final_calls()
        self.tool_complete = reason == 'stop' and self._block_complete and bool(self.call_certificates)
        self.finished = True
        self._eof_proof = None
