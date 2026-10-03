"""Internal opt-in recipe adapter for the isolated oMLX semantic horizon.

Never selected by public serving or MTP-OFF. Canonical protocol input is fed exactly once per emitted non-control token;
preview is the pinned Rust fork, never history reparsing.
"""
from collections import deque
from time import perf_counter_ns


class RecipeSemanticGuard:
    def __init__(self, processor, *, diagnostic=False, control_token_ids=()):
        self.processor = processor
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
        self.events.extend(self.processor.push(d.InferenceChunk.token(int(token_id))))
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
            self.events.extend(self.processor.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)))
        self.finished = True
        return 'stop'

    def finish_backend(self, reason='stop'):
        import deepseek_recipe as d
        if not self.processor.finished:
            self.events.extend(self.processor.push(d.InferenceChunk.finish(
                d.InferenceFinishReason.Length if reason == 'length' else d.InferenceFinishReason.Stop)))
        self.finished = True
