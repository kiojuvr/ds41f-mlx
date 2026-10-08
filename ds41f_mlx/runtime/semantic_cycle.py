"""Worker-confined recipe authorization and whole-cycle adoption (M54).

No target/sampler implementation here. Native preview is the semantic oracle;
M52/M51 remain the sole generation/settlement authorities. Tools additionally
use the native consuming-state EOF fork and unchanged M11 protocol predicates.
No exception fallback to ordinary generation.
"""
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class SemanticAuthorization:
    owner: object
    frontier: int
    ordinal: int
    anchor: int
    drafts: tuple[int, ...]
    snapshot: object
    terminal_identity: tuple | None
    eof_proofs: tuple = ()


class SemanticCycleAdapter:
    def __init__(self, m8, processor, *, parse_tool_calls=False, producer=None, eof_preview=None):
        if parse_tool_calls and eof_preview is None:
            raise RuntimeError('EOF-sensitive tool preview is not qualified for speculative authorization')
        self.eof_preview = eof_preview
        self._eof_proofs = []
        self.m8, self.processor = m8, processor
        self.generation = m8.generation
        self.ordinal = 0
        self.active = False
        self.failed = False
        self.producer = producer
        self.metrics = dict(cycles=0, offered=0, accepted=0, consumed=0,
                            proposal_s=0., authorization_s=0., execution_s=0., report_s=0.,
                            eof_preview_s=0., eof_consuming_probe_s=0.,
                            protected_steps=0, planned_target_inputs=0)

    def _preview(self, ids):
        before = self.processor.semantic_snapshot()
        t0 = perf_counter()
        result = self.processor.preview_tokens(list(ids))
        self.metrics['authorization_s'] += perf_counter() - t0
        if before != self.processor.semantic_snapshot() or not result.mapping_exact:
            raise RuntimeError('recipe preview is mutating or inexact')
        if result.terminal_kind is None and result.safe_token_count != len(ids):
            raise RuntimeError('recipe preview omitted canonical provenance')
        return result

    def _safe(self, ids):
        # Backend control IDs are suppressed, not protocol text. Do not preview
        # printable EOS spellings or permit interior controls in a verify span.
        controls = set(self.generation.stop_token_ids)
        end = next((i for i, t in enumerate(ids) if t in controls), len(ids))
        result = self._preview(ids[:end]) if end else None
        if result is not None and result.terminal_kind is not None:
            end = min(end, result.completing_token_index)
        identity = None if result is None or result.terminal_kind is None else (
            result.terminal_kind, result.source_start, result.source_end)
        if self.eof_preview is not None and result is not None:
            observed = end + 1 if result.terminal_kind is not None else end
            t0 = perf_counter()
            proof = self.eof_preview.preview(ids[:observed], self.ordinal)
            elapsed = perf_counter() - t0
            self.metrics['authorization_s'] += elapsed
            self.metrics['eof_preview_s'] += elapsed
            self._eof_proofs.append(proof)
            if proof.boundary == 'M11_TOOL_EOF':
                boundary = proof.observed_count - 1
                if boundary <= end:
                    end = boundary
                    identity = ('M11_TOOL_EOF', self.ordinal + boundary)
        return end, identity

    def authorize(self, cancelled=lambda: False):
        gen = self.generation
        if self.failed or self.active or gen is not self.m8.generation:
            raise RuntimeError('semantic cycle is not admissible')
        if self.producer is not None and (not self.producer.active or not self.producer.child.active):
            raise RuntimeError('retired proposal epoch is not a short-ring warm-up permission')
        if self.m8.token_history != gen.current_token_history():
            raise RuntimeError('recipe cycle starts at incoherent history')
        if (self.processor.semantic_terminal is not None or self.processor.finished
                or (self.eof_preview is not None and self.eof_preview.last_boundary is not None)):
            raise RuntimeError('canonical recipe is already terminal')
        self._eof_proofs = []
        anchor = int(gen._pending.item())
        snapshot = self.processor.semantic_snapshot()
        if cancelled():
            return None
        safe, terminal_identity = self._safe((anchor,))
        remaining = gen.max_tokens - len(gen.generated_tokens)
        drafts = ()
        producer = self.producer
        full = producer is not None and all(
            r.keys is not None and r.keys.shape[2] == producer.child.config.window_size
            for r in producer.rings)
        # Terminal anchor, short ring and length-final input are explicit
        # protected canonical steps, never exception-driven OFF fallback.
        if safe and full and remaining > 1:
            t0 = perf_counter()
            drafts = producer.propose()
            self.metrics['proposal_s'] += perf_counter() - t0
            candidates = (anchor,) + drafts[:remaining - 1]
            count, _ = self._safe(candidates)
            drafts = candidates[1:count]
        if cancelled():
            return None
        return SemanticAuthorization(gen, gen.token_frontier, self.ordinal,
                                     anchor, tuple(drafts), snapshot, terminal_identity, tuple(self._eof_proofs))

    def advance(self, observe, *, cancelled=lambda: False, _fault=lambda phase: None):
        """One protected operation; publish no external effects in observe.

        Returns settled consumed reports only. M8 adopts ALL reports before
        canonical recipe feed; the worker does not yield between these phases.
        Any post-mutation report failure burns the lease and forbids re-entry.
        """
        try:
            permission = self.authorize(cancelled)
            if permission is None:
                return []
            _fault('authorized')
            if cancelled():
                return []
            gen = self.generation
            if (permission.owner is not gen or permission.frontier != gen.token_frontier
                    or permission.ordinal != self.ordinal
                    or permission.anchor != int(gen._pending.item())
                    or permission.snapshot != self.processor.semantic_snapshot()):
                raise RuntimeError('stale semantic authorization')
            for proof in permission.eof_proofs:
                self.eof_preview.validate(proof, proof.candidate_ids, self.ordinal)
        except BaseException:
            self.failed = True
            if self.producer is not None:
                self.producer.retire()
            raise
        self.active = True
        mutated = False
        try:
            t0 = perf_counter()
            if permission.drafts:
                self.metrics['planned_target_inputs'] += 1 + len(permission.drafts)
                result = gen.speculative_cycle(permission.drafts, cancelled=cancelled)
                reports = result['reports']
                self.metrics['cycles'] += 1
                self.metrics['offered'] += len(permission.drafts)
                self.metrics['accepted'] += result['proposal_acceptance_count']
            else:
                report = gen.next_token()
                reports = [] if report is None else [report]
                self.metrics['planned_target_inputs'] += len(reports)
                self.metrics['protected_steps'] += 1
            self.metrics['execution_s'] += perf_counter() - t0
            mutated = bool(reports)
            _fault('settled')
            self.m8.adopt_cycle_reports(reports)
            _fault('adopted')
            t0 = perf_counter()
            for report in reports:
                observe(report)
                actual = self.processor.semantic_terminal
                identity = None if actual is None else (actual.kind, actual.source_start, actual.source_end)
                if self.eof_preview is not None and actual is None and not self.processor.finished:
                    t_eof = perf_counter()
                    complete = self.eof_preview.completed(self.ordinal + 1)
                    self.metrics['eof_consuming_probe_s'] += perf_counter() - t_eof
                    if complete:
                        identity = ('M11_TOOL_EOF', self.ordinal)
                expected = permission.terminal_identity if len(reports) == 1 else None
                if identity != expected:
                    raise RuntimeError('canonical recipe observation disagrees with authorization')
                if identity is not None and identity[0] == 'M11_TOOL_EOF':
                    proof = permission.eof_proofs[0]
                    self.metrics['tool_eof_agreement'] = dict(
                        authorization_revision=proof.revision,
                        canonical_ordinal_before=proof.ordinal,
                        candidate_ids=list(proof.candidate_ids),
                        observed_prefix_count=proof.observed_count,
                        includes_eof=proof.includes_eof, boundary=proof.boundary,
                        consuming_revision=self.processor.preview_revision,
                        completion_ordinal=self.ordinal, target_frontier=gen.token_frontier,
                        agreement=True)
                self.ordinal += 1
                _fault('observed')
            self.metrics['report_s'] += perf_counter() - t0
            self.metrics['consumed'] += len(reports)
            _fault('reported')
            if not gen._stopped and self.producer is not None and reports:
                self.producer.advance(gen.take_tap_receipt())
            return reports
        except BaseException:
            self.failed = True
            # A target failure is already burned by M52. A recipe/adoption or
            # derived advance failure cannot make its committed output retryable.
            if mutated or gen._failed:
                gen._invalidate()
                self.m8.closed = True
            if self.producer is not None:
                self.producer.retire()
            raise
        finally:
            self.active = False
