"""Worker-confined recipe delivery cursor over the existing M11/M8 authority.

No queue, executor, cache owner or generation engine lives here. advance() consumes
at most one complete target transaction. Its returned events are a delivery batch,
not historical diagnostics. The caller must not advance until that batch is sent.
"""
from __future__ import annotations

import json
from uuid import uuid4

from .tool_boundary_session import (
    M11AssistantTurn, _recipe_module, _make_processor_and_response,
    _extract_tool_calls, _protocol_finish_reason, _recent_tool_arguments_look_complete,
    _probe_tool_calls_complete,
)


class LiveRecipeTurn:
    def __init__(self, session, prepared, on_commit, *, cancelled=None):
        self.session = session
        self.prepared = prepared
        self.on_commit = on_commit
        self._cancelled = cancelled or (lambda: False)
        self.recipe = _recipe_module()
        self.response_id = str(uuid4())
        self.processor, self.response = _make_processor_and_response(
            self.recipe, prepared, self.response_id, session.model_id, session.tokenizer)
        self.events = []  # diagnostics only, max 64
        self.generated = []
        self.turn = None
        self.closed = False
        self.cancelled = False
        self.ready = True
        self.cycle_adapter = None
        self.eof_preview = None
        if (prepared.protocol == 'chat_completions'
                and prepared.conversation_request.parsing_options.parse_tool_calls
                and hasattr(self.processor, 'preview_eof_tokens')):
            from .tool_eof_preview import ToolEOFPreview
            self.eof_preview = ToolEOFPreview(self)
        if getattr(session, 'execution_strategy', 'off') == 'first-party-mtp-development':
            from .semantic_cycle import SemanticCycleAdapter
            from .dspark_proposal import DSparkProposalProducer
            gen = session.m8.generation
            seed = getattr(gen, '_prefill_tap_receipt', None)
            producer = DSparkProposalProducer(gen, seed)
            gen._prefill_tap_receipt = None
            self.cycle_adapter = SemanticCycleAdapter(
                session.m8, self.processor,
                parse_tool_calls=prepared.conversation_request.parsing_options.parse_tool_calls,
                producer=producer, eof_preview=self.eof_preview)

    def record(self, outputs):
        batch = []
        for output in outputs:
            # Serialize before transferring native recipe chunk ownership.
            data = json.loads(output.to_json())
            batch.append(data)
            self.events.append(data)
            del self.events[:-64]
            if self.response is not None:
                self.response.append(output)
        return batch

    def advance(self):
        if self.closed:
            return []
        if self.ready:
            self.ready = False
            return self.record(self.processor.push(self.recipe.InferenceChunk.ready(
                prompt_usage=self.recipe.PromptUsage(prompt_tokens=len(self.prepared.token_ids), prompt_cache_hit_tokens=0))))
        if self.cycle_adapter is not None:
            batch = []
            def observe(report):
                self.generated.append(int(report.token))
                suppressed = report.finish_reason == 'stop' and int(report.token) in set(self.prepared.stop_token_ids)
                if not suppressed:
                    batch.extend(self.record(self.processor.push(self.recipe.InferenceChunk.token(int(report.token)))))
            reports = self.cycle_adapter.advance(observe, cancelled=self._cancelled)
            tool_terminal = self.eof_preview is not None and self.eof_preview.last_boundary is not None
            terminal = tool_terminal or self.processor.semantic_terminal is not None or self.processor.finished
            if not reports or terminal or reports[-1].finish_reason:
                reason = 'tool_calls' if tool_terminal else ((reports[-1].finish_reason or 'stop') if reports else 'stop')
                # A known canonical terminal wins over a concurrent transport
                # cancellation. Disconnect is not permission to rewrite a settled
                # tool/EOS/length outcome into a different recipe EOF decision.
                known_boundary = terminal or bool(reports and reports[-1].finish_reason)
                batch.extend(self.finish(cancelled=self._cancelled() and not known_boundary, reason=reason))
            return batch
        report = self.session.m8.next_token()
        if report is None:
            return self.finish()
        self.generated.append(int(report.token))
        suppressed = report.finish_reason == 'stop' and int(report.token) in set(self.prepared.stop_token_ids)
        batch = [] if suppressed else self.record(self.processor.push(self.recipe.InferenceChunk.token(int(report.token))))
        tool_terminal = (self.eof_preview.completed(len(self.generated)) if self.eof_preview is not None
                         else (self.prepared.conversation_request.parsing_options.parse_tool_calls
                               and _recent_tool_arguments_look_complete(self.prepared.protocol, self.events)
                               and _probe_tool_calls_complete(self.recipe, self.prepared, self.response_id,
                                                             self.session.model_id, self.session.tokenizer, self.generated)))
        if tool_terminal or report.finish_reason:
            # Terminal publication/idle transfer precedes visible terminal events.
            batch.extend(self.finish(reason='tool_calls' if tool_terminal else report.finish_reason))
        return batch

    def abort(self):
        """Protected failure cleanup; never parse/complete or retry the turn."""
        if not self.closed:
            self.closed = True
            self.processor.close()

    def finish(self, *, cancelled=False, reason='stop'):
        if self.closed:
            return []
        self.cancelled = cancelled
        self.closed = True  # never retry settlement/parse on failure
        try:
            if cancelled:
                self.session.m8.cancel_turn('http_cancelled')
            else:
                self.session.m8.ensure_idle('http_protocol_boundary')
            recipe_reason = self.recipe.InferenceFinishReason.Length if cancelled or reason == 'length' else self.recipe.InferenceFinishReason.Stop
            batch = self.record(self.processor.push(self.recipe.InferenceChunk.finish(recipe_reason)))
            batch.extend(self.record(self.processor.finish()))
            response = None if self.response is None else json.loads(self.response.to_json())
            self.turn = M11AssistantTurn(
                finish_reason=_protocol_finish_reason(self.prepared.protocol, response, self.events) or reason,
                generated_tokens=tuple(self.generated), frontier_after_commit=self.session.m8.frontier,
                tool_calls=tuple(_extract_tool_calls(self.prepared.protocol, response, self.events)),
                response_json=response, stream_events=tuple(self.events),
                prompt_replay_count=self.session.m8.total_prompt_replay_count,
                full_cache_repack_count=self.session.m8.total_full_cache_repack_count)
            if self.cycle_adapter is not None:
                self.session.last_cycle_metrics = dict(self.cycle_adapter.metrics)
            self.on_commit(self.turn, cancelled)
            # This retains diagnostics, never the live delivery source.
            self.session.boundary_records.append(self.turn.to_json())
            del self.session.boundary_records[:-16]
            return batch
        finally:
            self.processor.close()
