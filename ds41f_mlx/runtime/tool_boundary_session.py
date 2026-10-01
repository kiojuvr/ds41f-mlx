"""M11 tool-call boundary continuation over the M8/M9 live-session substrate.

This module deliberately does not parse DeepSeek DSML/tool markup itself.  It
feeds model token chunks into the pinned ``deepseek-recipe`` ``StreamProcessor``
and records protocol-visible tool identity from recipe response chunks.  The
executable cache authority remains ``M8LiveContinuationSession``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Callable, Sequence, TYPE_CHECKING
from uuid import uuid4
import json

from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession, M8TurnRecord
from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession, PRODUCTION_PREFILL_SELECTOR
from ds41f_mlx.runtime.kv_persistence import M9ArtifactInfo, save_m8_idle_state, restore_m8_idle_state
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
if TYPE_CHECKING:
    from ds41f_mlx.serving.deepseek_recipe_backend import RecipePreparedRequest


class M11ToolBoundaryError(RuntimeError):
    """Invalid M11 tool/session-boundary transition."""


@dataclass(frozen=True)
class M11ParsedToolCall:
    id: str
    name: str
    arguments: str
    protocol: str
    index: int = 0
    response_item_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class M11AssistantTurn:
    finish_reason: str
    generated_tokens: tuple[int, ...]
    frontier_after_commit: int
    tool_calls: tuple[M11ParsedToolCall, ...] = ()
    response_json: dict[str, Any] | None = None
    stream_events: tuple[dict[str, Any], ...] = ()
    prompt_replay_count: int = 0
    full_cache_repack_count: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_json(self) -> dict[str, Any]:
        return {
            "finish_reason": self.finish_reason,
            "generated_tokens": list(self.generated_tokens),
            "frontier_after_commit": self.frontier_after_commit,
            "tool_calls": [tc.to_json() for tc in self.tool_calls],
            "response_json": self.response_json,
            "stream_events": list(self.stream_events),
            "prompt_replay_count": self.prompt_replay_count,
            "full_cache_repack_count": self.full_cache_repack_count,
        }


@dataclass
class M11RecipeToolSession:
    """Single live recipe session that can cross parsed tool-call boundaries."""

    model: Any
    tokenizer: Any
    checkpoint: Path
    omlx_path: Path
    recipe_path: Path
    m8: M8LiveContinuationSession
    protocol: str
    model_id: str
    request_count: int = 0
    boundary_records: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def start_from_prepared(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        checkpoint: Path,
        omlx_path: Path,
        recipe_path: Path,
        prepared: RecipePreparedRequest,
        sampler: Callable[[Any], Any] | None = None,
        max_tokens: int | None = None,
    ) -> "M11RecipeToolSession":
        if prepared.image_sources:
            raise M11ToolBoundaryError("M11 is text/tool-only; multimodal image input is not supported")
        if len(prepared.token_ids) < 2:
            raise M11ToolBoundaryError("encoded prompt must contain prefix and held-out terminal")
        cfg = OMLXDecodeConfig(omlx_path=omlx_path, checkpoint_path=checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
        prefill = DwarfStarMLXPrefillSession(model, omlx_path=omlx_path).prefill(prepared.token_ids[:-1])
        if getattr(prefill, "production_prefill_selector", None) != PRODUCTION_PREFILL_SELECTOR:
            raise M11ToolBoundaryError("production prefill selector gate failed")
        if int(prefill.frontier) != len(prepared.token_ids) - 1:
            raise M11ToolBoundaryError("prefill frontier does not match held-out prompt prefix")
        m8 = M8LiveContinuationSession.from_prefill_result(
            model=model,
            live_result=prefill.live_result,
            terminal_prompt_token=int(prepared.token_ids[-1]),
            config=cfg,
            max_tokens=max_tokens or _max_tokens(prepared),
            sampler=sampler,
        )
        # M8 creates turn records for begin_turn_from_recipe_tokens(); the first
        # request enters through the historical M7 handoff seam, so seed the same
        # bounded accounting record for M11 diagnostics before decoding tokens.
        if not m8.turn_records:
            m8.turn_records.append(M8TurnRecord(
                turn_index=0,
                frontier_before=len(prepared.token_ids) - 1,
                prompt_suffix_tokens=len(prepared.token_ids),
                appended_prefill_tokens=len(prepared.token_ids) - 1,
                terminal_token=int(prepared.token_ids[-1]),
                generated_tokens=(),
                frontier_after=len(m8.token_history),
                prompt_replay_count=int(m8.generation.prompt_replay_count) if m8.generation is not None else 0,
                full_cache_repack_count=int(getattr(prefill, "full_cache_repack_count", 0)),
                append_seconds=float(getattr(prefill, "seconds", 0.0)),
                decode_seconds=0.0,
            ))
        return cls(
            model=model,
            tokenizer=tokenizer,
            checkpoint=Path(checkpoint),
            omlx_path=Path(omlx_path),
            recipe_path=Path(recipe_path),
            m8=m8,
            protocol=prepared.protocol,
            model_id=prepared.model or "deepseek-v4.1-flash",
        )

    @classmethod
    def restore(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        checkpoint: Path,
        omlx_path: Path,
        recipe_path: Path,
        artifact_path: Path,
        protocol: str,
        model_id: str = "deepseek-v4.1-flash",
        sampler: Callable[[Any], Any] | None = None,
    ) -> "M11RecipeToolSession":
        cache, tokens, _manifest = restore_m8_idle_state(artifact_path=artifact_path, model=model, checkpoint=checkpoint, omlx_path=omlx_path)
        cfg = OMLXDecodeConfig(omlx_path=omlx_path, checkpoint_path=checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
        m8 = M8LiveContinuationSession.from_live_cache(model=model, live_cache=cache, token_history=tokens, config=cfg, sampler=sampler)
        return cls(model=model, tokenizer=tokenizer, checkpoint=Path(checkpoint), omlx_path=Path(omlx_path), recipe_path=Path(recipe_path), m8=m8, protocol=protocol, model_id=model_id)

    def continue_from_prepared(self, prepared: RecipePreparedRequest, *, max_tokens: int | None = None) -> None:
        """Validate exact-prefix extension and start the next model turn.

        All recipe/protocol validation must have happened before this call via
        official request conversion.  Prefix mismatch rejects before M8 mutates
        the live cache.
        """
        if prepared.protocol != self.protocol:
            raise M11ToolBoundaryError(f"protocol changed from {self.protocol} to {prepared.protocol}")
        before = self.m8.diagnostics()
        try:
            self.m8.begin_turn_from_recipe_tokens(prepared.token_ids, max_tokens=max_tokens or _max_tokens(prepared))
        except Exception as exc:
            after = self.m8.diagnostics()
            if after.get("frontier") != before.get("frontier") or after.get("state") != before.get("state"):
                raise M11ToolBoundaryError("failed continuation mutated live session") from exc
            raise
        self.request_count += 1

    def run_current_assistant_turn(self, prepared: RecipePreparedRequest) -> M11AssistantTurn:
        """Decode current active GenerationBatch to its natural stop/length and commit.

        The M11 commit/cache-extraction boundary is after the GenerationBatch
        response carrying ``finish_reason`` has been consumed and its token has
        been appended to M8 ``all_tokens``, then ``ensure_idle`` extracts the
        scheduler-owned cache.  Only after that are recipe protocol chunks
        finalized.  This avoids an early parser-visible truncation with a later
        cache frontier underneath.
        """
        d = _recipe_module()
        response_id = str(uuid4())
        processor, response = _make_processor_and_response(d, prepared, response_id, self.model_id, self.tokenizer)
        events: list[dict[str, Any]] = []
        response_json: dict[str, Any] | None = None
        prompt_usage = d.PromptUsage(prompt_tokens=len(prepared.token_ids), prompt_cache_hit_tokens=0)
        for out in processor.push(d.InferenceChunk.ready(prompt_usage=prompt_usage)):
            _record_protocol_output(out, response, events)
        finish_reason = "end_of_stream"
        generated: list[int] = []
        while True:
            report = self.m8.next_token()
            if report is None:
                break
            generated.append(int(report.token))
            for out in processor.push(d.InferenceChunk.token(int(report.token))):
                _record_protocol_output(out, response, events)
            # Official-parser early boundary: if a fresh official StreamProcessor
            # over the consumed token prefix would expose a completed tool call
            # on EOF/Stop, freeze the scheduler cache here instead of allowing
            # hidden post-tool tokens to advance the cache until max_tokens.
            if (prepared.conversation_request.parsing_options.parse_tool_calls
                    and _recent_tool_arguments_look_complete(prepared.protocol, events)
                    and _probe_tool_calls_complete(d, prepared, response_id, self.model_id, self.tokenizer, generated)):
                finish_reason = "tool_calls"
                for out in processor.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
                    _record_protocol_output(out, response, events)
                break
            if report.finish_reason is not None:
                finish_reason = str(report.finish_reason)
                recipe_reason = d.InferenceFinishReason.Length if report.finish_reason == "length" else d.InferenceFinishReason.Stop
                for out in processor.push(d.InferenceChunk.finish(recipe_reason)):
                    _record_protocol_output(out, response, events)
                break
        # Freeze the executable cache exactly at the consumed GenerationBatch frontier.
        self.m8.ensure_idle("m11_tool_or_assistant_boundary")
        for out in processor.finish():
            _record_protocol_output(out, response, events)
        if response is not None:
            response_json = json.loads(response.to_json())
        tool_calls = tuple(_extract_tool_calls(prepared.protocol, response_json, events))
        turn = M11AssistantTurn(
            finish_reason=_protocol_finish_reason(prepared.protocol, response_json, events) or finish_reason,
            generated_tokens=tuple(generated),
            frontier_after_commit=self.m8.frontier,
            tool_calls=tool_calls,
            response_json=response_json,
            stream_events=tuple(events[-64:]),
            prompt_replay_count=int(self.m8.diagnostics().get("total_prompt_replay_count", 0)),
            full_cache_repack_count=int(self.m8.diagnostics().get("total_full_cache_repack_count", 0)),
        )
        self.boundary_records.append(turn.to_json())
        self.boundary_records = self.boundary_records[-16:]
        return turn

    def persist_idle(self, *, artifact_root: Path, diagnostics: dict[str, Any] | None = None) -> M9ArtifactInfo:
        self.m8.ensure_idle("m11_persist_idle")
        diag = {"m8": self.m8.diagnostics(), "m11_boundaries": self.boundary_records[-4:]}
        if diagnostics:
            diag.update(diagnostics)
        return save_m8_idle_state(artifact_root=artifact_root, model=self.model, live_cache=self.m8.live_cache, all_tokens=self.m8.token_history, checkpoint=self.checkpoint, omlx_path=self.omlx_path, diagnostics=diag)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema": "ds41f.m11.tool-boundary-session.diagnostics.v1",
            "protocol": self.protocol,
            "request_count": self.request_count,
            "m8": self.m8.diagnostics(),
            "boundaries": self.boundary_records[-16:],
        }


def _recipe_module():
    try:
        import deepseek_recipe as d  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise M11ToolBoundaryError("deepseek-recipe Python bindings are required for M11") from exc
    return d


def _max_tokens(prepared: RecipePreparedRequest) -> int:
    value = prepared.inference_options.max_tokens
    return 128 if value is None else int(value)


def _make_processor_and_response(d: Any, prepared: RecipePreparedRequest, response_id: str, model_id: str, tokenizer: Any) -> tuple[Any, Any | None]:
    model = prepared.model or model_id
    # M11/M12 always keep an accumulator response as diagnostics/session
    # metadata even for streaming HTTP.  Protocol chunks are still generated by
    # the official streaming generator and can be replayed as SSE; the accumulator
    # is not a second model-state authority.
    if prepared.protocol == "chat_completions":
        gen = d.ChatCompletionChunkGenerator(response_id, model, prepared.include_usage, prepared.conversation_request.conversation.thinking_mode)
        response = d.ChatCompletionResponse(response_id, model, int(time()), 0, 0)
    elif prepared.protocol == "responses":
        gen = d.ResponsesChunkGenerator(response_id, model).with_custom_tool_names(prepared.custom_tool_names)
        response = d.ResponsesResponse(response_id, model, int(time()), 0, 0)
    elif prepared.protocol == "messages":
        gen = d.MessagesChunkGenerator(response_id, model, prepared.conversation_request.conversation.thinking_mode)
        response = d.MessagesResponse(response_id, model, int(time()), 0, 0)
    else:
        raise M11ToolBoundaryError(f"unsupported protocol {prepared.protocol!r}")
    return d.StreamProcessor(gen, prepared.conversation_request.parsing_options, tokenizer), response


def _record_protocol_output(out: Any, response: Any | None, events: list[dict[str, Any]]) -> None:
    data = json.loads(out.to_json())
    events.append(data)
    if response is not None:
        response.append(out)


def _recent_tool_arguments_look_complete(protocol: str, events: Sequence[dict[str, Any]]) -> bool:
    if not events:
        return False
    tail = events[-4:]
    if protocol == "chat_completions":
        for event in tail:
            delta = event.get("choices", [{}])[0].get("delta", {}) if isinstance(event, dict) else {}
            for tc in delta.get("tool_calls") or []:
                arg = ((tc.get("function") or {}).get("arguments"))
                if isinstance(arg, str) and "}" in arg:
                    return True
    if protocol == "responses":
        return any(e.get("type") == "response.function_call_arguments.delta" and "}" in str(e.get("delta", "")) for e in tail if isinstance(e, dict))
    if protocol == "messages":
        return any((e.get("delta") or {}).get("type") == "input_json_delta" and "}" in str((e.get("delta") or {}).get("partial_json", "")) for e in tail if isinstance(e, dict))
    return False


def _probe_tool_calls_complete(d: Any, prepared: RecipePreparedRequest, response_id: str, model_id: str, tokenizer: Any, token_ids: Sequence[int]) -> bool:
    if not token_ids:
        return False
    try:
        processor, response = _make_processor_and_response(d, prepared, response_id + "_probe", model_id, tokenizer)
        for out in processor.push(d.InferenceChunk.ready()):
            if response is not None:
                response.append(out)
        for token in token_ids:
            for out in processor.push(d.InferenceChunk.token(int(token))):
                if response is not None:
                    response.append(out)
        for out in processor.push(d.InferenceChunk.finish(d.InferenceFinishReason.Stop)):
            if response is not None:
                response.append(out)
        for out in processor.finish():
            if response is not None:
                response.append(out)
        if response is None:
            return False
        data = json.loads(response.to_json())
        return (_protocol_finish_reason(prepared.protocol, data, []) in {"tool_calls", "tool_use", "completed"}
                and bool(_extract_tool_calls(prepared.protocol, data, [])))
    except Exception:
        return False


def _extract_tool_calls(protocol: str, response_json: dict[str, Any] | None, events: Sequence[dict[str, Any]]) -> list[M11ParsedToolCall]:
    calls: list[M11ParsedToolCall] = []
    if protocol == "chat_completions" and response_json:
        msg = response_json.get("choices", [{}])[0].get("message", {})
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            calls.append(M11ParsedToolCall(id=str(tc.get("id", "")), name=str(fn.get("name", "")), arguments=str(fn.get("arguments", "")), protocol=protocol, index=i))
    elif protocol == "chat_completions" and events:
        partial: dict[int, dict[str, str]] = {}
        for event in events:
            delta = event.get("choices", [{}])[0].get("delta", {}) if isinstance(event, dict) else {}
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                row = partial.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    row["id"] = str(tc.get("id"))
                fn = tc.get("function") or {}
                if fn.get("name"):
                    row["name"] = str(fn.get("name"))
                if fn.get("arguments") is not None:
                    row["arguments"] += str(fn.get("arguments"))
        for idx in sorted(partial):
            row = partial[idx]
            if row["name"] or row["arguments"]:
                calls.append(M11ParsedToolCall(id=row["id"], name=row["name"], arguments=row["arguments"], protocol=protocol, index=idx))
    elif protocol == "responses" and response_json:
        for i, item in enumerate(response_json.get("output") or []):
            if item.get("type") == "function_call":
                calls.append(M11ParsedToolCall(id=str(item.get("call_id", "")), name=str(item.get("name", "")), arguments=str(item.get("arguments", "")), protocol=protocol, index=i, response_item_id=item.get("id")))
    elif protocol == "messages" and response_json:
        for i, block in enumerate(response_json.get("content") or []):
            if block.get("type") == "tool_use":
                calls.append(M11ParsedToolCall(id=str(block.get("id", "")), name=str(block.get("name", "")), arguments=json.dumps(block.get("input", {}), separators=(",", ":")), protocol=protocol, index=i))
    return calls


def _protocol_finish_reason(protocol: str, response_json: dict[str, Any] | None, events: Sequence[dict[str, Any]]) -> str | None:
    if response_json:
        if protocol == "chat_completions":
            return response_json.get("choices", [{}])[0].get("finish_reason")
        if protocol == "responses":
            return response_json.get("status")
        if protocol == "messages":
            return response_json.get("stop_reason")
    return None
