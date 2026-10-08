"""DeepSeek recipe serving backend for ds41f-owned standard-off generation."""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, time
from typing import Any, AsyncIterator
from uuid import uuid4
import importlib
import os
import subprocess
import sys

from ds41f_mlx.config import DEFAULT_CHECKPOINT, DEFAULT_KV_ROOT, DEFAULT_MODEL_ID, DEFAULT_OMLX, DEFAULT_RECIPE, RuntimeConfig
from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession, PRODUCTION_PREFILL_SELECTOR
from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.target_generation import TargetGenerationSession
from ds41f_mlx.runtime.tool_boundary_session import M11RecipeToolSession, M11AssistantTurn
MODEL_ALIASES = {DEFAULT_MODEL_ID, 'deepseek-v41-flash', 'deepseek-flash'}
REFERENCE_VERTICAL_SLICE_CLASSIFICATION = 'PRODUCTION_PREFILL_REGRESSED_TO_REFERENCE_VERTICAL_SLICE'


def build_production_prefill_state_guarded(checkpoint: Path, native_out_dir: Path, tokens: list[int]):
    """Permanent guard: the reference vertical slice is not production serving."""

    raise RuntimeError(
        f'{REFERENCE_VERTICAL_SLICE_CLASSIFICATION}: reference vertical slice serving is disabled. '
        'Production serving uses PRODUCTION_PREFILL_SELECTOR=DENSE_P0_P7; '
        'DS41F_ALLOW_REFERENCE_VERTICAL_SLICE_SERVING is ignored by the production path.'
    )


def build_diagnostic_reference_prefill_state(checkpoint: Path, native_out_dir: Path, tokens: list[int]):
    """Explicit diagnostic-only access to the old reference vertical slice."""

    from tools.run_m4_omlx_base_decode_qualification import build_prefill_state
    return build_prefill_state(checkpoint, native_out_dir, tokens)


@dataclass
class RecipePreparedRequest:
    protocol: str
    conversation_request: Any
    converted: Any
    token_ids: list[int]
    image_sources: list[Any]
    include_usage: bool = False
    custom_tool_names: frozenset[str] = frozenset()
    stop_token_ids: tuple[int, ...] = ()
    multimodal: Any = None
    resolved_max_tokens: int | None = None
    capacity: dict[str, Any] | None = None

    @property
    def model(self) -> str | None:
        return self.conversation_request.model

    @property
    def stream(self) -> bool:
        return self.conversation_request.stream

    @property
    def inference_options(self) -> Any:
        return self.conversation_request.inference_options


@dataclass
class RequestTrace:
    request_id: str
    protocol: str
    prompt_tokens: int
    prefill_tokens: int
    first_decode_input: int
    production_prefill_selector: str | None = None
    prefill_segment_count: int | None = None
    prefill_frontier: int | None = None
    same_live_cache_handoff: bool | None = None
    handoff_count: int | None = None
    prompt_replay_count: int | None = None
    full_cache_repack_count: int | None = None
    exported: bool | None = None
    initial_admitted_frontier: int | None = None
    frontier_after_first_input: int | None = None
    frontier_after_terminal: int | None = None
    prefill_seconds: float | None = None
    image_encoding_seconds: float = 0.0
    image_encoded_count: int = 0
    prefill_phase_timings_s: dict[str, float] | None = None
    generated_tokens: list[int] = field(default_factory=list)
    decode_latencies_s: list[float] = field(default_factory=list)
    cleanup_called: bool = False
    cancelled: bool = False

    def to_json(self) -> dict[str, Any]:
        med = None
        if self.decode_latencies_s:
            vals = sorted(self.decode_latencies_s)
            n = len(vals)
            med = vals[n//2] if n % 2 else (vals[n//2-1] + vals[n//2]) / 2
        return {
            'request_id': self.request_id,
            'protocol': self.protocol,
            'prompt_tokens': self.prompt_tokens,
            'prefill_tokens': self.prefill_tokens,
            'first_decode_input': self.first_decode_input,
            'production_prefill_selector': self.production_prefill_selector,
            'prefill_segment_count': self.prefill_segment_count,
            'prefill_frontier': self.prefill_frontier,
            'same_live_cache_handoff': self.same_live_cache_handoff,
            'handoff_count': self.handoff_count,
            'prompt_replay_count': self.prompt_replay_count,
            'full_cache_repack_count': self.full_cache_repack_count,
            'exported': self.exported,
            'initial_admitted_frontier': self.initial_admitted_frontier,
            'frontier_after_first_input': self.frontier_after_first_input,
            'frontier_after_terminal': self.frontier_after_terminal,
            'prefill_seconds': self.prefill_seconds,
            'image_encoding_seconds': self.image_encoding_seconds,
            'image_encoded_count': self.image_encoded_count,
            'prefill_phase_timings_s': self.prefill_phase_timings_s,
            'generated_tokens': list(self.generated_tokens),
            'completion_tokens': len(self.generated_tokens),
            'median_decode_tok_s': None if med is None else 1.0 / med,
            'decode_latencies_s': list(self.decode_latencies_s),
            'cleanup_called': self.cleanup_called,
            'cancelled': self.cancelled,
        }


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


@dataclass
class StatefulSessionRecord:
    session_id: str
    protocol: str = 'chat_completions'
    m11: M11RecipeToolSession | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    request_count: int = 0
    busy: bool = False
    closed: bool = False
    last_turn: dict[str, Any] | None = None
    last_error: str | None = None
    persisted_artifact: dict[str, Any] | None = None
    active_stream: Any = None  # transport control only, never an execution owner
    recovery_state: str = 'ready'
    response_reservation: Any = None  # sole bounded serialized-response/retry authority

    def to_json(self) -> dict[str, Any]:
        # GET must not observe worker-owned history between M8 adoption and
        # canonical recipe feed. Busy exposes reservation/outcome metadata only.
        diag = None if self.busy or self.m11 is None else self.m11.diagnostics()
        return {
            'id': self.session_id,
            'protocol': self.protocol,
            'state': 'closed' if self.closed else ('busy' if self.busy else ('unrecoverable' if self.recovery_state == 'unrecoverable' else ('empty' if self.m11 is None else self.m11.m8.state))),
            'recovery_state': self.recovery_state,
            'response_reservation': None if self.response_reservation is None else self.response_reservation.certificate(),
            'active_request_id': None if self.active_stream is None else self.active_stream.request_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'request_count': self.request_count,
            'last_error': self.last_error,
            'last_turn': self.last_turn,
            'persisted_artifact': self.persisted_artifact,
            'diagnostics': diag,
        }


class DeepSeekRecipeRuntimeBackend:
    """Single-flight adapter from recipe prepared requests to token chunks."""

    def __init__(self, *, checkpoint: Path = DEFAULT_CHECKPOINT, omlx_path: Path = DEFAULT_OMLX, recipe_path: Path = DEFAULT_RECIPE, model_id: str = DEFAULT_MODEL_ID, native_out_dir: Path = Path('artifacts/m7/deepseek-recipe-serving/native'), runtime_config: RuntimeConfig | None = None, execution_strategy: str = 'off'):
        if execution_strategy not in ('off', 'first-party-mtp-development'):
            raise ValueError('unknown standard execution strategy')
        self._execution_strategy = execution_strategy
        if runtime_config is not None:
            checkpoint = runtime_config.checkpoint_path
            omlx_path = runtime_config.omlx_path
            recipe_path = runtime_config.recipe_path
            model_id = runtime_config.model_id
            runtime_config.apply_environment()
        self.runtime_config = runtime_config
        self.checkpoint = Path(checkpoint)
        self.omlx_path = Path(omlx_path)
        self.recipe_path = Path(recipe_path)
        self.model_id = model_id
        self.native_out_dir = Path(native_out_dir)
        if str(self.omlx_path) not in sys.path:
            sys.path.insert(0, str(self.omlx_path))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ds41f-recipe-infer')
        self._lock = asyncio.Lock()
        self._runtime: OmlxRuntime | None = None
        self._model = None
        self.traces: deque[RequestTrace] = deque(maxlen=int(os.environ.get('DS41F_TRACE_HISTORY_LIMIT', '32')))
        self.last_trace: RequestTrace | None = None
        self.fatal_error: str | None = None
        self.active_generation_sessions = 0
        self.max_live_sessions = int(os.environ.get('DS41F_MAX_LIVE_SESSIONS', '4'))
        self.sessions: dict[str, StatefulSessionRecord] = {}
        self.session_traces: deque[dict[str, Any]] = deque(maxlen=int(os.environ.get('DS41F_TRACE_HISTORY_LIMIT', '32')))

    @property
    def execution_strategy(self):
        return self._execution_strategy

    def validate_development_body(self, body):
        if self.execution_strategy == 'off':
            return
        import json
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError('development first-party MTP requires a Chat object')
        if any(payload.get(key) for key in ('functions', 'function_call', 'response_format')):
            raise ValueError('legacy functions / structured output are not qualified for first-party MTP')
        if payload.get('reasoning_effort') != 'none':
            raise ValueError('development first-party MTP requires explicitly disabled reasoning')
        if payload.get('max_tokens') == 'auto' or payload.get('n', 1) != 1:
            raise ValueError('development first-party MTP requires explicit bounded single output')
        for message in payload.get('messages', []):
            if message.get('role') not in ('system', 'user', 'assistant', 'tool') or not isinstance(message.get('content'), (str, type(None))):
                raise ValueError('development first-party MTP admits text dialogue only')

    def load(self) -> None:
        if self._runtime is not None:
            return
        runtime = OmlxRuntime(OmlxRuntimeConfig(omlx_path=self.omlx_path, checkpoint_path=self.checkpoint, engram_ssd_offload=True, preserve_mtp=False, recipe_path=self.recipe_path))
        try:
            model, _ = runtime.load_model()
            if self.execution_strategy == 'first-party-mtp-development':
                runtime.admission.admit_proposal_child(model.language_model)
        except BaseException:
            runtime.close()
            raise
        self._runtime, self._model = runtime, model

    async def _call(self, fn, *args, on_cancel=None):
        """Cancellation cannot release ownership while the worker still mutates it."""
        from anyio import CancelScope
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, lambda: fn(*args))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            async def settle(pending):
                with CancelScope(shield=True):
                    while True:
                        try:
                            return await asyncio.shield(pending)
                        except asyncio.CancelledError:
                            if pending.done():
                                return pending.result()
            result = await settle(future)
            if on_cancel is not None:
                await settle(loop.run_in_executor(self._executor, lambda: on_cancel(result)))
            # Do not retain private tensor results in cancellation tracebacks.
            result = future = None
            raise

    def make_sampler(self, options: Any):
        temp = 0.0 if options.temperature is None else float(options.temperature)
        top_p = 0.0 if options.top_p is None else float(options.top_p)
        if temp < 0:
            raise ValueError('temperature must be >= 0')
        if top_p < 0 or top_p > 1:
            raise ValueError('top_p must be in [0, 1]')
        sample_utils = importlib.import_module('mlx_lm.sample_utils')
        return sample_utils.make_sampler(temp=temp, top_p=top_p)

    def max_tokens(self, options: Any) -> int:
        value = options.max_tokens
        if value is None:
            return 128
        value = int(value)
        if value <= 0:
            raise ValueError('max_tokens must be positive')
        return value

    def request_max_tokens(self, request: RecipePreparedRequest) -> int:
        maximum = getattr(request, 'resolved_max_tokens', None)
        return self.max_tokens(request.inference_options) if maximum is None else maximum

    def validate_stateful_admission(self, session_id, request):
        """Observation only; shared by preview and actual reservation."""
        self.validate_multimodal_request(request)
        if request.model is not None and request.model not in MODEL_ALIASES:
            raise RuntimeError('unsupported model')
        rec = self.get_stateful_session(session_id)
        if rec.busy:
            raise RuntimeError('session already has an active request')
        if rec.recovery_state == 'unrecoverable':
            raise RuntimeError('session is protocol-unrecoverable; DELETE required, no automatic replay')
        if rec.m11 is not None:
            history = rec.m11.m8.token_history
            if len(request.token_ids) <= len(history) or request.token_ids[:len(history)] != history:
                raise RuntimeError('next recipe encoding is not an exact extension of generated session history')
            if request.multimodal is not None:
                request.multimodal.validate_prefix(len(history), rec.m11.image_identities)
            elif rec.m11.image_identities:
                raise RuntimeError('continuation must retain committed inline image identities')
        return rec

    def validate_multimodal_request(self, request: RecipePreparedRequest) -> None:
        if self.execution_strategy == 'first-party-mtp-development':
            if request.protocol != 'chat_completions' or request.multimodal is not None or request.image_sources:
                raise ValueError('development first-party MTP admits stateful text Chat only')
            if request.conversation_request.parsing_options.parse_tool_calls:
                from ..runtime.tool_boundary_session import _recipe_module
                if not hasattr(_recipe_module().StreamProcessor, 'preview_eof_tokens'):
                    raise ValueError('native EOF-sensitive tool preview is required for first-party MTP')
            if request.conversation_request.conversation.thinking_mode:
                raise ValueError('development first-party MTP reasoning is unqualified')
            if len(request.token_ids) + self.request_max_tokens(request) > 512:
                raise ValueError('development first-party MTP reservation exceeds 512 positions')
            if self.request_max_tokens(request) > 64:
                raise ValueError('development first-party MTP output exceeds 64 tokens')
            if request.inference_options.temperature not in (None, 0, 0.0):
                raise ValueError('development application qualification is greedy only')
        if request.image_sources and request.multimodal is None:
            raise ValueError('image inputs must be prepared before execution')
        if request.multimodal is None and 129264 in request.token_ids:
            raise ValueError('missing image representation for image tokens')
        if request.multimodal is not None:
            if request.protocol != 'chat_completions':
                raise ValueError('multimodal support is qualified for Chat Completions only')
            from ds41f_mlx.runtime.multimodal import MAX_MULTIMODAL_CONTEXT
            if tuple(request.token_ids) != request.multimodal.token_ids:
                raise ValueError('multimodal tokens differ from prepared representation')
            if len(request.token_ids) + self.request_max_tokens(request) > MAX_MULTIMODAL_CONTEXT:
                raise ValueError('multimodal prompt plus output reservation exceeds qualified context')

    async def infer(self, request: RecipePreparedRequest) -> AsyncIterator[Any]:
        from deepseek_recipe import InferenceChunk, InferenceFinishReason, PromptUsage

        if self.execution_strategy != 'off':
            raise ValueError('development first-party MTP requires a fenced stateful session')
        if request.model is not None and request.model not in MODEL_ALIASES:
            raise ValueError(f"unsupported model {request.model!r}; supported aliases: {sorted(MODEL_ALIASES)}")
        self.validate_multimodal_request(request)
        if len(request.token_ids) < 2:
            raise ValueError(f'encoded prompt must contain prefix + first token, got {len(request.token_ids)} token(s)')
        prefix = request.token_ids[:-1]
        first = int(request.token_ids[-1])
        if len(prefix) + 1 != len(request.token_ids):
            raise AssertionError('off-by-one prompt split invariant failed')

        trace = RequestTrace(str(uuid4()), request.protocol, len(request.token_ids), len(prefix), first)
        self.last_trace = trace
        self.traces.append(trace)
        session: TargetGenerationSession | None = None
        session_counted = False
        lock_acquired = False
        try:
            await self._lock.acquire()
            lock_acquired = True
            await self._call(self.load)
            sampler = self.make_sampler(request.inference_options)
            max_tokens = self.request_max_tokens(request)
            prefill_session = DwarfStarMLXPrefillSession(self._model, omlx_path=self.omlx_path)
            image_embeddings = None
            if request.multimodal is not None:
                image_embeddings = await self._call(request.multimodal.encode_new, self._model)
                trace.image_encoding_seconds = image_embeddings.seconds
                trace.image_encoded_count = len(image_embeddings.rows)
            prefill_result = await self._call(lambda: prefill_session.prefill(prefix) if image_embeddings is None else prefill_session.prefill(prefix, image_embeddings=image_embeddings), on_cancel=lambda result: result.live_result.discard())
            image_embeddings = None
            trace.production_prefill_selector = getattr(prefill_result, 'production_prefill_selector', PRODUCTION_PREFILL_SELECTOR)
            trace.prefill_seconds = float(prefill_result.seconds)
            trace.prefill_phase_timings_s = {k: float(v) for k, v in prefill_result.phase_timings_s.items()}
            trace.prefill_segment_count = len(getattr(prefill_result, 'segment_metadata', ()))
            trace.prefill_frontier = int(prefill_result.frontier)
            trace.full_cache_repack_count = int(getattr(prefill_result, 'full_cache_repack_count', 0))
            trace.exported = bool(getattr(prefill_result, 'portable_state_exported', False))
            if trace.production_prefill_selector != PRODUCTION_PREFILL_SELECTOR or trace.prefill_frontier != len(prefix):
                raise RuntimeError('production DENSE_P0_P7 prefill selector/frontier gate failed')
            cfg = OMLXDecodeConfig(omlx_path=self.omlx_path, checkpoint_path=self.checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False, stop_token_ids=tuple(request.stop_token_ids))
            session = await self._call(lambda: handoff_to_generation(prefill_result.live_result, self._model, terminal_prompt_token=first, config=cfg, max_tokens=max_tokens, sampler=sampler), on_cancel=lambda generation: generation.close())
            self.active_generation_sessions += 1
            session_counted = True
            trace.initial_admitted_frontier = session.admitted_frontier
            trace.frontier_after_first_input = session.token_frontier
            trace.frontier_after_terminal = session.token_frontier
            trace.prompt_replay_count = session.prompt_replay_count
            trace.same_live_cache_handoff = True
            trace.handoff_count = int(prefill_result.live_result.handoff_count)
            if trace.prompt_replay_count != 0 or trace.initial_admitted_frontier != len(prefix) or trace.frontier_after_terminal != len(prefix) + 1:
                raise RuntimeError('P5 no-replay terminal-holdout accounting gate failed')
            yield InferenceChunk.ready(prompt_usage=PromptUsage(prompt_tokens=len(request.token_ids), prompt_cache_hit_tokens=0))
            while len(trace.generated_tokens) < max_tokens:
                report = await self._call(session.next_token)
                if report is None:
                    break
                trace.generated_tokens.append(int(report.token))
                trace.decode_latencies_s.append(float(report.latency_s))
                suppress_protocol_token = (report.finish_reason == 'stop' and int(report.token) in set(request.stop_token_ids))
                if not suppress_protocol_token:
                    yield InferenceChunk.token(int(report.token))
                if report.finish_reason is not None:
                    reason = InferenceFinishReason.Length if report.finish_reason == 'length' else InferenceFinishReason.Stop
                    yield InferenceChunk.finish(reason)
                    break
            else:
                yield InferenceChunk.finish(InferenceFinishReason.Length)
        except GeneratorExit:
            trace.cancelled = True
            raise
        except asyncio.CancelledError:
            trace.cancelled = True
            raise
        finally:
            try:
                if session is not None:
                    try:
                        await self._call(session.stop, 'request_cleanup')
                    finally:
                        try:
                            await self._call(session.close)
                        finally:
                            if session_counted:
                                self.active_generation_sessions = max(0, self.active_generation_sessions - 1)
                                session_counted = False
            finally:
                trace.cleanup_called = True
                if lock_acquired:
                    self._lock.release()

    async def create_stateful_session(self, *, session_id: str | None = None) -> StatefulSessionRecord:
        if self.execution_strategy != 'off' and any(not s.closed for s in self.sessions.values()):
            raise RuntimeError('development first-party MTP admits one live session')
        sid = session_id or f"sess_{uuid4().hex}"
        if sid in self.sessions and not self.sessions[sid].closed:
            raise ValueError(f"session {sid!r} already exists")
        if len([s for s in self.sessions.values() if not s.closed]) >= self.max_live_sessions:
            raise RuntimeError("maximum live session count reached")
        rec = StatefulSessionRecord(session_id=sid)
        self.sessions[sid] = rec
        return rec

    def get_stateful_session(self, session_id: str) -> StatefulSessionRecord:
        rec = self.sessions.get(session_id)
        if rec is None or rec.closed:
            raise KeyError(f"unknown or closed session {session_id!r}")
        return rec

    def _require_response_reservation(self, rec):
        slot = rec.response_reservation
        if (slot is not None and slot.state != 'active') or (slot is None and self.execution_strategy != 'off'):
            raise ValueError('fenced generation requires an active exact-byte response reservation')

    async def run_stateful_chat_turn(self, session_id: str, request: RecipePreparedRequest, *, tokenizer: Any, body=None, options=None) -> M11AssistantTurn:
        if request.protocol != 'chat_completions':
            raise ValueError('M12 stateful serving currently qualifies chat_completions only')
        if request.model is not None and request.model not in MODEL_ALIASES:
            raise ValueError(f"unsupported model {request.model!r}; supported aliases: {sorted(MODEL_ALIASES)}")
        rec = self.validate_stateful_admission(session_id, request)
        self._require_response_reservation(rec)
        slot = rec.response_reservation
        if body is None and slot is not None:
            body = slot.request_bytes
        if slot is not None and bytes(body) != slot.request_bytes:
            raise ValueError('canonical JSON publication request-body identity mismatch')
        if request.conversation_request.parsing_options.parse_tool_calls and body is None:
            raise ValueError('canonical tool publication requires original request body')
        request_id = str(uuid4())
        if slot is not None:
            slot.request_id = request_id
        rec.busy = True
        rec.updated_at = time()
        trace: dict[str, Any] = {'session_id': session_id, 'prompt_tokens': len(request.token_ids), 'stream': bool(request.stream), 'started_at': rec.updated_at}
        turn_t0 = perf_counter()
        self.session_traces.append(trace)
        lock_acquired = False
        try:
            await self._lock.acquire()
            lock_acquired = True
            await self._call(self.load)
            sampler = self.make_sampler(request.inference_options)
            max_tokens = self.request_max_tokens(request)
            def execute_and_publish() -> M11AssistantTurn:
                # Publish completed state and its protocol record before the worker
                # result can be lost to transport/task cancellation. No new KV owner.
                if rec.m11 is None:
                    rec.m11, assistant_turn = self._start_and_run_m11(tokenizer, request, sampler, max_tokens)
                    trace['created_runtime_session'] = True
                else:
                    before = rec.m11.m8.frontier
                    try:
                        rec.m11.m8.sampler = sampler
                        rec.m11.continue_from_prepared(request, max_tokens=max_tokens)
                        assistant_turn = self._run_recipe_turn(rec.m11, request)
                    except BaseException:
                        self._retire_failed_recipe_session(rec.m11)
                        raise
                    trace['frontier_before'] = before
                    trace['frontier_after'] = rec.m11.m8.frontier
                    trace['exact_prefix_extension'] = True
                    trace['created_runtime_session'] = False
                from .recipe_publication import publish_recipe_turn
                try:
                    publish_recipe_turn(self, rec, request, tokenizer, body, options, assistant_turn,
                                        request_id=request_id)
                except BaseException:
                    self._retire_failed_recipe_session(rec.m11)
                    raise
                slot = rec.response_reservation
                if slot is not None and slot.state == 'active' and slot.media_type == 'application/json':
                    from fastapi.responses import JSONResponse
                    try:
                        slot.append(JSONResponse(content=assistant_turn.response_json).body)
                        slot.complete()
                    except BaseException:
                        slot.burn()
                        rec.recovery_state = 'unrecoverable'
                        raise
                diag = rec.m11.diagnostics()
                m8diag = diag.get('m8', {})
                last_runtime_turn = (m8diag.get('turns') or [{}])[-1]
                trace.update({
                    'ok': True,
                    'image_encoding_seconds': diag.get('last_image_encoding_seconds', 0.0),
                    'image_encoded_count': diag.get('last_image_encoded_count', 0),
                    'finish_reason': assistant_turn.finish_reason,
                    'tool_call_count': len(assistant_turn.tool_calls),
                    'generated_token_count': len(assistant_turn.generated_tokens),
                    'frontier': m8diag.get('frontier'),
                    'all_cache_offsets_equal_frontier': m8diag.get('all_cache_offsets_equal_frontier'),
                    'cache_layer_count': m8diag.get('cache_layer_count'),
                    'cache_offsets_all': m8diag.get('cache_offsets_all'),
                    'prompt_suffix_tokens': last_runtime_turn.get('prompt_suffix_tokens'),
                    'appended_prefill_tokens': last_runtime_turn.get('appended_prefill_tokens'),
                    'append_seconds': last_runtime_turn.get('append_seconds'),
                    'first_token_latency_s': last_runtime_turn.get('first_token_latency_s'),
                    'decode_seconds': last_runtime_turn.get('decode_seconds'),
                    'decode_tok_s': last_runtime_turn.get('decode_tok_s'),
                    'prompt_replay_count': m8diag.get('total_prompt_replay_count'),
                    'full_cache_repack_count': m8diag.get('total_full_cache_repack_count'),
                    'elapsed_seconds': perf_counter() - turn_t0,
                })
                return assistant_turn
            return await self._call(execute_and_publish)
        except asyncio.CancelledError:
            trace['cancelled'] = True
            raise
        except Exception as exc:
            rec.last_error = str(exc)
            if rec.response_reservation is not None and rec.response_reservation.state == 'active':
                rec.response_reservation.burn()
            if self.execution_strategy != 'off':
                rec.recovery_state = 'unrecoverable'
            if rec.m11 is not None and rec.m11.m8.closed:
                rec.recovery_state = 'unrecoverable'
                generation, rec.m11.m8.generation = rec.m11.m8.generation, None
                if generation is not None:
                    await self._call(generation.close)
            trace.update({'ok': False, 'error': str(exc), 'error_type': type(exc).__name__})
            raise
        finally:
            rec.busy = False
            rec.updated_at = time()
            if lock_acquired:
                self._lock.release()

    async def open_stateful_stream(self, session_id, request, *, tokenizer, body, options=None, application_id=None):
        rec = self.get_stateful_session(session_id)
        self._require_response_reservation(rec)
        if rec.response_reservation is not None and bytes(body) != rec.response_reservation.request_bytes:
            raise ValueError('canonical SSE publication request-body identity mismatch')
        from .stateful_stream import StatefulStream
        return StatefulStream(self, session_id, request, tokenizer, body, options, application_id)

    async def cancel_stateful_stream(self, session_id, request_id):
        rec = self.get_stateful_session(session_id)
        stream = rec.active_stream
        if stream is None or stream.request_id != request_id:
            raise RuntimeError('no matching active stream; stale cancellation refused')
        stream.cancel.set()
        await stream.aclose()
        return {'id': session_id, 'request_id': request_id, 'cancellation_requested': True, 'session': rec.to_json()}

    def _start_and_run_m11(self, tokenizer: Any, request: RecipePreparedRequest, sampler: Any, max_tokens: int) -> tuple[M11RecipeToolSession, M11AssistantTurn]:
        sess = M11RecipeToolSession.start_from_prepared(model=self._model, tokenizer=tokenizer, checkpoint=self.checkpoint, omlx_path=self.omlx_path, recipe_path=self.recipe_path, prepared=request, sampler=sampler, max_tokens=max_tokens)
        try:
            return sess, self._run_recipe_turn(sess, request)
        except BaseException:
            self._retire_failed_recipe_session(sess)
            raise

    @staticmethod
    def _retire_failed_recipe_session(sess):
        # A failed recipe publication cannot use ensure_idle to extract a burned
        # target. Release its generation lease directly, even if M8 is closed.
        m8 = sess.m8
        generation, m8.generation = m8.generation, None
        try:
            if generation is not None:
                generation._invalidate()
                generation.close()
        finally:
            m8.live_cache = []
            m8.closed = True

    def _run_recipe_turn(self, sess, request):
        if self.execution_strategy == 'off':
            return sess.run_current_assistant_turn(request)
        from ds41f_mlx.runtime.live_turn import LiveRecipeTurn
        sess.execution_strategy = self.execution_strategy
        cursor = LiveRecipeTurn(sess, request, lambda turn, cancelled: None)
        try:
            while not cursor.closed:
                cursor.advance()
            return cursor.turn
        except BaseException:
            cursor.abort()
            raise

    async def persist_stateful_session(self, session_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
        if self.execution_strategy != 'off':
            raise ValueError('development first-party MTP persistence is unqualified')
        rec = self.get_stateful_session(session_id)
        if rec.busy:
            raise RuntimeError(f"session {session_id!r} already has an active request")
        if rec.m11 is None:
            raise ValueError('cannot persist an empty session')
        rec.busy = True
        lock_acquired = False
        try:
            await self._lock.acquire()
            lock_acquired = True
            default_root = (self.runtime_config.kv_root if self.runtime_config is not None else DEFAULT_KV_ROOT) / 'm12'
            def save_and_publish():
                info = rec.m11.persist_idle(artifact_root=artifact_root or default_root, diagnostics={'m12_session_id': session_id})
                rec.persisted_artifact = info.to_json()
                rec.updated_at = time()
                return rec.persisted_artifact
            return await self._call(save_and_publish)
        finally:
            rec.busy = False
            if lock_acquired:
                self._lock.release()

    async def restore_stateful_session(self, *, artifact_path: Path, tokenizer: Any, session_id: str | None = None) -> StatefulSessionRecord:
        if self.execution_strategy != 'off':
            raise ValueError('development first-party MTP restore is unqualified')
        rec = await self.create_stateful_session(session_id=session_id)
        rec.busy = True
        lock_acquired = False
        try:
            await self._lock.acquire()
            lock_acquired = True
            await self._call(self.load)
            def restore() -> M11RecipeToolSession:
                return M11RecipeToolSession.restore(model=self._model, tokenizer=tokenizer, checkpoint=self.checkpoint, omlx_path=self.omlx_path, recipe_path=self.recipe_path, artifact_path=artifact_path, protocol='chat_completions', model_id=self.model_id)
            rec.m11 = await self._call(restore, on_cancel=lambda session: session.m8.close())
            rec.persisted_artifact = {'path': str(artifact_path)}
            rec.updated_at = time()
            return rec
        except BaseException:
            rec.closed = True
            raise
        finally:
            rec.busy = False
            if lock_acquired:
                self._lock.release()

    async def close_stateful_session(self, session_id: str) -> dict[str, Any]:
        rec = self.get_stateful_session(session_id)
        if rec.busy:
            raise RuntimeError(f"session {session_id!r} already has an active request")
        rec.closed = True
        if rec.m11 is not None:
            await self._call(rec.m11.m8.close)
        rec.updated_at = time()
        result = rec.to_json()
        # GET already rejects closed records; retaining their full parser/history
        # payloads forever serves no public authority or recovery purpose.
        self.sessions.pop(session_id, None)
        if rec.response_reservation is not None:
            rec.response_reservation.retire()
        return result

    def close(self) -> None:
        for rec in list(self.sessions.values()):
            if rec.m11 is not None:
                try:
                    if rec.m11.m8.generation is not None:
                        self._retire_failed_recipe_session(rec.m11)
                    else:
                        rec.m11.m8.close()
                except Exception:
                    pass
            if rec.response_reservation is not None:
                rec.response_reservation.retire()
            rec.closed = True
        if self._runtime is not None:
            self._runtime.close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def provenance(self) -> dict[str, Any]:
        try:
            from importlib.metadata import version
            recipe_version = version('deepseek-recipe')
            mlx_version = version('mlx')
        except Exception:
            recipe_version = None; mlx_version = None
        return {
            'deepseek_recipe_path': str(self.recipe_path),
            'deepseek_recipe_revision': git_rev(self.recipe_path),
            'deepseek_recipe_version': recipe_version,
            'omlx_path': str(self.omlx_path),
            'omlx_revision': git_rev(self.omlx_path),
            'mlx_version': mlx_version,
            'checkpoint': str(self.checkpoint),
            'kv_root': str(self.runtime_config.kv_root if self.runtime_config is not None else DEFAULT_KV_ROOT),
            'model_id': self.model_id,
            'single_flight': True,
            'production_prefill_selector': PRODUCTION_PREFILL_SELECTOR,
            'execution_strategy': self.execution_strategy,
            'target_preserve_mtp': False,
            'mtp': 'OFF' if self.execution_strategy == 'off' else 'FIRST_PARTY_DEVELOPMENT',
            'dspark': 'OFF' if self.execution_strategy == 'off' else 'FIRST_PARTY_DEVELOPMENT',
            'speculative_decode': 'OFF' if self.execution_strategy == 'off' else 'SEMANTIC_AUTHORIZED_DEVELOPMENT',
        }
