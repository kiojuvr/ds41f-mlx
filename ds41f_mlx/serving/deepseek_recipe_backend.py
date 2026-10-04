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

    def to_json(self) -> dict[str, Any]:
        diag = None if self.m11 is None else self.m11.diagnostics()
        return {
            'id': self.session_id,
            'protocol': self.protocol,
            'state': 'closed' if self.closed else ('busy' if self.busy else ('empty' if self.m11 is None else self.m11.m8.state)),
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

    def __init__(self, *, checkpoint: Path = DEFAULT_CHECKPOINT, omlx_path: Path = DEFAULT_OMLX, recipe_path: Path = DEFAULT_RECIPE, model_id: str = DEFAULT_MODEL_ID, native_out_dir: Path = Path('artifacts/m7/deepseek-recipe-serving/native'), runtime_config: RuntimeConfig | None = None):
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

    def load(self) -> None:
        if self._runtime is not None:
            return
        self._runtime = OmlxRuntime(OmlxRuntimeConfig(omlx_path=self.omlx_path, checkpoint_path=self.checkpoint, engram_ssd_offload=True, preserve_mtp=False))
        self._model, _ = self._runtime.load_model()

    async def _call(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

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

    async def infer(self, request: RecipePreparedRequest) -> AsyncIterator[Any]:
        from deepseek_recipe import InferenceChunk, InferenceFinishReason, PromptUsage

        if request.model is not None and request.model not in MODEL_ALIASES:
            raise ValueError(f"unsupported model {request.model!r}; supported aliases: {sorted(MODEL_ALIASES)}")
        if request.image_sources:
            raise ValueError('multimodal/image input is not supported by ds41f text serving backend')
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
        await self._lock.acquire()
        try:
            await self._call(self.load)
            sampler = self.make_sampler(request.inference_options)
            max_tokens = self.max_tokens(request.inference_options)
            prefill_session = DwarfStarMLXPrefillSession(self._model, omlx_path=self.omlx_path)
            prefill_result = await self._call(prefill_session.prefill, prefix)
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
            session = await self._call(lambda: handoff_to_generation(prefill_result.live_result, self._model, terminal_prompt_token=first, config=cfg, max_tokens=max_tokens, sampler=sampler))
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
                self._lock.release()

    async def create_stateful_session(self, *, session_id: str | None = None) -> StatefulSessionRecord:
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

    async def run_stateful_chat_turn(self, session_id: str, request: RecipePreparedRequest, *, tokenizer: Any) -> M11AssistantTurn:
        if request.protocol != 'chat_completions':
            raise ValueError('M12 stateful serving currently qualifies chat_completions only')
        if request.model is not None and request.model not in MODEL_ALIASES:
            raise ValueError(f"unsupported model {request.model!r}; supported aliases: {sorted(MODEL_ALIASES)}")
        if request.image_sources:
            raise ValueError('multimodal/image input is not supported by ds41f stateful serving')
        rec = self.get_stateful_session(session_id)
        if rec.busy:
            raise RuntimeError(f"session {session_id!r} already has an active request")
        rec.busy = True
        rec.updated_at = time()
        trace: dict[str, Any] = {'session_id': session_id, 'prompt_tokens': len(request.token_ids), 'stream': bool(request.stream), 'started_at': rec.updated_at}
        turn_t0 = perf_counter()
        self.session_traces.append(trace)
        await self._lock.acquire()
        try:
            await self._call(self.load)
            sampler = self.make_sampler(request.inference_options)
            max_tokens = self.max_tokens(request.inference_options)
            if rec.m11 is None:
                turn = await self._call(lambda: self._start_and_run_m11(tokenizer, request, sampler, max_tokens))
                rec.m11 = turn[0]
                assistant_turn = turn[1]
                trace['created_runtime_session'] = True
            else:
                def cont() -> M11AssistantTurn:
                    assert rec.m11 is not None
                    before = rec.m11.m8.frontier
                    rec.m11.continue_from_prepared(request, max_tokens=max_tokens)
                    out = rec.m11.run_current_assistant_turn(request)
                    trace['frontier_before'] = before
                    trace['frontier_after'] = rec.m11.m8.frontier
                    trace['exact_prefix_extension'] = True
                    return out
                assistant_turn = await self._call(cont)
                trace['created_runtime_session'] = False
            rec.request_count += 1
            rec.updated_at = time()
            rec.last_turn = assistant_turn.to_json()
            rec.last_error = None
            diag = rec.m11.diagnostics() if rec.m11 is not None else {}
            m8diag = diag.get('m8', {}) if isinstance(diag, dict) else {}
            last_runtime_turn = (m8diag.get('turns') or [{}])[-1]
            trace.update({
                'ok': True,
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
        except Exception as exc:
            rec.last_error = str(exc)
            trace.update({'ok': False, 'error': str(exc), 'error_type': type(exc).__name__})
            raise
        finally:
            rec.busy = False
            rec.updated_at = time()
            self._lock.release()

    def _start_and_run_m11(self, tokenizer: Any, request: RecipePreparedRequest, sampler: Any, max_tokens: int) -> tuple[M11RecipeToolSession, M11AssistantTurn]:
        sess = M11RecipeToolSession.start_from_prepared(model=self._model, tokenizer=tokenizer, checkpoint=self.checkpoint, omlx_path=self.omlx_path, recipe_path=self.recipe_path, prepared=request, sampler=sampler, max_tokens=max_tokens)
        return sess, sess.run_current_assistant_turn(request)

    async def persist_stateful_session(self, session_id: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
        rec = self.get_stateful_session(session_id)
        if rec.busy:
            raise RuntimeError(f"session {session_id!r} already has an active request")
        if rec.m11 is None:
            raise ValueError('cannot persist an empty session')
        rec.busy = True
        await self._lock.acquire()
        try:
            default_root = (self.runtime_config.kv_root if self.runtime_config is not None else DEFAULT_KV_ROOT) / 'm12'
            info = await self._call(lambda: rec.m11.persist_idle(artifact_root=artifact_root or default_root, diagnostics={'m12_session_id': session_id}))
            rec.persisted_artifact = info.to_json()
            rec.updated_at = time()
            return rec.persisted_artifact
        finally:
            rec.busy = False
            self._lock.release()

    async def restore_stateful_session(self, *, artifact_path: Path, tokenizer: Any, session_id: str | None = None) -> StatefulSessionRecord:
        rec = await self.create_stateful_session(session_id=session_id)
        rec.busy = True
        await self._lock.acquire()
        try:
            await self._call(self.load)
            def restore() -> M11RecipeToolSession:
                return M11RecipeToolSession.restore(model=self._model, tokenizer=tokenizer, checkpoint=self.checkpoint, omlx_path=self.omlx_path, recipe_path=self.recipe_path, artifact_path=artifact_path, protocol='chat_completions', model_id=self.model_id)
            rec.m11 = await self._call(restore)
            rec.persisted_artifact = {'path': str(artifact_path)}
            rec.updated_at = time()
            return rec
        except Exception:
            rec.closed = True
            raise
        finally:
            rec.busy = False
            self._lock.release()

    async def close_stateful_session(self, session_id: str) -> dict[str, Any]:
        rec = self.get_stateful_session(session_id)
        if rec.busy:
            raise RuntimeError(f"session {session_id!r} already has an active request")
        rec.closed = True
        if rec.m11 is not None:
            await self._call(rec.m11.m8.close)
        rec.updated_at = time()
        return rec.to_json()

    def close(self) -> None:
        for rec in list(self.sessions.values()):
            if rec.m11 is not None:
                try:
                    rec.m11.m8.close()
                except Exception:
                    pass
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
            'mtp': 'OFF',
            'dspark': 'OFF',
            'speculative_decode': 'OFF',
        }
