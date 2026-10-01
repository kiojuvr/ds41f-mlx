"""DeepSeek recipe serving backend for the qualified oMLX GenerationBatch path."""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator
from uuid import uuid4
import importlib
import os
import subprocess
import sys

from ds41f_mlx.runtime.dwarfstar_prefill import DwarfStarMLXPrefillSession, PRODUCTION_PREFILL_SELECTOR
from ds41f_mlx.prefill_fp8_mlx import handoff_to_generation
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
from ds41f_mlx.runtime.omlx_generation import OMLXGenerationSession

DEFAULT_RECIPE = Path('/Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe')
DEFAULT_MODEL_ID = 'deepseek-v4.1-flash'
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


class DeepSeekRecipeRuntimeBackend:
    """Single-flight adapter from recipe prepared requests to token chunks."""

    def __init__(self, *, checkpoint: Path = DEFAULT_CHECKPOINT, omlx_path: Path = DEFAULT_OMLX, recipe_path: Path = DEFAULT_RECIPE, model_id: str = DEFAULT_MODEL_ID, native_out_dir: Path = Path('artifacts/m7/deepseek-recipe-serving/native')):
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
        session: OMLXGenerationSession | None = None
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
            cfg = OMLXDecodeConfig(omlx_path=self.omlx_path, checkpoint_path=self.checkpoint, engram_ssd_offload=True, preserve_mtp=False, speculation_enabled=False)
            session = await self._call(lambda: handoff_to_generation(prefill_result.live_result, self._model, terminal_prompt_token=first, config=cfg, max_tokens=max_tokens, sampler=sampler))
            self.active_generation_sessions += 1
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
                    await self._call(session.stop, 'request_cleanup')
                    await self._call(session.close)
                    self.active_generation_sessions = max(0, self.active_generation_sessions - 1)
            finally:
                trace.cleanup_called = True
                self._lock.release()

    def close(self) -> None:
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
            'model_id': self.model_id,
            'single_flight': True,
            'mtp_dspark': 'OFF',
        }
