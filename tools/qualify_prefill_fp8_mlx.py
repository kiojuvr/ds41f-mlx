#!/usr/bin/env python3
"""Bounded P3/P4 evidence only: 8192 same-input source and window checks.

Requires both real structural smokes to have passed before invocation. No
serving selection, timing, decoder handoff, or independent transformer loop.
Exactness here applies only to packed state produced by identical loaded
operators on identical input, not to whole-model backend trajectories.

Reusable Phase 2 helpers below construct fresh ordinary observations and collect
bounded physical diagnostics. They are not selected by this historical CLI and
have no numerical acceptance thresholds; see the Phase 2 reference report.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from ds41f_mlx.prefill_fp8_mlx import DwarfStarFP8MLXPrefillExecutorSetup, SweepPlanner, SweepCommandKind, SweepPhase
from tools.run_prefill_fp8_mlx_smoke import geometry
from ds41f_mlx.prefill_fp8_mlx.handoff import _dtype


def runtime_authority(lang):
    path = Path(lang.__file__).resolve()
    def git(*args):
        result = subprocess.run(['git', '-C', str(path.parent), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    # Bind the adapter's keyword contract before loading hundreds of GiB.
    contracts = {
        'hc_mixes': ((None,) * 5, {}), 'hc_pre_norm': ((None,) * 4, {}),
        'hc_post': ((None,) * 4, {}), 'rope': ((None,) * 4, {'inverse': True}),
        'pack_activation': ((None,), {'bits': 4, 'group_size': 16, 'e4m3_scale': True}),
        'quantize_activation': ((None,), {'bits': 4}),
        'packed_index_topk': ((None,) * 6, {'block_count': 2, 'block_size': 16}),
        'packed_index_scores': ((None,) * 6, {}),
        'packed_sparse_attention': ((None,) * 7, {}),
    }
    checks = {}
    for name, (args, kwargs) in contracts.items():
        try:
            operation = getattr(lang, name)
            if not callable(operation):
                raise TypeError('not callable')
            inspect.signature(operation).bind(*args, **kwargs)
            checks[name] = 'compatible'
        except (AttributeError, TypeError, ValueError) as exc:
            checks[name] = str(exc)
    import omlx
    return {
        'omlx_version': getattr(omlx, '__version__', None),
        'distribution_version': importlib.metadata.version('omlx'),
        'mlx_version': importlib.metadata.version('mlx'),
        'python_executable': sys.executable,
        'language_path': str(path), 'language_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'git_revision': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
        'operation_contracts': checks,
    }


def ordinary_prefill_reference(model, input_ids, *, pixel_values=None,
                               image_grids=(), image_spans=(), image_types=()):
    """Test-only fresh ordinary layer loop; caller owns admission/resources.

    No candidate cache, sparse embedding, prefix lookup, CED flag, or P6 plan
    is accepted. Pixel preparation and native DSpark math remain shared. This
    helper is not a production executor or a numerical acceptance authority.
    Returned arrays are live observations, not immutable checkpoints.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 0 < input_ids.shape[1] <= 4096:
        raise ValueError('reference requires 1..4096 unpadded singleton positions')
    if len(image_spans) > 1 or not (len(image_grids) == len(image_spans) == len(image_types)):
        raise ValueError('reference requires at most one complete image')
    if (pixel_values is None) != (len(image_spans) == 0):
        raise ValueError('reference pixels/image metadata mismatch')
    for (start, length), kinds in zip(image_spans, image_types):
        if start < 0 or length <= 0 or start + length > input_ids.shape[1] or length != len(kinds):
            raise ValueError('reference image span is incomplete')
    lm = model.language_model
    if len(lm.layers) != 40 or len(lm._config.dspark_target_layer_ids) != 3:
        raise ValueError('reference requires Flash 40-layer/3-tap topology')
    cache, rings = lm.make_cache(), lm.make_mtp_cache()
    if len(cache) != 40 or len(rings) != 3 or any(c.size() != 0 for c in cache) or any(r.offset != 0 or r.keys is not None for r in rings):
        raise ValueError('reference factories did not provide fresh 40/3 state')
    embeddings = model.get_input_embeddings(
        input_ids, pixel_values=pixel_values, image_grids=image_grids,
        image_spans=image_spans, image_types=image_types).inputs_embeds
    # Bypass DSparkMixin.__call__ too: no implicit primed-context owner is used.
    logits, hidden = lm._forward(
        input_ids, cache=cache,
        inputs_embeds=embeddings if pixel_values is not None else None,
        return_dspark_hidden=True)
    lm.dspark_append_context(hidden, rings, start_offset=0)
    end = input_ids.shape[1]
    if any(c.size() != end for c in cache) or any(r.offset != end for r in rings):
        raise ValueError('reference Target/DSpark frontier mismatch')
    return dict(target=cache, rings=rings, logits=logits, embeddings=embeddings,
                token_ids=tuple(int(t) for t in input_ids.tolist()[0]))


def physical_tensor_metrics(actual, reference, *, chunk_elements=65536):
    """Bounded host diagnostics, never a fitted tolerance or PASS decision.

    Integer/packed values get exact mismatch counts, NOT misleading numerical
    errors on quantized bytes. Floating metrics cover full physical arrays;
    meaningful-row selection/packed decoding belongs to the recorded boundary.
    No ring rotation, padding removal, or state mutation is performed here.
    """
    import numpy as np
    if chunk_elements <= 0:
        raise ValueError('chunk_elements must be positive')
    if actual is None or reference is None:
        return dict(structure_equal=actual is None and reference is None,
                    actual_present=actual is not None, reference_present=reference is not None)
    out = dict(actual=geometry(actual), reference=geometry(reference))
    # P6 retains NumPy Engram history while ordinary _forward publishes MLX
    # history. Compare exact primitive dtype names, not backend dtype objects;
    # keep both original representations in geometry for disclosure.
    out['structure_equal'] = actual.shape == reference.shape and _dtype(actual) == _dtype(reference)
    if not out['structure_equal']:
        return out
    floating = 'float' in str(actual.dtype)
    a, b = actual.reshape(-1), reference.reshape(-1)
    mismatches, samples, nonfinite, nonfinite_mismatches = 0, [], 0, 0
    delta2 = ref2 = actual2 = dot = maximum = 0.0
    for start in range(0, a.size, chunk_elements):
        def host(value):
            if str(value.dtype).endswith('bfloat16'):
                import mlx.core as mx
                value = value.astype(mx.float32)
            return np.asarray(value)
        x, y = host(a[start:start+chunk_elements]), host(b[start:start+chunk_elements])
        different = x != y
        mismatches += int(np.count_nonzero(different))
        for index in np.flatnonzero(different)[:max(0, 4-len(samples))]:
            # Strings keep nonfinite samples valid in strict JSON evidence.
            samples.append(dict(flat_index=start+int(index), actual=str(x[index]), reference=str(y[index])))
        if floating:
            x, y = x.astype(np.float64), y.astype(np.float64)
            nonfinite += int(np.count_nonzero(~(np.isfinite(x) & np.isfinite(y))))
            nonfinite_mismatches += int(np.count_nonzero(
                (~(np.isfinite(x) & np.isfinite(y))) &
                ~((np.isnan(x) & np.isnan(y)) | (x == y))))
            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                delta = x-y
                maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
                delta2 += float(np.sum(delta*delta))
                ref2 += float(np.sum(y*y))
                actual2 += float(np.sum(x*x))
                dot += float(np.sum(x*y))
    out.update(elements=int(a.size), mismatches=mismatches, samples=samples)
    if floating:
        out.update(nonfinite_pairs=nonfinite, nonfinite_mismatches=nonfinite_mismatches,
                   max_absolute=None if nonfinite else maximum,
                   rms_relative=None if nonfinite or ref2 == 0 else (delta2/ref2)**0.5,
                   cosine=None if nonfinite or ref2 == 0 or actual2 == 0 else dot/(ref2*actual2)**0.5)
    return out


def meaningful_target_metrics(actual, reference, layer, config, *, chunk_elements=65536):
    """Singleton committed-state masks, derived from installed ordinary owners.

    Each decoded diagnostic is at most 16 rows; no full decoded cache survives.
    Physical shape/byte diagnostics remain separate in paired_physical_metrics.
    Refuse unsupported padding/layouts instead of silently trimming them.
    """
    import mlx.core as mx
    from omlx.patches.deepseek_v41.quantization import unpack_activation

    def selection(cache):
        end = cache.size()
        ratio = config.compress_ratios[layer] if layer in config.kv_source_layers else 0
        offset = cache.cache[0]
        if offset is None or offset.shape != (1,) or offset.dtype != mx.int32:
            raise ValueError('meaningful collector requires exact singleton offset layout')
        if end < 0 or cache.compress_ratio != ratio:
            raise ValueError('unsupported committed compression layout')
        for name in ('left_padding', 'lengths'):
            value = getattr(cache, name, None)
            if value is not None and (value.size != 1 or int(value.item()) != (0 if name == 'left_padding' else end)):
                raise ValueError('meaningful collector requires unpadded singleton state')
        rows = [None, min(end, config.window_size), end // ratio if ratio else 0,
                end // ratio if ratio and layer in config.index_source_layers else 0,
                end % ratio if ratio > 1 else 0, end % ratio if ratio > 1 else 0,
                config.engram_max_ngram_size - 1 if layer == 0 and config.engram_layer_ids else 0]
        # Slot 6 is exact integer history (including -1 sentinels), not floats.
        for slot, count in enumerate(rows[1:], 1):
            value = cache.cache[slot]
            expected_rank = 2 if slot == 6 else 3
            if value is None:
                if count:
                    raise ValueError(f'layer {layer} slot {slot}: missing meaningful rows')
            elif value.ndim != expected_rank or value.shape[0] != 1 or value.shape[1] < count:
                raise ValueError(f'layer {layer} slot {slot}: unsupported meaningful layout')
            if value is not None and slot in (1, 2, 3):
                width = config.index_head_dim if slot == 3 else config.head_dim
                bits, group = {1: (8, 32), 2: (4, 16), 3: (4, 32)}[slot]
                if width % group or value.shape[-1] != width*bits//8 + width//group or value.dtype != mx.uint8:
                    raise ValueError(f'layer {layer} slot {slot}: unsupported packed layout')
            if value is not None and slot == 6 and (
                    value.shape[1] != count or _dtype(value) != 'int64'):
                raise ValueError('unsupported exact Engram history layout')
        return dict(frontier=end, rows=rows,
                    window_positions=[max(0, end-config.window_size), end],
                    compressed_source=layer in config.kv_source_layers,
                    index_owner=layer in config.kv_source_layers and layer in config.index_source_layers)

    left, right = selection(actual), selection(reference)
    result = dict(selectors=[left, right], selectors_equal=left == right, slots=[])
    if left != right:
        return result  # different frontiers are not numerical comparisons
    for slot, (a, b) in enumerate(zip(actual.cache, reference.cache)):
        count = left['rows'][slot]
        item = dict(slot=slot, meaningful_rows=count)
        result['slots'].append(item)
        if slot == 0 or slot == 6:
            item['exact'] = physical_tensor_metrics(a, b, chunk_elements=chunk_elements)
            continue
        if a is None or b is None:
            item['presence_equal'] = a is None and b is None
            continue
        if a.dtype != b.dtype or a.shape[-1] != b.shape[-1]:
            item['layout_equal'] = False
            continue
        item['layout_equal'] = True
        item['chunks'] = []
        for start in range(0, count, 16):
            x, y = a[:, start:min(start+16, count)], b[:, start:min(start+16, count)]
            if slot in (1, 2, 3):
                bits, group, e4m3 = {1: (8, 32, False), 2: (4, 16, True), 3: (4, 32, False)}[slot]
                if a.dtype != mx.uint8 or a.shape[-1] % (group*bits//8+1):
                    raise ValueError('unsupported packed activation geometry')
                x = unpack_activation(x, bits, group, e4m3, dtype=mx.float32)
                y = unpack_activation(y, bits, group, e4m3, dtype=mx.float32)
            item['chunks'].append(dict(row_start=start, metrics=physical_tensor_metrics(x, y, chunk_elements=chunk_elements)))
    return result


def paired_physical_metrics(actual, reference, *, chunk_elements=65536, config=None):
    """Observe ALL 40 x 7 slots and 3 native rings; no acceptance inference.

    Inputs are dictionaries returned by ordinary_prefill_reference or equivalent
    candidate observation adapters. Metadata and physical bytes are retained as
    compact metrics. With admitted config, also decode bounded meaningful rows.
    This remains observation, not a numerical acceptance authority.
    """
    if any(len(p['target']) != 40 or len(p['rings']) != 3 for p in (actual, reference)):
        raise ValueError('incomplete 40-layer/3-ring observation')
    result = dict(status='OBSERVED_ONLY', target=[], rings=[])
    def metric(a, b):
        return physical_tensor_metrics(a, b, chunk_elements=chunk_elements)
    for index, (a, b) in enumerate(zip(actual['target'], reference['target'])):
        if len(a.cache) != 7 or len(b.cache) != 7:
            raise ValueError(f'layer {index}: incomplete seven-slot observation')
        result['target'].append(dict(
            layer=index, offsets=[a.size(), b.size()],
            metadata_equal=(a.size() == b.size() and tuple(a.meta_state) == tuple(b.meta_state) and a.compress_ratio == b.compress_ratio),
            meta_state=[list(a.meta_state), list(b.meta_state)],
            compress_ratio=[a.compress_ratio, b.compress_ratio],
            left_padding=metric(getattr(a, 'left_padding', None), getattr(b, 'left_padding', None)),
            lengths=metric(getattr(a, 'lengths', None), getattr(b, 'lengths', None)),
            slots=[metric(x, y) for x, y in zip(a.cache, b.cache)]))
        if config is not None:
            result['target'][-1]['meaningful'] = meaningful_target_metrics(
                a, b, index, config, chunk_elements=chunk_elements)
    for index, (a, b) in enumerate(zip(actual['rings'], reference['rings'])):
        def physical_positions(ring):
            if ring.max_size <= 0 or ring.offset < 0:
                raise ValueError('invalid ring capacity/offset')
            return [ring.offset-1-((ring.offset-1-slot) % ring.max_size)
                    for slot in range(min(ring.offset, ring.max_size))]
        result['rings'].append(dict(ring=index, capacity=[a.max_size, b.max_size],
                                    offsets=[a.offset, b.offset],
                                    physical_positions=[physical_positions(a), physical_positions(b)],
                                    mapping_equal=(a.max_size == b.max_size and a.offset == b.offset and physical_positions(a) == physical_positions(b)),
                                    keys=metric(a.keys, b.keys)))
        for ring in (a, b):
            count = min(ring.offset, ring.max_size)
            if (ring.keys is None and count) or (ring.keys is not None and
                    (ring.keys.ndim != 4 or ring.keys.shape[:3] != (1, 1, count))):
                raise ValueError(f'ring {index}: invalid committed physical layout')
    return result


def paired_boundary_metrics(actual, reference, *, boundary, config, chunk_elements=65536):
    """Read-only boundary guard for existing capture/P6/P5 diagnostic seams.

    P5 may use independently recomputed ordinary Target(N) and rings(N-1).
    Never advance/settle either side to manufacture frontier equivalence.
    Require canonical consumed token identities, not only integer offsets.
    Image identities/metadata must additionally be bound by the case adapter.
    """
    if boundary not in ('n-2-capture', 'n-1-seal', 'p5-terminal'):
        raise ValueError('unknown numerical observation boundary')
    def frontiers(pair):
        if len(pair['target']) != 40 or len(pair['rings']) != 3:
            raise ValueError('incomplete 40-layer/3-ring observation')
        target = [c.size() for c in pair['target']]
        rings = [r.offset for r in pair['rings']]
        if len(set(target)) != 1 or len(set(rings)) != 1:
            raise ValueError('unaligned state within an observation')
        return [target[0], rings[0]]
    left, right = frontiers(actual), frontiers(reference)
    for pair, (target, ring) in zip((actual, reference), (left, right)):
        if 'token_ids' not in pair or len(pair['token_ids']) != target or not 0 <= ring <= target:
            raise ValueError('observation lacks complete canonical consumed identity')
    identity_equal = tuple(actual['token_ids']) == tuple(reference['token_ids'])
    out = dict(boundary=boundary, frontiers=[left, right], comparable=left == right and identity_equal,
               token_identity_equal=identity_equal,
               status='OBSERVED_ONLY')
    valid_deltas = (0, 1) if boundary == 'p5-terminal' else (0,)
    if any(t-r not in valid_deltas for t, r in (left, right)):
        out['comparable'] = False
    if out['comparable']:
        out['metrics'] = paired_physical_metrics(actual, reference, config=config,
                                                 chunk_elements=chunk_elements)
    return out


def observe_case_a_n2(backend, request, tokenizer, *, identity):
    """Single cold-text experiment using the existing native capture callback.

    The task ends at capture via normal exception unwinding, before N-1/P5.
    No production state is advanced for observation. Reference runs only after
    _start has restored its temporary layer wrappers and released its lease.
    Caller owns the admitted backend/model, external supervision and cleanup.
    """
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from ds41f_mlx.runtime.mtp_resources import MTPWiredLimitLease
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpoint

    ids = tuple(request.token_ids)
    if request.multimodal is not None or not 3 <= len(ids) <= 4096:
        raise ValueError('Case A requires bounded canonical text only')
    lm = backend._model.language_model
    original_layers = tuple(lm.layers)
    saved = None
    class CaptureComplete(Exception):
        pass
    def capture(target, rings, consumed):
        nonlocal saved
        if tuple(consumed) != ids[:-2]:
            raise ValueError('candidate capture is not canonical N-2')
        saved = PairedCheckpoint.capture(identity, target, rings, consumed, mx)
        print(f'Case A: captured native N-2 frontier {len(consumed)}', flush=True)
        raise CaptureComplete()
    rec = QualificationSession('phase2-step1a')
    try:
        backend._start(rec, request, tokenizer, {}, checkpoint_capture=capture)
    except CaptureComplete:
        pass
    if saved is None or rec.owner is not None:
        raise RuntimeError('N-2 capture did not stop before native generation')
    if any(a is not b for a, b in zip(lm.layers, original_layers)):
        raise RuntimeError('candidate layer wrappers were not restored')
    candidate = dict(target=saved.target, rings=saved.rings, token_ids=saved.tokens)
    print('Case A: candidate unwound; fresh ordinary reference', flush=True)
    with mx.stream(generation_stream), MTPWiredLimitLease(mx, generation_stream):
        reference = ordinary_prefill_reference(model=backend._model,
                                               input_ids=mx.array([ids[:-2]], mx.int32))
        result = paired_boundary_metrics(candidate, reference, boundary='n-2-capture', config=lm._config)
        mx.synchronize(generation_stream)
    return result


def layer0_window_observation(cache, config):
    """Copy live, already committed L0 bytes; never mistake a checkpoint for live state."""
    import numpy as np
    import mlx.core as mx
    from omlx.patches.deepseek_v41.quantization import unpack_activation
    end, value = cache.size(), cache.cache[1]
    if not 0 < end <= config.window_size or value.shape != (1, end, config.head_dim + config.head_dim//32) or value.dtype != mx.uint8:
        raise ValueError('Layer 0 observer requires a short unpadded FP8 window')
    for name in ('left_padding', 'lengths'):
        v = getattr(cache, name, None)
        if v is not None and (v.size != 1 or int(v.item()) != (0 if name == 'left_padding' else end)):
            raise ValueError('Layer 0 observer requires unpadded committed state')
    packed = np.array(value, copy=True)
    decoded = np.array(unpack_activation(value, dtype=mx.float32), copy=True)
    return dict(packed=packed, decoded=decoded, frontier=end,
                absolute_positions=list(range(end)), meta_state=list(cache.meta_state),
                compress_ratio=cache.compress_ratio,
                offset=np.asarray(cache.cache[0]).tolist(),
                left_padding=None if cache.left_padding is None else np.asarray(cache.left_padding).tolist(),
                lengths=None if cache.lengths is None else np.asarray(cache.lengths).tolist(),
                memory=dict(active=mx.get_active_memory(), cache=mx.get_cache_memory(), peak=mx.get_peak_memory()))


def layer0_prefix_metrics(a, b, rows):
    if rows <= 0 or any(o['absolute_positions'][:rows] != list(range(rows)) for o in (a, b)):
        raise ValueError('prefix rows are not the same absolute positions')
    return {key: physical_tensor_metrics(a[key][:, :rows], b[key][:, :rows])
            for key in ('packed', 'decoded')}


@contextmanager
def layer0_numerical_handles(lm):
    """Temporary installed-oMLX L0 hooks: retain original handles, no tensor ops.

    No eval, slicing, copies or host reads inside calls. Read handles only after
    the original execution has synchronized/unwound. Retention extends graph
    lifetime; compiled internals are deliberately not instrumented. This focused
    diagnostic is removable, not a general tracer or a numerical implementation.
    """
    import omlx.patches.deepseek_v41.language as lang
    layer = lm.layers[0]
    # The admitted Flash-1017 checkpoint has Engram at layers 1/14, not 0.
    # Tiny controls may include it at L0; never require an absent operation.
    engram, attn = getattr(layer, 'engram', None), layer.attn
    projections = {id(attn.wkv): 'attention_wkv'}
    if engram is not None:
        projections[id(engram.wkv)] = 'engram_wkv'
    handles, active = {}, [False]
    def keep(name, value):
        if name in handles:
            raise ValueError('unexpected repeated Layer 0 observation: ' + name)
        handles[name] = value
    with ExitStack() as stack:
        def wrap(cls, name, factory):
            original = getattr(cls, name)
            stack.enter_context(patch.object(cls, name, factory(original)))
        def block_call(original):
            def call(self, h, pre, *args, **kwargs):
                if self is layer:
                    keep('block_input', h)
                    keep('block_pre', pre)
                return original(self, h, pre, *args, **kwargs)
            return call
        def engram_call(original):
            def call(self, h, *args, **kwargs):
                out = original(self, h, *args, **kwargs)
                if self is engram:
                    keep('engram_input', h)
                    keep('engram_output', out)
                return out
            return call
        def projection_call(original):
            def call(self, x, *args, **kwargs):
                out = original(self, x, *args, **kwargs)
                if id(self) in projections:
                    keep(projections[id(self)] + '_input', x)
                    if not isinstance(self, lang.QuantizedProjection):
                        keep(projections[id(self)] + '_output', out)
                return out
            return call
        def project(original):
            def call(self, x, *args, **kwargs):
                out = original(self, x, *args, **kwargs)
                if id(self) in projections:
                    keep(projections[id(self)] + '_qdq', x)
                    keep(projections[id(self)] + '_output', out)
                return out
            return call
        def attention_call(original):
            def call(self, x, *args, **kwargs):
                if self is not attn:
                    return original(self, x, *args, **kwargs)
                keep('attention_input', x)
                active[0] = True
                try:
                    return original(self, x, *args, **kwargs)
                finally:
                    active[0] = False
            return call
        def norm_call(original):
            def call(self, x):
                out = original(self, x)
                if self is attn.kv_norm:
                    keep('kv_norm_output', out)
                return out
            return call
        pack = lang.pack_activation
        def observed_pack(x, *args, **kwargs):
            out = pack(x, *args, **kwargs)
            if active[0] and x.shape[1]:
                keep('prepack', x)
            return out
        if isinstance(layer, lang.Block):
            wrap(type(layer), '__call__', block_call)
        if engram is not None:
            wrap(type(engram), '__call__', engram_call)
            if not isinstance(engram.wkv, lang.QuantizedProjection):
                wrap(type(engram.wkv), '__call__', projection_call)
        wrap(lang.QuantizedProjection, '__call__', projection_call)
        wrap(lang.QuantizedProjection, 'project_quantized', project)
        wrap(type(attn), '__call__', attention_call)
        wrap(type(attn.kv_norm), '__call__', norm_call)
        stack.enter_context(patch.object(lang, 'pack_activation', observed_pack))
        yield handles


def observe_case_c_n2(backend, seed_request, request, tokenizer, *, identity, layer0=False):
    """Diagnostic paired restore + positive append, NOT a Scheduler lookup.

    Cold seed execution only constructs the checkpoint. The ordinary reference
    consumes the entire current N-2 prefix on fresh factories, never the pair.
    Optional Step 1E mode returns live L0 observations and original tensor handles
    instead of repeating the all-layer comparison. Caller owns handle retirement;
    this is focused observation, not general numerical qualification.
    """
    import numpy as np
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from ds41f_mlx.runtime.mtp_resources import MTPWiredLimitLease
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpoint, validate_pair
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend

    seed_ids, ids = tuple(seed_request.token_ids), tuple(request.token_ids)
    C, end = len(seed_ids)-2, len(ids)-2
    if any(r.multimodal is not None for r in (seed_request, request)) or not 0 < C < end <= 4094:
        raise ValueError('Case C requires bounded text and a positive missing suffix')
    if ids[:C] != seed_ids[:C]:
        raise ValueError('Case C incompatible canonical prefix')
    lm = backend._model.language_model
    original_layers = tuple(lm.layers)
    executions = []
    execute = DeferredPrefillAppend.execute_all
    def observed_execute(app):
        executions.append(dict(start=app.C, end=app.T, rows=app.T-app.C))
        return execute(app)
    class CaptureComplete(Exception):
        pass
    observations = {}
    def run(rec, req, label):
        saved = None
        def capture(target, rings, consumed):
            nonlocal saved
            if tuple(consumed) != tuple(req.token_ids[:-2]):
                raise ValueError('candidate capture is not canonical N-2')
            if layer0:
                observations[label] = dict(window=layer0_window_observation(target[0], lm._config),
                    pair_logical_bytes=sum(v.nbytes for c in target for v in (*c.cache, c.left_padding, c.lengths) if v is not None)
                        + sum(r.keys.nbytes for r in rings if r.keys is not None))
            saved = PairedCheckpoint.capture(identity, target, rings, consumed, mx)
            print(f'Case C: captured N-2 frontier {len(consumed)}', flush=True)
            raise CaptureComplete()
        with layer0_numerical_handles(lm) if layer0 else nullcontext() as handles:
            try:
                backend._start(rec, req, tokenizer, {}, checkpoint_capture=capture)
            except CaptureComplete:
                pass
        if layer0:
            observations[label]['handles'] = handles
        if saved is None or rec.owner is not None or any(a is not b for a, b in zip(lm.layers, original_layers)):
            raise RuntimeError('Case C capture/unwind failed')
        return saved
    def fingerprint(saved):
        # Bounded exact value digest; BF16->FP32 is lossless, not normalization.
        h = hashlib.sha256()
        h.update(repr((saved.identity, saved.tokens, saved.images,
                       [(c.meta_state, c.compress_ratio) for c in saved.target],
                       [(r.max_size, r.offset) for r in saved.rings])).encode())
        for value in [v for c in saved.target for v in (*c.cache, c.left_padding, c.lengths)] + [r.keys for r in saved.rings]:
            h.update(repr(None if value is None else (value.shape, str(value.dtype))).encode())
            if value is not None:
                flat = value.reshape(-1)
                for start in range(0, flat.size, 65536):
                    chunk = flat[start:start+65536]
                    if str(chunk.dtype).endswith('bfloat16'):
                        chunk = chunk.astype(mx.float32)
                    h.update(np.asarray(chunk).tobytes())
        return h.hexdigest()
    with patch.object(DeferredPrefillAppend, 'execute_all', observed_execute):
        seed_rec = QualificationSession('phase2-step1c-seed')
        seed = run(seed_rec, seed_request, 'A')
        seed_rec.cache = seed_rec.rings = None
        before = fingerprint(seed)
        rec = QualificationSession('phase2-step1c-append')
        rec.cache, rec.rings, rec.canonical = seed.restore(identity, ids, mx)
        validate_pair(rec.cache, rec.rings, rec.canonical)
        if len(rec.canonical) != C or tuple(rec.canonical) != ids[:C]:
            raise ValueError('incorrect restored frontier')
        distinct = all(a is not b and a.cache is not b.cache for a, b in zip(rec.cache, seed.target)) and all(a is not b for a, b in zip(rec.rings, seed.rings))
        if not distinct:
            raise ValueError('restored mutable containers alias checkpoint')
        if layer0:
            observations['B0'] = dict(window=layer0_window_observation(rec.cache[0], lm._config))
            restored_prefix = layer0_prefix_metrics(observations['A']['window'], observations['B0']['window'], C)
            if any(m['mismatches'] for m in restored_prefix.values()):
                return dict(classifications=['UNRESOLVED', 'PARTIAL'],
                            comparisons=dict(A_B0=restored_prefix), observations=observations,
                            executions=dict(A=list(executions), B1=[]))
            restore_metrics = None
        else:
            restore_metrics = paired_physical_metrics(
                dict(target=rec.cache, rings=rec.rings), dict(target=seed.target, rings=seed.rings))
        seed_executions = list(executions)
        executions.clear()
        candidate = run(rec, request, 'B1')
        after = fingerprint(seed)
        if before != after:
            raise ValueError('seed checkpoint mutated during append')
    if executions != [dict(start=C, end=end, rows=end-C)]:
        raise ValueError('unexpected append execution boundaries')
    if layer0:
        comparisons = {name: layer0_prefix_metrics(observations[a]['window'], observations[b]['window'], C)
                       for name, a, b in (('A_B0', 'A', 'B0'), ('B0_B1', 'B0', 'B1'), ('A_B1', 'A', 'B1'))}
        result = dict(comparisons=comparisons, executions=dict(A=seed_executions, B1=list(executions)),
                      restore=dict(identity=seed.identity, distinct_mutable_containers=distinct,
                                   snapshot_before=before, snapshot_after=after))
        if any(m['packed']['mismatches'] for m in comparisons.values()):
            result.update(classifications=['PREFIX_MODIFIED', 'PARTIAL'], observations=observations)
            return result
        with patch.object(DeferredPrefillAppend, 'execute_all', observed_execute):
            executions.clear()
            cold_rec = QualificationSession('phase2-step1e-cold16')
            cold = run(cold_rec, request, 'C')
            result['executions']['C'] = list(executions)
        with mx.stream(generation_stream), MTPWiredLimitLease(mx, generation_stream):
            with layer0_numerical_handles(lm) as handles:
                reference = ordinary_prefill_reference(backend._model, mx.array([ids[:-2]], mx.int32))
                mx.synchronize(generation_stream)
            observations['D'] = dict(window=layer0_window_observation(reference['target'][0], lm._config), handles=handles)
        for name, a, b in (('A_C', 'A', 'C'), ('C_D', 'C', 'D'), ('B1_C', 'B1', 'C')):
            comparisons[name] = layer0_prefix_metrics(observations[a]['window'], observations[b]['window'], C)
        result.update(observations=observations, classifications=['PREFIX_PRESERVED'])
        if comparisons['A_C']['packed']['mismatches']:
            result['classifications'].append('COLD_SHAPE_DIFFERENCE')
        if comparisons['C_D']['packed']['mismatches']:
            result['classifications'].append('P6_ORDINARY_DIFFERENCE')
        return result
    print('Case C: append unwound; fresh full-current-prefix ordinary reference', flush=True)
    with mx.stream(generation_stream), MTPWiredLimitLease(mx, generation_stream):
        reference = ordinary_prefill_reference(backend._model, mx.array([ids[:-2]], mx.int32))
        result = paired_boundary_metrics(
            dict(target=candidate.target, rings=candidate.rings, token_ids=candidate.tokens),
            reference, boundary='n-2-capture', config=lm._config)
        mx.synchronize(generation_stream)
    result['restore'] = dict(kind='diagnostic PairedCheckpoint.restore', identity=seed.identity,
        saved_frontier=len(seed.tokens), restored_frontier=C, appended_length=end-C,
        distinct_mutable_containers=distinct, snapshot_before=before, snapshot_after=after,
        snapshot_unchanged=before == after, seed_executions=seed_executions,
        append_executions=executions, metrics=restore_metrics)
    return result


def observe_case_b_n2(backend, request, tokenizer, *, identity, source_bytes):
    """Single-image observation; direct _start, NOT Scheduler/HTTP acceptance.

    Reuse the production span encoder unchanged. Passive wrappers retain its
    rows and compare its CPU patches to independently prepared reference pixels.
    Reference never receives candidate embeddings, taps or cache state.
    """
    from hashlib import sha256
    from io import BytesIO
    import mlx.core as mx
    from PIL import Image
    from mlx_lm.generate import generation_stream
    from ds41f_mlx.model_execution import processing
    from ds41f_mlx.runtime.mtp_resources import MTPWiredLimitLease
    from ds41f_mlx.serving.internal_mtp import QualificationSession
    from ds41f_mlx.serving.paired_checkpoint import PairedCheckpoint
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend

    mm, ids = request.multimodal, tuple(request.token_ids)
    if mm is None or len(mm.spans) != 1 or not 3 <= len(ids) <= 4096:
        raise ValueError('Case B requires one bounded complete image')
    span = mm.spans[0]
    if (tuple(mm.token_ids) != ids or span.start < 0 or
            span.start + span.length > len(ids)-2 or
            span.length != len(span.kinds) or
            any(t == mm._config.image_token_id for t in ids[-2:]) or
            any(t != mm._config.image_token_id for t in ids[span.start:span.start+span.length])):
        raise ValueError('Case B image/canonical N-2 boundary mismatch')
    if sha256(source_bytes).hexdigest() != span.digest or mm._sources.get(span.start) != source_bytes:
        raise ValueError('Case B independent source identity mismatch')
    if getattr(backend._encode_new_images, '__func__', None) is not ProductionMTPBackend._encode_new_images:
        raise ValueError('Case B requires the existing production image encoder')
    # Separate decode/resize/normalize/patch construction from verified bytes.
    with Image.open(BytesIO(source_bytes)) as image:
        patches, h, w, kinds = processing.image_patch_array(image, mm._config)
    if (h, w) != span.grid or tuple(kinds) != span.kinds:
        raise ValueError('Case B independent geometry mismatch')
    reference_pixels = mx.array(patches).astype(mx.bfloat16)
    pixels_observed, candidate_rows = [], []
    original_patch = processing.image_patch_array
    original_encoder = backend._encode_new_images
    def observe_patches(*args, **kwargs):
        result = original_patch(*args, **kwargs)
        value, ch, cw, ck = result
        pixels_observed.append(dict(
            geometry_equal=(ch, cw) == (h, w) and tuple(ck) == tuple(kinds),
            patches=physical_tensor_metrics(value, patches),
            prepared_bfloat16=physical_tensor_metrics(mx.array(value).astype(mx.bfloat16), reference_pixels)))
        return result
    def observe_encoder(*args, **kwargs):
        result = original_encoder(*args, **kwargs)
        candidate_rows.extend(result.rows)
        return result
    lm = backend._model.language_model
    original_layers, saved = tuple(lm.layers), None
    class CaptureComplete(Exception):
        pass
    def capture(target, rings, consumed):
        nonlocal saved
        if tuple(consumed) != ids[:-2]:
            raise ValueError('candidate capture is not canonical N-2')
        saved = PairedCheckpoint.capture(identity, target, rings, consumed, mx, images=mm.identities())
        print(f'Case B: captured native N-2 frontier {len(consumed)}', flush=True)
        raise CaptureComplete()
    rec = QualificationSession('phase2-step1b')
    trace = {}
    with patch.object(processing, 'image_patch_array', observe_patches), patch.object(backend, '_encode_new_images', observe_encoder):
        try:
            backend._start(rec, request, tokenizer, trace, checkpoint_capture=capture)
        except CaptureComplete:
            pass
    if saved is None or rec.owner is not None or any(a is not b for a, b in zip(lm.layers, original_layers)):
        raise RuntimeError('Case B capture/unwind did not restore native boundary')
    if len(pixels_observed) != 1 or len(candidate_rows) != 1 or candidate_rows[0][0] != span.start:
        raise RuntimeError('Case B did not observe exactly one production Vision span')
    print('Case B: candidate unwound; independent image ordinary reference', flush=True)
    candidate = dict(target=saved.target, rings=saved.rings, token_ids=saved.tokens)
    with mx.stream(generation_stream), MTPWiredLimitLease(mx, generation_stream):
        reference = ordinary_prefill_reference(backend._model, mx.array([ids[:-2]], mx.int32),
            pixel_values=reference_pixels, image_grids=[(h, w)],
            image_spans=[(span.start, span.length)], image_types=[list(kinds)])
        result = dict(image=dict(identity=span.identity(), kinds=list(kinds),
            absolute_positions=list(range(span.start, span.start+span.length)),
            mask_positions=[i for i, t in enumerate(ids[:-2]) if t == mm._config.image_token_id],
            patches=pixels_observed[0], embeddings_by_kind=[]), trace=trace)
        a, b = candidate_rows[0][1], reference['embeddings'][0, span.start:span.start+span.length]
        result['image']['embeddings'] = physical_tensor_metrics(a, b)
        for kind in sorted(set(kinds)):
            positions = [i for i, k in enumerate(kinds) if k == kind]
            ix = mx.array(positions, mx.int32)
            result['image']['embeddings_by_kind'].append(dict(kind=kind,
                absolute_positions=[span.start+i for i in positions],
                metrics=physical_tensor_metrics(a[ix], b[ix])))
        result['boundary'] = paired_boundary_metrics(candidate, reference,
            boundary='n-2-capture', config=lm._config)
        mx.synchronize(generation_stream)
    return result


def qualify(args, report):
    import mlx.core as mx
    import omlx.patches.deepseek_v41.language as lang
    from omlx.patches.deepseek_v41.loading import load
    report['runtime_authority'] = runtime_authority(lang)
    incompatible = {k: v for k, v in report['runtime_authority']['operation_contracts'].items() if v != 'compatible'}
    if incompatible:
        raise RuntimeError(f'oMLX API drift: {incompatible}')
    print(json.dumps({'runtime_authority': report['runtime_authority']}), flush=True)
    report['stage'] = 'checkpoint_load'
    model = load(args.checkpoint, preserve_mtp=False, engram_ssd_offload=True)[0]
    try:
        lm = model.language_model
        report['stage'] = 'planner_and_setup'
        with tempfile.TemporaryDirectory(prefix='ds41f-qualification-') as tmp:
            planner = SweepPlanner(load_native_prefill_library(compile_native_prefill_library(Path(tmp))))
            plan = planner.build_from_model(lm, ctx=32768, remaining=8192)
        setup = DwarfStarFP8MLXPrefillExecutorSetup(lm, mx=mx).prepare(plan, [1] * 8192)
        arena, runner = setup.arena, setup.block_runner
        cache20 = runner.working_cache[20]
        attn20 = lm.layers[20].attn
        role = 'command'
        calls = {'command_compressor': [], 'reference_compressor': [], 'command_index_wk': [], 'reference_index_wk': [], 'ordinary_decoder_blocks': []}
        report['operation_calls'] = calls
        original_compressor = type(attn20.compressor).__call__
        original_wk = type(attn20.indexer.wk).__call__
        original_block = lang.Block.__call__
        def compressor(obj, x, cache, start):
            if obj is attn20.compressor:
                calls[role + '_compressor'].append({'start': start, 'rows': int(x.shape[1])})
            return original_compressor(obj, x, cache, start)
        def wk(obj, x, *a, **kw):
            if obj is attn20.indexer.wk:
                calls[role + '_index_wk'].append(int(x.shape[1]))
            return original_wk(obj, x, *a, **kw)
        def block(obj, *a, **kw):
            if obj.attn._layer >= 20:
                calls['ordinary_decoder_blocks'].append(obj.attn._layer)
                raise RuntimeError('ordinary Block fallback during suffix qualification')
            return original_block(obj, *a, **kw)
        def exact(actual, expected, boundary):
            report['stage'] = boundary
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(f'{boundary}: geometry/dtype mismatch')
            mismatches = int(mx.sum(actual != expected).item())
            if mismatches:
                report['first_divergence'] = {'boundary': boundary, 'mismatched_bytes': mismatches}
                raise RuntimeError(f'{boundary}: {mismatches} mismatched packed bytes')
            return {'shape': list(actual.shape), 'dtype': str(actual.dtype), 'mismatched_bytes': 0}
        window_checks = []
        report['window_checks'] = window_checks
        source_reference = None
        seen_layers = set()
        with ExitStack() as stack:
            stack.enter_context(patch.object(type(attn20.compressor), '__call__', compressor))
            stack.enter_context(patch.object(type(attn20.indexer.wk), '__call__', wk))
            stack.enter_context(patch.object(lang.Block, '__call__', block))
            for command in plan.commands:
                report['stage'] = f'command:{command.index}:{command.kind.value}:layer{command.layer}'
                full_prepare = command.kind is SweepCommandKind.DECODER_PREPARE_SUFFIX and arena.decoder_prepare_role(command) == 'decoder_full_source_publish'
                local_prepare = command.kind is SweepCommandKind.DECODER_PREPARE_SUFFIX and not full_prepare
                before_frontier = None
                if local_prepare:
                    layer = lm.layers[command.layer]
                    cache = runner.working_cache[command.layer]
                    before_frontier = int(cache[0].item()) if cache[0] is not None else 0
                    offset, rows = command.offset, command.rows
                    h = arena.carry.current.value[:, offset:offset+rows]
                    pre = arena.carry.pre.value[:, offset:offset+rows]
                    x = lang.hc_pre_norm(h, pre, layer.attn_norm.weight, layer.attn_norm.eps)
                    _, kv_input = layer.attn._input_projections(x)
                    expected_window = lang.pack_activation(lang.rope(layer.attn.kv_norm(kv_input), mx.arange(offset, offset+rows), lm._config, bool(lm._config.compress_ratios[command.layer])))
                    mx.eval(expected_window)
                    # Deliberately non-contiguous distinctive prior state. The full
                    # replacement must equal the independent current-row projection.
                    cache[1] = mx.full((1, 128, expected_window.shape[-1]), 255, dtype=mx.uint8)
                    mx.eval(cache[1])
                runner.execute_command(command, arena)
                mx.eval(arena.carry.current.value, arena.carry.next.value, arena.carry.pre.value,
                        *[v for cache in runner.working_cache for v in cache.cache if v is not None])
                if command.kind is SweepCommandKind.ENCODE_ROWS:
                    seen_layers.add(command.layer)
                if full_prepare:
                    if cache20[0] is not None and int(cache20[0].item()) != 0:
                        raise RuntimeError('full-source prepare advanced logical frontier')
                    pending = setup.publication_manager.pending_cumulative_by_layer[20]
                    report['full_prepare_has_row_spans'] = bool(setup.publication_manager.pending_spans_by_layer.get(20))
                    assert not report['full_prepare_has_row_spans']
                    assert setup.publication_manager.visible_cumulative['kv'].layer != 20
                    assert pending['kv'].value is cache20[2]
                    report['producer_private_before_frontier'] = True
                    # Existing ordinary Attention is the reference source path. Its
                    # full query output is not evaluated or used as a decoder oracle.
                    role = 'reference'
                    reference_cache = lm.make_cache()[20]
                    x = lang.hc_pre_norm(arena.encoder_final_h, arena.encoder_final_pre, lm.layers[20].attn_norm.weight, lm.layers[20].attn_norm.eps)
                    reference_shared = {}
                    unused_output = attn20(x, reference_cache, reference_shared, 0)
                    mx.eval(reference_cache[2], reference_cache[3])
                    source_reference = reference_cache[2], reference_cache[3]
                    del unused_output, reference_shared, reference_cache, x
                    role = 'command'
                    report['source_comparison'] = {
                        'reference': 'loaded ordinary Attention.__call__, same full encoder-final pre-norm input, start=0',
                        'compressed_kv': exact(cache20[2], source_reference[0], 'source_compressor'),
                        'index_k': exact(cache20[3], source_reference[1], 'index_k'),
                    }
                if local_prepare:
                    after_frontier = int(cache[0].item()) if cache[0] is not None else 0
                    assert after_frontier == before_frontier
                    result = exact(cache[1], expected_window, f'local_window_preparation:layer{command.layer}')
                    window_checks.append({'layer': command.layer, 'offset': command.offset, 'rows': command.rows, 'stale_seed': 255, 'frontier_unchanged': True, **result})
                    del expected_window, h, pre, x, kv_input
                if command.kind is SweepCommandKind.ENCODE_ROWS and command.layer == 20:
                    exact(cache20[2], source_reference[0], 'layer20_query_preserves_compressed_kv')
                    exact(cache20[3], source_reference[1], 'layer20_query_preserves_index_k')
        report['stage'] = 'final_invariants'
        frontiers = [int(c[0].item()) for c in runner.working_cache]
        assert frontiers == [8192] * 40 and seen_layers == set(range(40))
        assert calls['command_compressor'] == [{'start': 0, 'rows': 8192}]
        assert calls['command_index_wk'] == [8192]
        assert calls['ordinary_decoder_blocks'] == []
        assert runner.suffix_math.full_source_generation_count == 1
        assert not runner.prefill_continuation_exported and runner.full_cache_repack_count == 0
        assert runner.final_logits_suppressed and runner.final_logits is None
        assert all(int(c[1].shape[1]) == 128 for c in runner.working_cache)
        assert all(c.shape == (1, 0, 512) and c.dtype == mx.bfloat16 for c in cache20.cache[4:6])
        first_queries = {}
        for layer, start, old_length in runner.suffix_math.query_old_lengths:
            first_queries.setdefault(layer, {'absolute_start': start, 'old_length': old_length})
        assert len(first_queries) == 20 and all(v['old_length'] == 127 for v in first_queries.values())
        report.update(status='PASS', cache_frontiers=frontiers,
                      layer20_slots=[geometry(v) for v in cache20.cache],
                      all_cache_slots=[[geometry(v) for v in c.cache] for c in runner.working_cache],
                      first_suffix_queries=first_queries,
                      full_source_generation_count=runner.suffix_math.full_source_generation_count,
                      PrefillContinuationState_export_count=0, full_cache_repack_count=0,
                      final_prefix_logits_suppressed=True,
                      executed_commands=len(runner.records), executed_layers=sorted(seen_layers))
    finally:
        model.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = {'status': 'FAIL', 'structural_and_bounded_same_input_only': True,
              'checkpoint': str(args.checkpoint), 'tokens': 8192,
              'load_kwargs': {'preserve_mtp': False, 'engram_ssd_offload': True},
              'comparison_policy': 'Exact packed bytes only for same-input identical loaded source/window operators; no cross-topology hidden/logit equality gate.'}
    try:
        qualify(args, report)
    except Exception as exc:
        report['reason'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'stage': report.get('stage'), 'evidence': str(args.out)}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
