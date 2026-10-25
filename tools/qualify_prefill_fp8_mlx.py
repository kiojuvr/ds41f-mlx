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
from contextlib import ExitStack
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
    out['structure_equal'] = actual.shape == reference.shape and actual.dtype == reference.dtype
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
            if value is not None and slot == 6 and (value.shape[1] != count or value.dtype != mx.int64):
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
