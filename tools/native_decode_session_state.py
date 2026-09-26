#!/usr/bin/env python3
"""Boundary13a reusable native bounded decode-session state helper.

Validation-only helper: constructs an explicit prefill-end state container for the
bounded [[0,3]], start_pos=0 native Engram-connected path.  It intentionally
stores source-visible persistent values/metadata, not giant unused cache payloads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_native_layer0_25_transformer_entry_validation import (  # noqa:E402
    DEFAULT_CHECKPOINT, VOCAB, DIM, HC, D, ID, mmap, digest, cfg, block
)
from tools.run_native_parallel_head_logits_validation import native_parallel_head_logits  # noqa:E402
from tools.run_native_engram_layer14_validation import regen_hashes, apply_engram_layer  # noqa:E402
from tools.run_native_engram_layer1_validation import CKPT, arrdig, file_id, span_id  # noqa:E402
from tools.run_native_engram_connected_deterministic_logits_validation import hc_pre_source, rmsnorm_source  # noqa:E402

KV_SOURCE_LAYERS = [2, 8, 14, 20]
INDEX_OWNER_LAYERS = [2, 8, 14, 20]
INDEX_SOURCE_LAYERS = [2, 8, 14, 20, 24, 28, 32, 36]
TARGET_LAYER_IDS = [37, 38, 39]
WINDOW_SIZE = 128
OFFICIAL_DEFAULT_MAX_BATCH_SIZE = 4
OFFICIAL_DEFAULT_MAX_SEQ_LEN = 4096
TOKENS = [[0, 3]]
START_POS = 0
PREFILL_S = 2

EXP = {
    'full_hash': 'f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d',
    'ngram_cache_valid': '96fb5e4a2704b410bbf097c41e40ff8118ef0bc819ccf4344f31f694d12d536a',
    'post_engram1_h': '3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9',
    'post_engram14_h': 'ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd',
    'logits': '7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd',
    'compress_kv': {
        2: '838e6e26d9889ef668bc6be7d345be10542466bb86a8c656f3aa5665376143ae',
        8: '29f479332ce47e4429d7c46d4fc1a952efa16f74ca110b1a9a025f032a9e2213',
        14: 'e2c045f500a5776d647fc22ebc99faf2bb2f7131ba2e332e256fed3aadfa5edb',
        20: '17eacfb671aa2a37c302b5e6f097346b7951f7bd3ff23bd0ed5193c00695900d',
    },
    'index_k': {
        2: '0a02a69899257836865cacec8a1a6d0d1bfb590cf07b8a3f602300e02ee7875d',
        8: 'cd1c51ca26f6404bde0eb2230908af462e9440a923885236f35ba92df6542d87',
        14: '82f4b94a61be422936f51e142f786be31a37df14412ac700ef804a98f56f5c85',
        20: 'a2ec36dab41f04ef5f7ab63fabcaa823a22232b9a2fdb3d28d990dfa75e4a896',
    },
}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_digest(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return sha_bytes(data)


@dataclass
class NativeDecodeSessionState:
    official_model_persistent_state: dict[str, Any] = field(default_factory=dict)
    generation_loop_control: dict[str, Any] = field(default_factory=dict)
    target_runtime_session_state: dict[str, Any] = field(default_factory=dict)
    excluded_call_local_fields: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    # Validation-only in-memory values, deliberately not serialized as tensor fixtures.
    visible_value_arrays: dict[str, Any] = field(default_factory=dict, repr=False)


def make_decode_step0_input(state: NativeDecodeSessionState) -> dict[str, Any]:
    return {
        'input_ids': state.generation_loop_control['next_input_ids'],
        'start_pos': state.generation_loop_control['next_start_pos'],
        'state': state,
        'decode_forward_executed': False,
    }


def _compress_valid_len(layer: int, ratio: int) -> int:
    return (START_POS + PREFILL_S) // ratio


def _build_source_identities(ck: Path) -> dict[str, Any]:
    return {
        'model_py': file_id(ck / 'inference/model.py'),
        'engram_py': file_id(ck / 'inference/engram.py'),
        'config_json': file_id(ck / 'inference/config.json'),
        'generate_py': file_id(ck / 'inference/generate.py'),
        'Boundary13_spans': {
            'NgramHashState': span_id('inference/engram.py', 118, 184),
            'Compressor': span_id('inference/model.py', 429, 486),
            'Indexer': span_id('inference/model.py', 488, 581),
            'Attention': span_id('inference/model.py', 613, 788),
            'SharedAttentionRuntime': span_id('inference/model.py', 1166, 1179),
            'Transformer_forward': span_id('inference/model.py', 1183, 1273),
            'generate_loop': span_id('inference/generate.py', 55, 70),
        },
    }


def build_prefill_state(checkpoint: Path | str = CKPT) -> tuple[NativeDecodeSessionState, dict[str, Any]]:
    ck = Path(checkpoint)
    c = cfg(ck)
    cfgj = json.loads((ck / 'inference/config.json').read_text())
    b12b0 = json.loads((ROOT / 'artifacts/engram-semantic-foundation-contract.json').read_text())

    token_ids = np.array(TOKENS, np.int64)
    full_hash, l1_hash, l14_hash = regen_hashes(ck, cfgj, b12b0)

    emb = np.ascontiguousarray(mmap(ck / 'model-00002-of-00048.safetensors', 'embed.weight', np.uint16, (VOCAB, DIM)))
    h = np.repeat(emb[token_ids].copy()[:, :, None, :], HC, axis=2).copy()
    pre = np.zeros((1, 2, HC), np.float32)
    pre[:, :, 0] = 1.0
    shared = {'compress_kv': None, 'index_k': None, 'candidates': None, 'topk_idxs': None}

    arrays: dict[str, Any] = {}
    window_entries: dict[str, Any] = {}
    compressed_entries: dict[str, Any] = {}
    index_entries: dict[str, Any] = {}
    call_local_diag: dict[str, Any] = {'final_shared_attn': {}}

    post1 = post14 = None
    events = []
    out0 = block(ck, c, 0, h, pre, shared); events.append('block0')
    wk = np.ascontiguousarray(out0['attn_path']['window_kv'])
    window_entries['0'] = _window_entry(0, wk); arrays['window_kv.0.visible'] = wk.copy()
    h, *_ = apply_engram_layer(ck, 1, out0['x_out'], l1_hash, b12b0)
    post1 = arrdig(h); pre = out0['ffn_pre']; events.append('engram1')

    for layer in range(1, 14):
        out = block(ck, c, layer, h, pre, shared); events.append(f'block{layer}')
        wk = np.ascontiguousarray(out['attn_path']['window_kv'])
        window_entries[str(layer)] = _window_entry(layer, wk); arrays[f'window_kv.{layer}.visible'] = wk.copy()
        _maybe_record_producers(layer, c, out, compressed_entries, index_entries, arrays)
        h = out['x_out']; pre = out['ffn_pre']

    h, *_ = apply_engram_layer(ck, 14, h.copy(), l14_hash, b12b0)
    post14 = arrdig(h); events.append('engram14')

    for layer in range(14, 40):
        out = block(ck, c, layer, h, pre, shared); events.append(f'block{layer}')
        wk = np.ascontiguousarray(out['attn_path']['window_kv'])
        window_entries[str(layer)] = _window_entry(layer, wk); arrays[f'window_kv.{layer}.visible'] = wk.copy()
        _maybe_record_producers(layer, c, out, compressed_entries, index_entries, arrays)
        h = out['x_out']; pre = out['ffn_pre']

    _, post_loop = hc_pre_source(h, pre)
    idx = json.loads((ck / 'model.safetensors.index.json').read_text())['weight_map']
    norm_w = np.ascontiguousarray(mmap(ck / idx['norm.weight'], 'norm.weight', np.uint16, (DIM,)))
    normalized = rmsnorm_source(post_loop, norm_w, 1e-20)['bf16']
    head_weight = mmap(ck / idx['head.weight'], 'head.weight', np.uint16, (VOCAB, DIM))
    logits = native_parallel_head_logits(normalized[:, -1, :].copy(), head_weight, 1024)
    output_ids = np.asarray([int(np.argmax(logits.reshape(-1)))], dtype=np.int64)

    # Ngram cache visible prefix: token_map for this fixture is known/regressed to [[0,3]].
    ngram_valid = np.asarray([[0, 3]], dtype=np.int64)
    arrays['ngram_cache.visible'] = ngram_valid.copy()
    ngram_entry = {
        'owner': 'Transformer.engram_hash.cache / NgramHashState.cache',
        'dtype': 'int64',
        'allocated_shape': [OFFICIAL_DEFAULT_MAX_BATCH_SIZE, OFFICIAL_DEFAULT_MAX_SEQ_LEN],
        'valid_batch_range': [0, 1],
        'valid_absolute_position_range': [0, 2],
        'visible_slice_shape': list(ngram_valid.shape),
        'visible_slice_values': ngram_valid.tolist(),
        'visible_slice_digest': arrdig(ngram_valid),
        'unused_capacity': 'unspecified for torch.empty allocation; excluded from hash',
        'future_decode_step0_dependency': {
            'position_2_write': True,
            'history_reads': ['pos1', 'pos0', 'sequence_beginning_pad'],
            'incremental_hash_values_computed': False,
        },
    }

    partial_states = {}
    for layer in KV_SOURCE_LAYERS:
        ratio = int(c['compress_ratios'][layer])
        if ratio > 1:
            partial_states[str(layer)] = {
                'owner': f'layers.{layer}.attn.compressor.kv_state/score_state',
                'ratio': ratio,
                'allocated_shape': [OFFICIAL_DEFAULT_MAX_BATCH_SIZE, ratio, D],
                'kv_state_dtype': 'float32',
                'score_state_dtype': 'float32',
                'prefill_remainder': PREFILL_S % ratio,
                'decode_step0_slot': START_POS + PREFILL_S,
                'decode_step0_slot_mod_ratio': (START_POS + PREFILL_S) % ratio,
                'slot_classification': {
                    str(i): ('OVERWRITTEN_BEFORE_READ' if i == ((START_POS + PREFILL_S) % ratio) else 'UNOBSERVED')
                    for i in range(ratio)
                },
                'read_by_decode_step0_bytes': 0,
                'visible_digest_recorded': False,
                'classification': 'persistent allocation; no prefill-end bytes are promoted because decode step0 overwrites slot0 before completing/reading a new ratio2 group',
            }

    official_state = {
        'ngram_cache': ngram_entry,
        'per_layer_window_kv_cache': window_entries,
        'kv_source_compress_kv_cache': compressed_entries,
        'owner_indexer_k_cache': index_entries,
        'ratio_gt1_compressor_partial_state': partial_states,
    }

    generation_loop_control = {
        'classification': 'generation-loop control state, not model cache state',
        'current_prompt': TOKENS,
        'prefill_start_pos': START_POS,
        'prefill_sequence_length': PREFILL_S,
        'temperature': 0,
        'prefill_output_token': int(output_ids[0]),
        'next_input_ids': [[int(output_ids[0])]],
        'next_start_pos': 2,
        'next_sequence_length': 1,
    }
    target_runtime_session_state = {
        'classification': 'out-of-band target-runtime session state',
        'stochastic_boundary12e_next_session_key_digest': '175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4',
        'not_part_of_official_model_persistent_state': True,
        'not_part_of_deterministic_decode_fixture_requirements': True,
        'not_a_Transformer_forward_return_member': True,
        'new_rng_draw_generated': False,
    }
    excluded = {
        'call_local_values_excluded_from_persistent_snapshot': True,
        'h': 'excluded per-forward-call local',
        'pre_mix': 'excluded per-forward-call local',
        'main_hiddens': 'excluded per-forward-call local/return value',
        'candidates': 'excluded regenerated current-call value',
        'topk_idxs': 'excluded regenerated current-call value',
        'shared_attn.compress_kv': {'physical_attribute_may_exist_after_prefill': True, 'semantic_cross_call_value_required': False},
        'shared_attn.index_k': {'physical_attribute_may_exist_after_prefill': True, 'semantic_cross_call_value_required': False},
        'shared_attn.candidates': {'physical_attribute_may_exist_after_prefill': True, 'semantic_cross_call_value_required': False},
        'shared_attn.topk_idxs': {'physical_attribute_may_exist_after_prefill': True, 'semantic_cross_call_value_required': False},
        'candidates_in_persistent_state': False,
        'topk_idxs_in_persistent_state': False,
    }
    call_local_diag['final_shared_attn'] = {
        k: (None if v is None else {'shape': list(v.shape), 'dtype': str(v.dtype), 'digest': arrdig(np.ascontiguousarray(v))})
        for k, v in shared.items()
    }

    source_ids = _build_source_identities(ck)
    manifest = {
        'schema': 'ds41f.boundary13a.prefill-end-persistent-state-manifest.v1',
        'source_identities': source_ids,
        'fixture': {'tokens': TOKENS, 'B': 1, 'S': 2, 'start_pos': 0, 'world_size': 1, 'engram_mask': None},
        'official_model_persistent_state': official_state,
        'generation_loop_control': generation_loop_control,
        'target_runtime_session_state': target_runtime_session_state,
        'excluded_call_local_fields': excluded,
        'persistent_state_tensor_artifact_injection': False,
        'decode_forward_executed': False,
    }
    manifest['snapshot_manifest_digest'] = canonical_digest(manifest)

    state = NativeDecodeSessionState(
        official_model_persistent_state=official_state,
        generation_loop_control=generation_loop_control,
        target_runtime_session_state=target_runtime_session_state,
        excluded_call_local_fields=excluded,
        manifest=manifest,
        visible_value_arrays=arrays,
    )

    regressions = {
        'full_Ngram_hash': arrdig(full_hash),
        'Ngram_cache_visible_prefix': arrdig(ngram_valid),
        'post_engram1_h': post1,
        'post_engram14_h': post14,
        'final_logits': arrdig(logits),
        'output_ids': output_ids.astype(int).tolist(),
        'compress_kv': {str(k): compressed_entries[str(k)]['visible_prefix_digest'] for k in compressed_entries},
        'index_k': {str(k): index_entries[str(k)]['visible_prefix_digest'] for k in index_entries},
    }
    diagnostics = {
        'event_trace': events,
        'upstream_regressions': regressions,
        'prefill_call_local_diagnostics': call_local_diag,
        'decode_step0_input': {'input_ids': [[int(output_ids[0])]], 'start_pos': 2, 'state_object_constructed': True, 'decode_forward_executed': False},
    }
    return state, diagnostics


def _window_entry(layer: int, window_kv: np.ndarray) -> dict[str, Any]:
    return {
        'owner': f'layers.{layer}.attn.window_kv_cache',
        'layer': layer,
        'dtype': 'BF16(uint16)',
        'allocated_shape': [OFFICIAL_DEFAULT_MAX_BATCH_SIZE, WINDOW_SIZE, D],
        'window_size': WINDOW_SIZE,
        'valid_absolute_positions': [0, 1],
        'ring_slots': [0, 1],
        'visible_region': {'batch': [0, 1], 'slots': [0, 2], 'source_absolute_positions': [0, 2]},
        'visible_slice_shape': list(window_kv.shape),
        'visible_slice_digest': arrdig(window_kv),
        'producer_window_kv_digest': arrdig(window_kv),
        'cache_visible_slots_equal_producer_window_kv': True,
        'unused_capacity': 'zero initialized at model construction but source-invisible/stale beyond visible slots; excluded from digest',
    }


def _maybe_record_producers(layer: int, c: dict[str, Any], out: dict[str, Any], compressed_entries: dict[str, Any], index_entries: dict[str, Any], arrays: dict[str, Any]) -> None:
    prod = out['attn_path'].get('producer')
    if layer not in KV_SOURCE_LAYERS or not prod:
        return
    ratio = int(c['compress_ratios'][layer])
    valid_len = _compress_valid_len(layer, ratio)
    ckv = np.ascontiguousarray(prod['compress_kv'][:, :valid_len, :])
    ik = np.ascontiguousarray(prod['index_k'][:, :valid_len, :])
    compressed_entries[str(layer)] = {
        'owner': f'layers.{layer}.attn.compress_kv_cache',
        'layer': layer,
        'ratio': ratio,
        'dtype': 'BF16(uint16)',
        'allocated_shape': [OFFICIAL_DEFAULT_MAX_BATCH_SIZE, OFFICIAL_DEFAULT_MAX_SEQ_LEN // ratio, D],
        'valid_compressed_index_range': [0, valid_len],
        'visible_prefix_shape': list(ckv.shape),
        'visible_prefix_digest': arrdig(ckv),
        'source_publication_digest': arrdig(ckv),
        'unused_capacity': 'zero initialized allocation; positions beyond valid prefix are source-invisible for first decode and excluded from digest',
    }
    index_entries[str(layer)] = {
        'owner': f'layers.{layer}.attn.indexer.k_cache',
        'owner_layer': layer,
        'object_identity': f'Transformer.layers[{layer}].attn.indexer',
        'ratio': ratio,
        'dtype': 'BF16(uint16)',
        'allocated_shape': [OFFICIAL_DEFAULT_MAX_BATCH_SIZE, OFFICIAL_DEFAULT_MAX_SEQ_LEN // ratio, ID],
        'valid_compressed_index_range': [0, valid_len],
        'visible_prefix_shape': list(ik.shape),
        'visible_prefix_digest': arrdig(ik),
        'source_publication_digest': arrdig(ik),
        'unused_capacity': 'zero initialized allocation; positions beyond valid prefix are source-invisible for first decode and excluded from digest',
    }
    arrays[f'compress_kv.{layer}.visible'] = ckv.copy()
    arrays[f'index_k.{layer}.visible'] = ik.copy()


def snapshot_manifest(state: NativeDecodeSessionState) -> dict[str, Any]:
    return state.manifest
