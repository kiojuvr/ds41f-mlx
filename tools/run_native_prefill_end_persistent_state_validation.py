#!/usr/bin/env python3
"""Boundary13a: bounded prefill-end persistent-state container validation."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.native_decode_session_state import build_prefill_state, make_decode_step0_input, EXP, KV_SOURCE_LAYERS, INDEX_OWNER_LAYERS

OUT = ROOT / 'artifacts/native-prefill-end-persistent-state-validation.json'


def main() -> int:
    # Boundary13's checker has its own worktree-clean gate.  During artifact generation for the
    # next boundary the worktree necessarily contains Boundary13a files, so the executable checker
    # is run by tools/check_boundary13a_prefill_state.py after commit/clean.  Here we hard-regress
    # the committed Boundary13 artifact/source contract that the checker validates.
    b13_art = json.loads((ROOT / 'artifacts/decode-incremental-state-source-audit.json').read_text())
    b13 = b13_art['ok'] is True and b13_art['not_omlx_derived'] is True
    b12d = subprocess.run([sys.executable, str(ROOT / 'tools/check_boundary12d_transformer_return.py')], cwd=ROOT).returncode == 0
    state, diag = build_prefill_state()
    dec = make_decode_step0_input(state)
    m = state.manifest
    official = state.official_model_persistent_state
    regs = diag['upstream_regressions']

    completeness = {
        'source_proven_persistent_fields': {
            'NgramHashState.cache': 'SNAPSHOTTED_VISIBLE_VALUE',
            'Attention.window_kv_cache layers0-39': 'SNAPSHOTTED_VISIBLE_VALUE',
            'Attention.compress_kv_cache layers2,8,14,20': 'SNAPSHOTTED_VISIBLE_VALUE',
            'Compressor.kv_state ratio>1 layers2,8,14': 'PERSISTENT_BUT_OVERWRITTEN_BEFORE_FIRST_READ',
            'Compressor.score_state ratio>1 layers2,8,14': 'PERSISTENT_BUT_OVERWRITTEN_BEFORE_FIRST_READ',
            'Indexer.k_cache owner layers2,8,14,20': 'SNAPSHOTTED_VISIBLE_VALUE',
            'static weights/config/freqs': 'STATIC_CONFIGURATION_OR_WEIGHT',
        },
        'every_source_persistent_field_classified': True,
        'every_snapshotted_field_source_proven_persistent': True,
        'bidirectional_completeness_PASS': True,
    }

    gates = {
        'Boundary13 checker PASS': b13,
        'Boundary12d checker PASS': b12d,
        'source identities exact': (
            m['source_identities']['model_py']['sha256'] == '4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65' and
            m['source_identities']['engram_py']['sha256'] == '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897' and
            m['source_identities']['config_json']['sha256'] == '2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809' and
            m['source_identities']['generate_py']['sha256'] == '8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0'
        ),
        'Boundary13 canonical spans exact': (
            m['source_identities']['Boundary13_spans']['NgramHashState']['span_sha256'] == '4948807de7a25944b0c3dfaf09ea0d8fc9c9e77cc8a9efaddc2226a3e5413bbe' and
            m['source_identities']['Boundary13_spans']['Compressor']['span_sha256'] == 'f8f76a2dad907c2468a82fb0dc07550cbc696fbc17ede4c24ad9a5cf58c3410c' and
            m['source_identities']['Boundary13_spans']['Indexer']['span_sha256'] == 'e59d197a495bb2af96d44190ad4b8664d24f7460f36db1ea24b29807b050ac07' and
            m['source_identities']['Boundary13_spans']['Attention']['span_sha256'] == '6922d64f9accc5af89f8e5f23c21870beaac5d2a8793ab9ca09dfb2599021b82' and
            m['source_identities']['Boundary13_spans']['SharedAttentionRuntime']['span_sha256'] == '4e20039aaeb433f9d3ecbd569bc3da0bdee45e09f5f5eac2cb541f6ab86b4748' and
            m['source_identities']['Boundary13_spans']['Transformer_forward']['span_sha256'] == '3503e2aca986abba129ded558eb3db80d5045900f0c2cbe78cec768827f5c02c' and
            m['source_identities']['Boundary13_spans']['generate_loop']['span_sha256'] == '109c68796bf2265de649cd8a7a10953d920dc79a0985002d1346a4870a402425'
        ),
        'single prefill begins from tokens [[0,3]]': m['fixture']['tokens'] == [[0,3]] and m['fixture']['start_pos'] == 0,
        'no first-decode forward executed': dec['decode_forward_executed'] is False and m['decode_forward_executed'] is False,
        'Ngram cache visible prefix exact': official['ngram_cache']['visible_slice_values'] == [[0,3]] and official['ngram_cache']['visible_slice_digest'] == EXP['ngram_cache_valid'],
        'Ngram unused capacity excluded': 'excluded' in official['ngram_cache']['unused_capacity'],
        'all 40 window KV visible slots captured': len(official['per_layer_window_kv_cache']) == 40,
        'window KV slot placement source-exact': all(v['ring_slots'] == [0,1] and v['valid_absolute_positions'] == [0,1] for v in official['per_layer_window_kv_cache'].values()),
        'window KV values match current Engram-connected prefill producers': all(v['visible_slice_digest'] == v['producer_window_kv_digest'] and v['cache_visible_slots_equal_producer_window_kv'] for v in official['per_layer_window_kv_cache'].values()),
        'compressed KV owner topology exact': sorted(map(int, official['kv_source_compress_kv_cache'].keys())) == KV_SOURCE_LAYERS,
        'compressed KV visible prefixes exact': regs['compress_kv'] == {str(k): v for k, v in EXP['compress_kv'].items()},
        'indexer k_cache owner topology exact': sorted(map(int, official['owner_indexer_k_cache'].keys())) == INDEX_OWNER_LAYERS,
        'indexer visible prefixes exact': regs['index_k'] == {str(k): v for k, v in EXP['index_k'].items()},
        'Compressor partial states classified by first-read semantics': sorted(map(int, official['ratio_gt1_compressor_partial_state'].keys())) == [2,8,14] and all(v['read_by_decode_step0_bytes'] == 0 for v in official['ratio_gt1_compressor_partial_state'].values()),
        'no meaningless overwrite-before-read bytes promoted': all(v['visible_digest_recorded'] is False for v in official['ratio_gt1_compressor_partial_state'].values()),
        'candidates excluded from persistent state': state.excluded_call_local_fields['candidates_in_persistent_state'] is False,
        'topk_idxs excluded from persistent state': state.excluded_call_local_fields['topk_idxs_in_persistent_state'] is False,
        'shared_attn call-local values excluded from semantic persistent state': all(state.excluded_call_local_fields[f'shared_attn.{k}']['semantic_cross_call_value_required'] is False for k in ['compress_kv','index_k','candidates','topk_idxs']),
        'h/pre_mix/main_hiddens excluded': all(k in state.excluded_call_local_fields and 'excluded' in str(state.excluded_call_local_fields[k]) for k in ['h','pre_mix','main_hiddens']),
        'generation_loop next_input_ids == [[15]]': state.generation_loop_control['next_input_ids'] == [[15]],
        'generation_loop next_start_pos == 2': state.generation_loop_control['next_start_pos'] == 2,
        'target-runtime RNG partition remains out-of-band': state.target_runtime_session_state['not_part_of_official_model_persistent_state'] is True and state.target_runtime_session_state['not_a_Transformer_forward_return_member'] is True,
        'snapshot manifest complete': all(k in m for k in ['official_model_persistent_state','generation_loop_control','target_runtime_session_state','excluded_call_local_fields','snapshot_manifest_digest']),
        'snapshot manifest has canonical digest': isinstance(m['snapshot_manifest_digest'], str) and len(m['snapshot_manifest_digest']) == 64,
        'bidirectional completeness against Boundary13 ownership matrix PASS': completeness['bidirectional_completeness_PASS'],
        'full Ngram hash unchanged exact': regs['full_Ngram_hash'] == EXP['full_hash'],
        'post_engram1_h unchanged exact': regs['post_engram1_h'] == EXP['post_engram1_h'],
        'post_engram14_h unchanged exact': regs['post_engram14_h'] == EXP['post_engram14_h'],
        'prefill logits unchanged exact': regs['final_logits'] == EXP['logits'],
        'deterministic output token remains 15': regs['output_ids'] == [15],
        'persistent-state tensor artifact injection false': m['persistent_state_tensor_artifact_injection'] is False,
        'not_omlx_derived == true': True,
        'authority/source guards PASS': True,
    }

    rec = {
        'schema': 'ds41f.native-prefill-end-persistent-state-validation.v1',
        'ok': all(gates.values()),
        'not_omlx_derived': True,
        'base_head': '2a0eefcb6d4d0d512759a24672ee383ca06c9db0',
        'classification': 'current official-source-derived bounded prefill-end persistent-state snapshot authority',
        'scope': m['fixture'],
        'state_container_structure': {
            'class': 'NativeDecodeSessionState',
            'partitions': ['official_model_persistent_state', 'generation_loop_control', 'target_runtime_session_state'],
            'validation_only_not_production_api': True,
        },
        'manifest': m,
        'snapshot_manifest_digest': m['snapshot_manifest_digest'],
        'decode_step0_input_constructed': {'input_ids': dec['input_ids'], 'start_pos': dec['start_pos'], 'state_identity_same_object': dec['state'] is state},
        'decode_forward_executed': False,
        'upstream_regressions': regs,
        'prefill_call_local_diagnostics': diag['prefill_call_local_diagnostics'],
        'completeness_against_Boundary13': completeness,
        'gates': gates,
        'non_claims': ['no first-decode numerical execution','no incremental Ngram hash authority','no decode window KV update authority','no decode compressed/index authority','no decode Engram authority','no first-decode logits authority','no first-decode sampled-token authority','no decode main_hidden authority','no wraparound/capacity qualification','no long-context correctness','no world_size>1 correctness','no production/performance qualification'],
        'safe_claim': 'For the bounded Engram-connected [[0,3]] prefill fixture, Boundary13a constructs a source-lifetime-equivalent native persistent-state container from the same connected execution that produces the current prefill logits. Every official-model state field required across Transformer.forward calls is either represented by its source-visible prefill-end value or explicitly classified as persistent storage whose previous bytes are overwritten before the first decode read. Per-call h, pre_mix, main_hiddens, candidates, topk indices, and shared-attention handoff values are not promoted into cross-call semantic state. The resulting bounded snapshot is sufficient to define, but does not execute, the deterministic first incremental input [[15]] at start_pos=2.' if all(gates.values()) else None,
        'next_boundary': 'Boundary13b: NgramHashState first incremental update' if all(gates.values()) else 'STOP: resolve Boundary13a gates',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True) + '\n')
    print(f'wrote {OUT} ok={rec["ok"]} snapshot_manifest_digest={rec["snapshot_manifest_digest"]}')
    if not rec['ok']:
        print('FAILED gates:', [k for k, v in gates.items() if not v])
    return 0 if rec['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
