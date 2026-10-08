"""Assert final canonical block qualification and matched executed topology."""
import hashlib
import json
from pathlib import Path

ROOT = Path('artifacts/m56')


def load(name):
    return json.loads((ROOT / name).read_text())


def counter(topology, suffix):
    return sum(value for key, value in topology.items() if key.endswith(suffix))


def main():
    qualification = {}
    for name in ('final-all-prefix-state.json', 'all-widths-state.json'):
        receipt = load(name)
        assert receipt['status'] == 'PASS'
        assert all(c['exact_all_280_state_and_metadata_off_match']
                   and c['exact_same_forward_taps_off_match']
                   and c['exact_full_logits_off_match'] for c in receipt['cycles'])
        assert all(c['off_match'] and c['same_forward_tap_match'] for c in receipt['arbitrary_prefixes'])
        assert len(receipt['faults']) == 15
        assert all(c['off_match'] for c in receipt['cancellation'])
        assert all(c['replay'] == c['repack'] == 0 for c in receipt['setup_counters'])
        assert receipt['off_regression']['all_state_byte_match']
        for path, digest in receipt['runtime_source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        qualification[name] = dict(frontier=receipt['initial_frontier'],
            width=receipt['block_width'], faults=len(receipt['faults']),
            additional_width_prefix_trials=len(receipt.get('width_matrix', [])),
            off_latency_ratio=receipt['off_regression']['latency_ratio_m51_over_m50'])
    widths = load('all-widths-state.json')['width_matrix']
    assert {(r['offered'], r['accepted']) for r in widths} == {
        (width, prefix) for width in range(2, 8) for prefix in range(width + 1)}
    assert all(r['all_state_off_match'] and r['same_forward_tap_match']
               and r['full_logits_match'] for r in widths)
    proposal = load('final-proposal-loop.json')
    assert 'error' not in proposal and proposal['decision'].startswith('MATRIX RUN')
    assert proposal['matrix']['greedy_off_parity']['exact']
    assert proposal['matrix']['stochastic_off_parity']['exact']
    assert not proposal['retirement']['active']
    assert proposal['retirement']['receipts'] == proposal['retirement']['producers'] == 0
    operation = load('real-operational.json')
    assert operation['decision'].endswith('PROBE PASS')
    assert operation['counts']['forbidden'] == operation['duplicate_effects'] == 0
    assert operation['counts']['effects'] == 5
    assert operation['worker_eof_disconnect']['exact_complete_tool_deltas']
    old_operation = json.loads(Path('artifacts/m55/real-operational.json').read_text())
    assert len(operation['cases']) == len(old_operation['cases']) == 17
    for actual, oracle in zip(operation['cases'], old_operation['cases']):
        assert actual['turn']['generated_tokens'] == oracle['turn']['generated_tokens']
        assert actual['turn']['frontier_after_commit'] == oracle['turn']['frontier_after_commit']
    lanes, outcome, request = {}, None, None
    for lane in ('off', 'first-party-mtp-development', 'candidate'):
        receipt = load('matched-' + lane + '.json')
        assert 'error' not in receipt and 'MEASUREMENT COMPLETE' in receipt['decision']
        request = request or receipt['request_sha256']
        assert receipt['request_sha256'] == request
        rows = [row for row in receipt['rows'] if not row['warmup']]
        assert len(rows) == 3
        for row in rows:
            actual = (row['generated_tokens'], row['frontier'],
                row['response']['choices'][0]['message'], row['response']['choices'][0]['finish_reason'])
            outcome = outcome or actual
            assert actual == outcome
            assert row['replay'] == row['repack'] == 0
        lanes[lane] = dict(
            decode_tok_s=sum(r['tokens'] for r in rows) / sum(r['end_to_end_decode_s'] for r in rows),
            http_tok_s=sum(r['tokens'] for r in rows) / sum(r['http_latency_s'] for r in rows),
            offered=sum((r['acceptance'] or {}).get('offered', 0) for r in rows),
            accepted=sum((r['acceptance'] or {}).get('accepted', 0) for r in rows),
            mean_phase_s={key: sum(r['phase_s'].get(key, 0) for r in rows) / 3
                          for key in {k for r in rows for k in r['phase_s']}})
    topology = {}
    for lane in ('first-party-mtp-development', 'candidate'):
        receipt = load('topology-' + lane + '.json')
        assert 'error' not in receipt and receipt['request_sha256'] == request
        row = [r for r in receipt['rows'] if not r['warmup']][0]
        assert (row['generated_tokens'], row['frontier'], row['response']['choices'][0]['message'],
                row['response']['choices'][0]['finish_reason']) == outcome
        counts = row['verify_topology']
        if lane == 'first-party-mtp-development':
            forwards = counter(counts, '/runtime/target_forward.py:forward')
            regions = counter(counts, '/runtime/state_production.py:block')
            attention = counter(counts, '/model_execution/kernels.py:packed_sparse_attention')
            assert forwards == 7 and regions == 280
        else:
            forwards = counter(counts, 'batch_generator.py:_call_backbone_captured')
            regions = counter(counts, 'language.py:_input_projections')
            attention = counter(counts, 'kernels.py:packed_sparse_attention')
            assert forwards == 10 and regions == 400 and attention == 400
        histogram = {int(key.rsplit('_', 1)[1]): value for key, value in counts.items()
                     if key.startswith('observed_target_width_')}
        assert sum(histogram.values()) == forwards
        assert histogram == ({6: 6, 5: 1} if lane.startswith('first') else {4: 10})
        topology[lane] = dict(target_forwards=forwards, backbone_layer_regions=regions,
                             attention_regions=attention, width_histogram=histogram,
                             verified_rows=sum(k * v for k, v in histogram.items()))
    r1 = load('full-r1.json')
    assert r1['status'] == 'PASS' and len(r1['gates']) == 25
    result = dict(decision='CANONICAL BLOCK EXECUTION QUALIFIED; BOUNDED LOCAL CANDIDATE, NO RELEASE PROMOTION',
        base='5bad422cf9b72f0096e0ce63f3132f9b97fd967a', qualification=qualification,
        proposal=dict(target_forwards=proposal['target_forward_calls'],
                      physical_rows=proposal['physical_target_rows'], decode=proposal['decode']),
        operational=dict(turns=17, counts=operation['counts'],
                         all_M55_tokens_and_frontiers_exact=True, duplicate_effects=0),
        request_sha256=request, matched_generated_inputs=len(outcome[0]), frontier=outcome[1],
        lanes=lanes, topology=topology,
        M55_row_topology=dict(target_forwards=41, backbone_layer_regions=1640, attention_regions=1640),
        replay=0, full_cache_repack=0, hidden_target_reexecution=0,
        full_R1=dict(status=r1['status'], gates=25, runs=1),
        notes=['Region counts are observed Python/native execution boundaries, NOT GPU dispatch counts.',
               'Extended topology trials are separate from three-sample rates; one warm-up excluded in every lane.',
               'Attention alone splits at canonical sparse-list widths; complete target/backbone does not serialize rows.',
               'Batched GEMV and bounded dense RHS scratch preserve OFF reduction; no new Metal/MMA/FP8/MoE kernels.',
               'M51 packed-byte prefix selection copies remain; they are not full-cache repacking.',
               'Chat/greedy/text/single-session/M54 bounds only; no new Vision, restore, context or stochastic application admission.'])
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
