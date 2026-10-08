"""Compare M55 matched receipts without turning topology counters into GPU timers."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('artifacts/m55')
    summary = {}
    outcome = request = None
    for lane in ('off', 'first-party-mtp-development', 'candidate'):
        data = json.loads((root / f'matched-{lane}.json').read_text())
        assert 'error' not in data and 'MEASUREMENT COMPLETE' in data['decision']
        if request is None:
            request = data['request_sha256']
        assert request == data['request_sha256']
        rows = [r for r in data['rows'] if not r['warmup']]
        assert len(rows) == 3
        for r in rows:
            actual = (r['generated_tokens'], r['frontier'],
                      r['response']['choices'][0]['message'],
                      r['response']['choices'][0]['finish_reason'])
            if outcome is None:
                outcome = actual
            assert actual == outcome, 'unmatched HTTP outcome'
            assert r['replay'] == r['repack'] == 0
        offered = sum((r['acceptance'] or {}).get('offered', 0) for r in rows)
        accepted = sum((r['acceptance'] or {}).get('accepted', 0) for r in rows)
        summary[lane] = dict(
            tokens=rows[0]['tokens'], frontier=rows[0]['frontier'],
            integrated_decode_tok_s=sum(r['tokens'] for r in rows) / sum(r['end_to_end_decode_s'] for r in rows),
            full_http_tok_s=sum(r['tokens'] for r in rows) / sum(r['http_latency_s'] for r in rows),
            offered=offered, accepted=accepted, acceptance=accepted / offered if offered else None,
            mean_phase_s={k: sum(r['phase_s'].get(k, 0) for r in rows) / len(rows)
                          for k in sorted({k for r in rows for k in r['phase_s']})},
            resources=data['resources'])
    topology = {}
    for name in ('before-topology-first-party', 'intermediate-host-copy-topology',
                 'after-topology-first-party', 'candidate-topology'):
        data = json.loads((root / f'{name}.json').read_text())
        assert 'error' not in data and 'MEASUREMENT COMPLETE' in data['decision']
        r = [r for r in data['rows'] if not r['warmup']][0]
        assert request == data['request_sha256']
        assert (r['generated_tokens'], r['frontier'], r['response']['choices'][0]['message'],
                r['response']['choices'][0]['finish_reason']) == outcome
        topology[name] = dict(integrated_decode_tok_s=r['end_to_end_decode_tok_s'],
                              phase_s=r['phase_s'], phase_calls=r['phase_calls'],
                              counters=r['verify_topology'], metrics=r['metrics'])
    before = topology['before-topology-first-party']
    after = topology['after-topology-first-party']
    candidate = topology['candidate-topology']
    def count(row, suffix):
        return sum(v for k, v in row['counters'].items() if k.endswith(suffix))
    fp_rows = count(after, '/target_forward.py:forward')
    candidate_calls = count(candidate, '/batch_generator.py:_call_backbone_captured')
    candidate_rows = candidate['metrics']['cycles'] + sum(candidate['metrics']['depth_drafted'])
    assert fp_rows == count(before, '/target_forward.py:forward') == 41
    assert candidate_calls == 10 and candidate_rows == 40
    assert count(after, '/language.py:_input_projections') == 40 * fp_rows
    assert count(candidate, '/language.py:_input_projections') == 40 * candidate_calls
    cost_model = dict(
        first_party_verify_rows=fp_rows, first_party_forward_calls=fp_rows,
        candidate_verify_rows=candidate_rows, candidate_forward_calls=candidate_calls,
        first_party_layer_regions=40 * fp_rows, candidate_layer_regions=40 * candidate_calls,
        before_journal_and_tap_host_array_copies=count(before, 'native:numpy:array'),
        after_journal_and_tap_host_array_copies=count(after, 'native:numpy:array'),
        diagnostic_before_verify_s=before['phase_s']['target_verify_s'],
        diagnostic_after_verify_s=after['phase_s']['target_verify_s'],
        diagnostic_candidate_verify_s=candidate['phase_s']['candidate_target_verify_s'],
        diagnostic_single_row_s=after['phase_s']['target_verify_s'] / fp_rows,
        diagnostic_four_row_block_s=candidate['phase_s']['candidate_target_verify_s'] / candidate_calls,
        diagnostic_verify_cost_ratio=after['phase_s']['target_verify_s'] / candidate['phase_s']['candidate_target_verify_s'],
        matched_verify_cost_ratio=summary['first-party-mtp-development']['mean_phase_s']['target_verify_s'] / summary['candidate']['mean_phase_s']['candidate_target_verify_s'],
        interpretation='41 serial one-row regions versus 10 four-row regions; similar per-forward latency, not equal per-row cost. Diagnostic ratio is a calibrated aggregate model, not an independently measured GPU operator sum. Counters disprove duplicate target rows as the main explanation; they do not count GPU dispatches.')
    qualifications = {}
    for name in ('pre-change-r1', 'post-change-r1', 'real-prefix-state', 'real-prefix-with-taps', 'real-operational'):
        path = root / f'{name}.json'
        data = json.loads(path.read_text())
        if name.endswith('r1'):
            assert data['status'] == 'PASS' and len(data['gates']) == 25
        elif name.startswith('real-prefix'):
            assert data['status'] == 'PASS'
        else:
            assert 'PASS' in data['decision'] and 'NOT PASS' not in data['decision']
        qualifications[name] = dict(decision=data.get('decision'), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    detail = json.loads((root / 'residual-phase-detail.json').read_text())
    assert 'error' not in detail and 'MEASUREMENT COMPLETE' in detail['decision']
    row = [r for r in detail['rows'] if not r['warmup']][0]
    assert detail['request_sha256'] == request
    assert (row['generated_tokens'], row['frontier'], row['response']['choices'][0]['message'],
            row['response']['choices'][0]['finish_reason']) == outcome
    p = row['phase_s']
    residual = dict(phase_s=p, phase_calls=row['phase_calls'],
                    observed_integrated_decode_s=row['end_to_end_decode_s'],
                    disjoint_execution_frame_sum_s=sum(p[k] for k in (
                        'M52_cycle_inclusive_s', 'protected_target_step_s', 'proposal_s', 'derived_ring_publication_s')),
                    M52_sampling_history_receipt_remainder_s=p['M52_cycle_inclusive_s'] - sum(p[k] for k in (
                        'target_verify_s', 'M51_journal_setup_s', 'M51_settlement_s')))
    final = json.loads((root / 'post-change-r1.json').read_text())
    for name, digest in final['candidate_source'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, 'not final-source R1'
    record = dict(decision='BOUNDED MATERIALIZATION CHANGE QUALIFIED; ROW-SERIAL TARGET BLOCKER REMAINS; NO PROMOTION',
                  request_sha256=request, matched_correctness=True, lanes=summary,
                  topology=topology, execution_cost_model=cost_model, residual_phase_detail=residual, qualifications=qualifications,
                  limits='Inclusive host-wall phases overlap; counters are not GPU kernel times. Extended topology samples are diagnostic, separate from three-sample matched rates. Candidate chain counters include its next proposal; target-only projection/attention counts are separately identifiable.')
    (root / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
