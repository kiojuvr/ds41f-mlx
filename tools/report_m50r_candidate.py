"""Summarize M50R receipts without converting incomplete evidence into PASS.

No GPU dispatch/transfer counts are inferred from Python/native API counters.
Supplementary numerical, native boundary/Metal and bounded reload receipts are
required for PASS. M51R remains explicitly unauthorized even after baseline PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools.freeze_m50r_candidate import dispersion


def summarize(root):
    receipts = {name: json.loads((root / (name + '.json')).read_text())
                for name in ('identity', 'performance', 'trace')}
    identity, performance, trace = (receipts[k] for k in ('identity', 'performance', 'trace'))
    for name, receipt in receipts.items():
        assert receipt['mode'] == name
        assert receipt['head'] == performance['head']
        assert receipt['identity']['identity_sha256'] == performance['identity']['identity_sha256']
        assert receipt['identity']['checkpoint'] == performance['identity']['checkpoint']
        assert hashlib.sha256((root / (name + '.tool.py')).read_bytes()).hexdigest() == receipt['tool_sha256']
    assert identity['status'] == 'IDENTITY_CAPTURED'
    assert performance['status'] == trace['status'] == 'CAPTURED_NOT_GATE_PASS'
    assert len(identity['checkpoint_bytes']) == 48, 'fresh full checkpoint byte hashing required'
    assert sum(s['bytes'] for s in identity['checkpoint_bytes']) == sum(s['bytes'] for s in identity['identity']['checkpoint']['shards'])
    for actual, expected in zip(identity['checkpoint_bytes'], identity['identity']['checkpoint']['shards']):
        assert actual['name'] == expected['name'] and actual['sha256'] == expected['lfs_sha256']
    rows = [r for r in performance['rows'] if not r['warmup']]
    assert len(rows) >= 3 and performance['rows'][0]['warmup']
    assert performance['token_identity']
    assert performance['server'] == trace['server'] or {
        k:v for k,v in performance['server'].items() if k != 'port'} == {
        k:v for k,v in trace['server'].items() if k != 'port'}
    assert performance['server']['protocol'] == 'LocalH11Protocol'
    for row in [*rows, *trace['rows']]:
        t = row['trace']
        assert row['retired']
        assert len(t['target_offsets']) == 40 and set(t['target_offsets']) == {t['canonical_frontier']}
        assert set(t['dspark_offsets']) == {t['canonical_frontier']}
        assert t['queue_empty'] and t['prediction_retired']
        assert t['prompt_replay'] == t['full_cache_repack'] == 0
        assert t['quiescence']['counters']['new_verify_cycles'] == t['quiescence']['counters']['new_proposals'] == 0
    ordinary = trace['rows'][0]
    assert ordinary['request_sha256'] == rows[0]['request_sha256']
    assert ordinary['trace']['canonical_generated'] == rows[0]['trace']['canonical_generated']
    topology = []
    for row in trace['rows']:
        counts = row['topology']
        widths = Counter()
        cycles = []
        for event in row['causal_events']:
            if event['name'] == '_call_backbone_captured' and event['event'] == 'call':
                widths[str(event['state']['input_shape'][-1])] += 1
            if event['name'] == '_run_verify_cycle_chain' and event['event'] == 'return':
                cycles.append(event['state'])
        def regions(suffix):
            return sum(v for k,v in counts.items() if k.endswith(suffix))
        assert regions('language.py:_input_projections') == 40 * len(cycles)
        assert regions('kernels.py:packed_sparse_attention') == 40 * len(cycles)
        for width in {c['k'] + 1 for c in cycles if c['k'] > 0}:
            assert widths[str(width)] == sum(c['k'] + 1 == width for c in cycles)
        topology.append(dict(sample=row['sample'], cancelled=row['trace']['cancelled'],
            requested_cancel=row['cancel_requested'], terminal_matches=row['trace']['terminal_matches'],
            target_width_histogram=dict(widths),
            verify_layer_projections=regions('language.py:_input_projections'),
            verify_packed_attention=regions('kernels.py:packed_sparse_attention'),
            cycles=[{k:c.get(k) for k in ('k','m','m_gpu','stash_before','ring_offsets','queue')} for c in cycles],
            quiescence=row['trace']['quiescence']))
    timing = {key: dispersion([r[key] for r in rows]) for key in
              ('decode_tok_s', 'settled_decode_tok_s', 'http_s', 'request_tok_s', 'ttft_content_s')}
    phase = {key: dispersion([r['trace'][key] for r in rows]) for key in
             ('prefill_handoff_s', 'cleanup_s', 'decode_s', 'elapsed_s')}
    telemetry = {key: dispersion([r['trace']['mtp_stats'][key]/1000 for r in rows]) for key in
                 ('backbone_ms', 'mtp_head_ms', 'sample_ms', 'cache_ops_ms')}
    # Noise budget: same-process CV is not a cross-process/day confidence bound.
    # >5% outside measured uncertainty is a mandatory blocking screen, not an
    # allowance for changing topology. No independent samples => no approval.
    from tools.qualify_m50r_closure import supplement
    closure=supplement(root,performance['identity']['identity_sha256'],rows[0]['request_sha256'],rows)
    passed=not closure['blockers']
    return dict(schema='ds41f.m50r.summary.v2', decision='PASS' if passed else 'BLOCK',
        m51r_prerequisite_satisfied=passed, m51r_authorized=False, closure=closure,
        head=performance['head'],
        identity_sha256=performance['identity']['identity_sha256'],
        checkpoint_manifest_sha256=hashlib.sha256(json.dumps(identity['checkpoint_bytes'], sort_keys=True).encode()).hexdigest(),
        checkpoint_bytes=sum(s['bytes'] for s in identity['checkpoint_bytes']),
        measured_trials=len(rows), tokens=len(rows[0]['trace']['canonical_generated']),
        request_sha256=rows[0]['request_sha256'], frontier=rows[0]['trace']['canonical_frontier'],
        timing=timing, phase_s=phase, built_in_telemetry_s=telemetry,
        topology=topology, model_config=performance['model_config'],
        causal_coverage=dict(
            gpu_proposal_mismatch=any(c['m_gpu'] < c['k'] for row in topology for c in row['cycles']),
            backend_or_budget_clamp=any(c['m'] < c['m_gpu'] for row in topology for c in row['cycles']),
            protected_recipe_terminal=any(row['terminal_matches'] for row in topology),
            early_cancel=any(row['cancelled'] and not row['cycles'] for row in topology),
            committed_queue_cancel=any(row['cancelled'] and row['quiescence'].get('queue_drained_tokens') for row in topology),
            fresh_uncertain_mutation_faults=False,
            independent_numerical_state_oracle=passed,
            qualified_native_boundary_and_metal_trace=passed,
            complete_internal_kernel_transfer_sync_trace=False),
        blockers=closure['blockers'], screening=dict(decode_and_verify_regression=0.05,
            policy='>5% outside measured uncertainty blocks; no topology allowance or optimization-backlog waiver',
            noise_budget='original six plus twelve bounded process-reload sessions when closure PASS; empirical full decode/verify ranges <1%, not a confidence interval or cross-day allowance. Matched current candidate control required; >1% control drift or new resource regime requires more repeats/causal isolation. >5% regression beyond uncertainty BLOCKS.'),
        caveats=[
            'decode_tok_s uses summed existing _next worker timers; settled_decode_tok_s subtracts built-in load/prefill timers from settled request elapsed wall time, including delivery/recipe/settlement and worker scheduling.',
            'backbone_ms includes graph enqueue, host materialization and outstanding lazy work; mtp_head_ms is asynchronous proposal enqueue, not GPU proposal latency; these are NOT additive isolated device phases.',
            'Counts include canonical suppressed backend stop token, not just visible text.',
            'after_close MLX residency includes the loaded process-scoped model; DELETE retires sessions, not model weights.',
            'Serialized quiescence window_size derives from dspark_block_size (5); actual native ring max_size is window_size (128).',
            'HTTP uses canonical LocalH11 options and explicit preloaded worker with lifespan off; no model-cold operator-launch/TTFT performance claim.'
        ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('artifacts/m50r'))
    args = ap.parse_args()
    result = summarize(args.root)
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
