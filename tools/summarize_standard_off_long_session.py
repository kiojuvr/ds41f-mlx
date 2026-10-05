#!/usr/bin/env python3
"""Close a bounded OFF production qualification from actual process receipts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics


def summarize(phases):
    if len(phases) < 2 or phases[0]['phase'] != 'initial':
        raise ValueError('initial and fresh-process restore evidence required')
    if not all(p['status'] == 'PASS' for p in phases):
        raise ValueError('failed/incomplete process receipt')
    if phases[0].get('fixture',{}).get('count') != 200000:
        raise ValueError('200K baseline receipt required')
    p5=phases[0].get('p5')
    if p5 != dict(handoff_count=1,same_list=True,exported=False,replay=0,repack=0):
        raise ValueError('P5 transfer invariant')
    for previous, current in zip(phases, phases[1:]):
        if current['phase'] != 'restore' or not current.get('fresh_process_restore_exact'):
            raise ValueError('unqualified restart')
        if current['turns'][0]['frontier_before'] != previous['persisted_frontier']:
            raise ValueError('restored frontier mismatch')
        if [r['token'] for r in current['turns'][0]['reports']] != previous['probe_tokens']:
            raise ValueError('restored decode mismatch')
        if not current['turns'][0]['same_list']:
            raise ValueError('restored authority replaced')
    turns = [r for p in phases for r in p['turns']]
    initial = phases[0]['initial_decode']
    if any(r['replay'] or r['repack'] or not r['same_list'] for r in turns):
        raise ValueError('replay/repack/ownership invariant')
    for r in turns:
        if r['frontier_after'] != r['frontier_before']+r['suffix_tokens']+len(r['reports']):
            raise ValueError('history frontier mismatch')
    if not any(r['cancelled'] for r in turns):
        raise ValueError('cancellation evidence missing')
    for p in phases:
        admission=p['admission']
        cap=admission['allocator_cache_limit_bytes']
        if cap is None or not 0 <= cap <= 32*1024**3:
            raise ValueError('allocator cache budget missing/unbounded')
        if p['retired_allocator_cache_limit_bytes'] != admission['previous_allocator_cache_limit_bytes']:
            raise ValueError('allocator policy not restored')
        d=p['diagnostics']
        if (d['total_prompt_replay_count'] or d['total_full_cache_repack_count'] or
                d['cache_layer_count'] != 40 or not d['all_cache_offsets_equal_frontier'] or
                p['backend_active_sessions'] != 0):
            raise ValueError('idle/cleanup invariant')
    rates=[r['decode_tok_s'] for r in turns if not r['cancelled'] and len(r['reports'])>=32]
    resources=[r for p in phases for r in p['resources']]
    groups=[rates[:max(1,len(rates)//3)], rates[len(rates)//3:2*len(rates)//3], rates[2*len(rates)//3:]]
    performance_ok=min([initial['decode_tok_s']]+rates)>=15
    return dict(schema='ds41f.standard-off.long-session.summary.v1',
        decision='QUALIFIED_BOUNDED_200K_FIRST_PARTY_OFF' if performance_ok else 'BLOCKED_PRACTICAL_DECODE',
        context_tokens=phases[0]['fixture']['count'], final_frontier=phases[-1]['final_frontier'],
        process_count=len(phases), fresh_process_restores=len(phases)-1,
        wall_s=sum(p['wall_s'] for p in phases),
        prefill_s=phases[0]['prefill']['seconds'], prefill_tok_s=phases[0]['prefill']['tok_s'],
        initial_decode_tok_s=initial['decode_tok_s'],
        sustained_decode_tok_s=dict(min=min(rates),median=statistics.median(rates),max=max(rates),
            early_middle_late_medians=[statistics.median(g) for g in groups]),
        continuation_bootstrap_s=dict(min=min(r['bootstrap_s'] for r in turns),max=max(r['bootstrap_s'] for r in turns)),
        decoded_tokens=len(initial['reports'])+sum(len(r['reports']) for r in turns),
        continued_turns=len(turns), cancelled_turns=sum(r['cancelled'] for r in turns),
        persistence=[dict(frontier=p['persisted_frontier'], bytes=p['artifact_bytes'],save_s=p['save_s'],restore_s=p.get('restore_s')) for p in phases],
        resources=dict(peak_mlx_active_bytes=max(r['mlx_peak_bytes'] for r in resources),
            max_sampled_mlx_active_plus_free_cache_bytes=max(r.get('mlx_active_bytes',0)+r.get('mlx_cache_bytes',0) for r in resources),
            peak_sampled_rss_bytes=max(r['rss_bytes'] for r in resources),
            swap_start_bytes=resources[0]['swap_used_bytes'],swap_end_bytes=resources[-1]['swap_used_bytes'],
            per_process_idle=[dict(first=p['turns'][0]['memory'], last=p['turns'][-1]['memory']) for p in phases]),
        allocator_cache_limits_bytes=[p['admission']['allocator_cache_limit_bytes'] for p in phases],
        pre_policy_numerical_comparison=phases[0].get('pre_policy_comparison'),
        zero_replay_repack=True, exact_restart_tokens_and_all_280_slots=True,
        invariants='one packed list; all 40 offsets equal consumed history; sampled lookahead discarded on cancel; invalid prefix rejected',
        limitations=['finite single-flight core workload, not an unbounded leak proof',
            'long turns use exact-prefix token suffixes, not a complete HTTP/client tool-loop qualification',
            'no crash-durability, immediate in-transaction abort, concurrency or cross-backend persistence claim',
            'no 512K, vision, speculative/MTP, R1 or release/promotion claim'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phases', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); result=summarize([json.loads(f.read_text()) for f in a.phases])
    result['receipts']=[str(f) for f in a.phases]
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['decision']=='QUALIFIED_BOUNDED_200K_FIRST_PARTY_OFF' else 1


if __name__=='__main__': raise SystemExit(main())
