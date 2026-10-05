#!/usr/bin/env python3
"""Qualify measured very-long OFF frontiers, never configured maxima alone."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.summarize_standard_off_long_session import summarize


def qualify(phases):
    context = phases[0]['fixture']['count']
    if context < 524288:
        raise ValueError('512K-class actual input required')
    result = summarize(phases, context_tokens=context)
    for p in phases:
        if p['runtime_sources'] != phases[0]['runtime_sources']:
            raise ValueError('runtime implementation changed across restore')
        if (not p['admission'].get('wired_limit_bytes') or
                p.get('retired_wired_limit_bytes') != p['admission'].get('previous_wired_limit_bytes')):
            raise ValueError('model-lifetime residency policy missing/not restored')
    initial = phases[0]
    if len(initial['initial_decode']['reports']) < 32:
        raise ValueError('sustained initial decode missing')
    if initial['turns'][0]['frontier_before'] != context+len(initial['initial_decode']['reports']):
        raise ValueError('initial decode/continuation frontier mismatch')
    for p in phases:
        for before, after in zip(p['turns'], p['turns'][1:]):
            if before['frontier_after'] != after['frontier_before']:
                raise ValueError('discontinuous continuation history')
        if p['final_frontier'] != p['turns'][-1]['frontier_after']:
            raise ValueError('final measured frontier mismatch')
        cap=p.get('configured_max_seq_len')
        if cap is None or cap != initial['configured_max_seq_len'] or p['final_frontier'] > cap:
            raise ValueError('outside admitted checkpoint context range')
    prefill = initial['prefill']
    scheduling = prefill['p7_scheduling_evidence']
    if (prefill['production_prefill_selector'] != 'DENSE_P0_P7' or
            prefill['token_count'] != context-1 or prefill['frontier'] != context-1 or
            prefill['portable_state_exported'] or prefill['full_cache_repack_count'] or
            not scheduling['enabled'] or scheduling['policy'] != 'FULL_RESIDENT_BACKBONE_SSD_ENGRAM' or
            scheduling['foreground_fallback'] or scheduling['foreground_engram_fallback'] or
            scheduling['consume_events'] <= 0 or
            scheduling['consume_events'] != scheduling['logical_match_consumptions']):
        raise ValueError('first-party production/SSD prefill invariant')
    if not any(r['suffix_tokens'] == 2049 for p in phases for r in p['turns']):
        raise ValueError('2K continuation evidence required')
    result['prefill_telemetry'] = scheduling
    result['bootstrap_s'] = initial['initial_decode']['bootstrap_s']
    result['suffix_bootstrap_s'] = {
        str(size): dict(count=len(rows), min=min(rows), median=statistics.median(rows), max=max(rows))
        for size in sorted({r['suffix_tokens'] for p in phases for r in p['turns']})
        if (rows := [r['bootstrap_s'] for p in phases for r in p['turns'] if r['suffix_tokens'] == size])
    }
    result['configured_max_seq_len'] = next((p['configured_max_seq_len'] for p in phases
                                                  if 'configured_max_seq_len' in p), None)
    result['max_qualified_frontier'] = max(p['final_frontier'] for p in phases)
    result['persistence'] = [dict(frontier=p['persisted_frontier'], bytes=p['artifact_bytes'],
        save_s=p['save_s'], save_performed=p.get('save_performed', True), restore_s=p.get('restore_s')) for p in phases]
    samples = [r for p in phases for r in p['resources']]
    result['resources']['min_system_available_bytes'] = min(r['system_available_bytes'] for r in samples)
    result['resources']['max_allocator_cache_bytes'] = max(r['mlx_cache_bytes'] for r in samples)
    result['resources']['per_process_swap_bytes'] = [dict(start=p['start_memory']['swap_used_bytes'],
        loaded=p['loaded_memory']['swap_used_bytes'], end=p['closed_memory']['swap_used_bytes']) for p in phases]
    def disk_delta(before, after):
        return {name: {key: counters[key]-before['disk_io'][name][key]
                      for key in ('read_bytes', 'write_bytes', 'read_count', 'write_count')}
                for name, counters in after.get('disk_io', {}).items() if name in before.get('disk_io', {})}
    result['global_disk_io_deltas'] = [dict(
        model_load=disk_delta(p['start_memory'], p['loaded_memory']),
        production=disk_delta(p['loaded_memory'], p['closed_memory'])) for p in phases]
    result['schema'] = 'ds41f.standard-off.very-long-frontier.v1'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phases', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = qualify([json.loads(p.read_text()) for p in args.phases])
    result['receipts'] = [str(p) for p in args.phases]
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['decision'] == 'QUALIFIED_BOUNDED_VERY_LONG_FIRST_PARTY_OFF' else 1


if __name__ == '__main__':
    raise SystemExit(main())
