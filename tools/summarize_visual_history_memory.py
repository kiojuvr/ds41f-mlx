#!/usr/bin/env python3
"""Recompute bounded visual-history probe receipts without loading MLX/model."""
import argparse
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    rows = [json.loads(x) for x in (a.run/'samples.jsonl').read_text().splitlines()]
    hosts = [json.loads(x) for x in (a.run/'host.jsonl').read_text().splitlines()]
    result = json.loads((a.run/'result.json').read_text())
    assert result['status'] == 'COMPLETED' and not result['unsafe']
    successful = []
    reuse = 0
    previous_images = []
    for row in rows:
        if row['event'] != 'prepare:after' or row['case'] in ('queued-cancel', 'identity-miss', 'historical-miss'):
            continue
        images = row['identities']
        assert images[:len(previous_images)] == previous_images
        previous_images = images
        t = json.loads((a.run/(row['case']+'-trace.json')).read_text())
        C = t['canonical_frontier']
        assert len(t['target_offsets']) == 40 and len(t['dspark_offsets']) == 3
        assert set(t['target_offsets']+t['dspark_offsets']) == {C}
        assert t['queue_empty'] and t['prediction_retired'] and t['cache_published']
        assert t['prompt_replay'] == t['full_cache_repack'] == 0
        assert t['quiescence']['counters']['history_replay'] == 0
        assert t['quiescence']['counters']['full_cache_repack'] == 0
        cached = t['cached_tokens']
        fresh = sum(s['start']+s['length'] > cached for s in images)
        assert fresh == t['vision_encoded_count']
        assert all(not old or s['start']+s['length'] <= cached for old, s in zip(row['historical'], images))
        reuse += len(images)-fresh
        successful.append(dict(case=row['case'], images=len(images), prompt_tokens=row['prompt_tokens'],
            cached=cached, suffix=t['remaining_suffix_tokens'], canonical=C, fresh=fresh,
            vision_s=t['vision_s'], restore_s=t['paired_restore_s'], elapsed_s=t['elapsed_s']))
    maxima = {}
    for field in ('footprint', 'resident', 'wired', 'mlx_active', 'mlx_cache', 'mlx_peak',
                  'python_current', 'python_peak', 'root_logical_bytes', 'roots'):
        row = max(rows, key=lambda x: x.get(field, 0))
        maxima[field] = dict(value=row.get(field, 0), case=row['case'], event=row['event'])
    events = {}
    for event in sorted({x['event'] for x in rows}):
        selected = [x for x in rows if x['event'] == event]
        events[event] = dict(n=len(selected), **{field: max(x.get(field, 0) for x in selected)
            for field in ('footprint', 'mlx_active', 'mlx_cache', 'python_current')})
    deltas = []
    for i, row in enumerate(rows):
        if row['event'] not in ('restore:before', 'capture:before', '_encode_new_images:before'):
            continue
        after = rows[i+1]
        assert after['event'] == row['event'].replace(':before', ':after')
        deltas.append(dict(case=row['case'], stage=row['event'].split(':')[0],
            **{field: after.get(field, 0)-row.get(field, 0)
               for field in ('mlx_active', 'footprint', 'python_current')}))
    def pages(host, field):
        return int(re.search(re.escape(field)+r':\s+(\d+)', host['vm'])[1])*16384
    for h in hosts:
        assert 'used = 0.00M' in h['swap']
        assert pages(h, 'Swapins') == pages(h, 'Swapouts') == 0
    summary = dict(kind='MEASURED REAL MODEL; stage endpoints, not continuous process high-water',
        successful=successful, fresh_encodes=sum(x['fresh'] for x in successful),
        historical_reuses=reuse, maxima=maxima, stage_endpoint_maxima=events,
        deltas=deltas, host=dict(samples=len(hosts), max_system_wired=max(pages(h, 'Pages wired down') for h in hosts),
            min_free=min(pages(h, 'Pages free') for h in hosts),
            min_free_inactive_speculative=min(sum(pages(h, f) for f in ('Pages free', 'Pages inactive', 'Pages speculative')) for h in hosts),
            compressor_pages=sorted({pages(h, 'Pages occupied by compressor')//16384 for h in hosts}),
            swapins=0, swapouts=0),
        final=rows[-1], expected_failures=[x for x in result['traces'] if x.get('error')])
    assert maxima['roots']['value'] == 18 and rows[-1]['roots'] == 0
    (a.run/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ('fresh_encodes', 'historical_reuses', 'maxima', 'host')}, indent=2))


if __name__ == '__main__':
    main()
