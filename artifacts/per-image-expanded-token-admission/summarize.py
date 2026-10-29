"""Recompute scalar resource summary from the finite native receipts."""
import json
from pathlib import Path
import re
p = Path(__file__).parent/'native-01'
samples = [json.loads(x) for x in (p/'samples.jsonl').read_text().splitlines()]
host = [json.loads(x) for x in (p/'host.jsonl').read_text().splitlines()]
requests = json.loads((p/'requests.json').read_text())
loaded = next(r for r in samples if r['event']=='loaded')
settled = [r for r in samples if r['event']=='settled']
assert len(requests)==len(settled)==8
summary = dict(requests=8,fresh_encodes=sum(r['vision_encoded_count'] for r in requests),
    aligned=all(r['aligned'] for r in requests),published=all(r['cache_published'] for r in requests),
    replay=sum(r['prompt_replay'] for r in requests),repack=sum(r['full_cache_repack'] for r in requests),
    per_request=[dict(case=r['case'],cached_tokens=r['cached_tokens'],canonical_frontier=r['canonical_frontier'],
        fresh_encodes=r['vision_encoded_count'],peak_mlx_active=s['mlx_peak'],
        settled_mlx_active=s['mlx_active'],root_logical_bytes=s['root_logical_bytes'])
        for r,s in zip(requests,settled)],
    maxima={k:max(r.get(k,0) for r in samples) for k in ('footprint','lifetime_max_footprint',
        'mlx_active','mlx_peak','mlx_cache','root_logical_bytes','roots')},
    minimum_reserve_proxy=min(r['reserve_proxy'] for r in host),
    pressure_percent_range=[min(int(re.search(r'free percentage: (\d+)%',r['pressure'])[1]) for r in host),
                            max(int(re.search(r'free percentage: (\d+)%',r['pressure'])[1]) for r in host)],
    swap_readings=sorted(set(r['swap'].strip() for r in host)),
    swapins=sorted(set(int(re.search(r'Swapins:\s+(\d+)',r['vm'])[1]) for r in host)),
    swapouts=sorted(set(int(re.search(r'Swapouts:\s+(\d+)',r['vm'])[1]) for r in host)),
    loaded=loaded,closed=samples[-1],active_bytes_after_close_minus_loaded=samples[-1]['mlx_active']-loaded['mlx_active'])
(p.parent/'native-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
