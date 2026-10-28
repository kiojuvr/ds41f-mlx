"""Recompute bounded native scalar summary from retained receipts."""
import json
from pathlib import Path
import re
p = Path(__file__).parent/'native-01'
rows = [json.loads(s) for s in (p/'samples.jsonl').read_text().splitlines()]
host = [json.loads(s) for s in (p/'host.jsonl').read_text().splitlines()]
requests = json.loads((p/'requests.json').read_text())
summary = dict(requests=len(requests), first_pair_crossing=next(r for r in requests if r.get('legacy_pair_bytes_exceeded')),
    first_root_crossing=next(r for r in requests if r.get('legacy_root_bytes_exceeded')), last=requests[-1],
    maxima={k: max(r.get(k, 0) for r in rows) for k in ('footprint', 'lifetime_max_footprint', 'mlx_active',
        'mlx_peak', 'mlx_cache', 'retained_logical_bytes', 'pair_logical_bytes', 'roots')},
    minimum_reserve_proxy=min(r['reserve_proxy'] for r in host),
    swap_readings=sorted(set(r['swap'].strip() for r in host)),
    swapins=sorted(set(int(re.search(r'Swapins:\s+(\d+)', r['vm'])[1]) for r in host)),
    swapouts=sorted(set(int(re.search(r'Swapouts:\s+(\d+)', r['vm'])[1]) for r in host)),
    pressure_percent_range=[min(int(re.search(r'free percentage: (\d+)%', r['pressure'])[1]) for r in host),
                            max(int(re.search(r'free percentage: (\d+)%', r['pressure'])[1]) for r in host)],
    fresh_encodes=sum(r['vision_encoded_count'] for r in requests),
    replay=sum(r['prompt_replay'] for r in requests), repack=sum(r['full_cache_repack'] for r in requests),
    publications=sum(r['cache_published'] for r in requests),
    aligned=all(r['aligned'] for r in requests), captures_aligned=all(r['aligned'] for r in rows if 'aligned' in r),
    close=rows[-1])
(Path(__file__).parent/'native-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
