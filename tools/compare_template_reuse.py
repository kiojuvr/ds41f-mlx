#!/usr/bin/env python3
"""Summarize finite #4354 receipts and assert generation/state equality."""
import gzip
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]/'artifacts/template-reuse'
def load_production(version):
    path = root/f'production-{version}.json.gz'
    return json.loads(gzip.decompress(path.read_bytes()))

before = load_production('before')
after = load_production('after')

def project(value):
    if isinstance(value, dict):
        return {k:project(v) for k,v in value.items() if k not in ('id','created','lifetime') and not k.endswith(('_s','_ms'))}
    if isinstance(value,list):
        return [project(v) for v in value]
    return value

rows=[]
warm=[[],[]]
for bcase,acase in zip(before['cases'],after['cases']):
    assert bcase['requested_tokens']==acase['requested_tokens']
    for b,a in zip(bcase['rows'],acase['rows']):
        assert b['label']==a['label']
        assert b['state']==a['state']
        assert project(b['response'])==project(a['response'])
        for key in ('canonical_generated','mtp_stats','canonical_frontier','target_offsets','dspark_offsets','cached_tokens','remaining_suffix_tokens','quiescence','authority_connection','protocol_events','prompt_replay','full_cache_repack','prediction_retired','queue_empty','cancelled','cache_published'):
            assert project(b['trace'][key])==project(a['trace'][key]), key
        for r in (b,a):
            t=r['trace']
            assert t['prompt_replay']==t['full_cache_repack']==0
            assert set(t['target_offsets']+t['dspark_offsets'])=={t['canonical_frontier']}
        if b['label'].startswith('hit'):
            warm[0].append(b['trace']['decode_s']);warm[1].append(a['trace']['decode_s'])
        rows.append(dict(prompt_tokens=b['usage']['prompt_tokens'],label=b['label'],exact=True,
                         wall_s=[b['wall_s'],a['wall_s']],ttft_s=[b['trace']['total_ttft_s'],a['trace']['total_ttft_s']],
                         decode_s=[b['trace']['decode_s'],a['trace']['decode_s']],
                         new_custom_identities=[len(r.get('new_custom_identities',[])) for r in (b,a)]))
leaf={}
for version in ('before','after'):
    receipts=[json.loads((root/f'{version}-leaf-{n}.json').read_text()) for n in (0,1)]
    leaf[version]={}
    for family in dict.fromkeys(r['family'] for r in receipts[0]['rows']):
        sets=[[r for r in j['rows'] if r['family']==family] for j in receipts]
        leaf[version][family]=[dict(size=sets[0][i]['size'],identities=sets[0][i]['new_identities'],first_ms=statistics.mean(rr[i]['first_ms'] for rr in sets),warm_ms=statistics.mean(rr[i]['warm_median_ms'] for rr in sets)) for i in range(len(sets[0]))]
    leaf[version]['identities_including_unadopted_paired_negative_control']=receipts[0]['total_identities']
    leaf[version]['peak_memory_bytes']=receipts[0]['peak_memory_bytes']
warm_s=[statistics.mean(v) for v in warm]
summary=dict(production_rows=rows,all_exact=True,replay_repack=0,
             warm_hit_mean_decode_s=warm_s,warm_hit_tokens_per_decode_second=[64/v for v in warm_s],
             warm_decode_time_change_percent=100*(warm_s[1]/warm_s[0]-1),
             production_peak_memory_bytes=[before['peak_memory_bytes'],after['peak_memory_bytes']],leaf=leaf)
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('production_rows','leaf')},indent=2))
