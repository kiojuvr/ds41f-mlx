"""Compare completed, real matched HTTP measurements; exclude warm-ups/failures."""
import json
from pathlib import Path


def main():
    root = Path('artifacts/m54-eof')
    files = dict(off='performance-off.json', first_party_mtp='performance-mtp.json', candidate='performance-candidate.json')
    summary = {}
    baseline = None
    digest = None
    for lane, name in files.items():
        data = json.loads((root/name).read_text())
        assert (root/name.replace('.json','.exit')).read_text().strip() == '0'
        assert 'error' not in data and 'MEASUREMENT COMPLETE' in data['decision']
        if digest is None: digest = data['request_sha256']
        assert data['request_sha256'] == digest
        rows = [r for r in data['rows'] if not r['warmup']]
        assert len(rows) == 3
        outcomes = [(r['generated_tokens'],r['frontier'],r['response']['choices'][0]['message'],r['response']['choices'][0]['finish_reason']) for r in rows]
        assert all(o == outcomes[0] for o in outcomes)
        if baseline is None: baseline = outcomes[0]
        assert all(o == baseline for o in outcomes), 'not a matched correctness workload'
        phases = {k:sum(r['phase_s'].get(k,0) for r in rows)/len(rows)
                  for k in sorted({k for r in rows for k in r['phase_s']})}
        offered = sum((r['acceptance'] or {}).get('offered',0) for r in rows)
        accepted = sum((r['acceptance'] or {}).get('accepted',0) for r in rows)
        summary[lane] = dict(samples=len(rows), tokens_per_sample=rows[0]['tokens'], frontier=rows[0]['frontier'],
            integrated_recipe_decode_tok_s=sum(r['tokens'] for r in rows)/sum(r['end_to_end_decode_s'] for r in rows),
            complete_http_request_tok_s=sum(r['tokens'] for r in rows)/sum(r['http_latency_s'] for r in rows),
            mean_decode_s=sum(r['end_to_end_decode_s'] for r in rows)/len(rows),
            mean_http_request_s=sum(r['http_latency_s'] for r in rows)/len(rows),
            acceptance=None if not offered else accepted/offered,
            offered=offered,accepted=accepted,mean_phase_s=phases,resources=data['resources'])
    record = dict(decision='MATCHED PERFORMANCE / CORRECTNESS MEASUREMENT PASS; NO PROMOTION',
        request_sha256=digest, generated_token_identity=True,message_identity=True,frontier_identity=True,
        lanes=summary, warmup='one warm-up excluded per lane; three sequential fresh-session samples',
        method='sparse Python profiling observes original call boundaries; inclusive host wall times, no duplicated execution or new synchronization',
        decode_definition='normal: canonical recipe turn construction to idle/final recipe completion; candidate: first decode step to settled protocol completion, including async scheduling/reporting',
        request_definition='actual loopback HTTP request latency, includes prefill/handoff and transport/final response overhead; excludes model load',
        phase_limits='inclusive phases overlap; target_forward includes bootstrap and overlaps target_verify. M51 setup/settlement separated from verification. Native candidate cache_ops is not M51. OFF protected_target_step combines target execution and canonical sampling. Response/application phase is final worker publication/freezing, not all route/transport overhead.',
        candidate='existing InternalMTPQualificationBackend on standard routes with delivered patched oMLX; not an MTP default/public-profile promotion. Proposal widths remain each existing implementation setting.',
        historical='M53 13.81 tok/s remains core-only, not a matched operational measurement',
        exclusions=['pre-worker-race MTP measurement','candidate missing locked jsonschema dependency attempt','candidate executor self-join harness attempt'])
    (root/'matched-performance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
