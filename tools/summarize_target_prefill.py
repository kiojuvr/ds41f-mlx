"""Summarize completed native target-prefill receipts, never additive GPU claims."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path); a=p.parse_args()
    native=json.loads((a.root/'native.json').read_text())
    controls=json.loads((a.root/'final-controls.json').read_text())
    assert native['admission']['native_profile']==controls['admission']['native_profile']=='native'
    assert controls['status']=='PASS'
    scaling=[]
    for x in native['rows']:
        if x.get('probe') or x['repeat']!=1: continue
        work=sum(s['count']*20+(24150 if s['mode']=='FINAL_ENCODER_DECODER' else 0) if s['mode']!='ORDINARY_COMPLETE_RANGE' else s['count']*40 for s in x['segments'])
        scaling.append(dict(length=x['length'],prefill_s=x['prefill_s'],tokens_per_s=x['tokens_per_s'],logical_block_rows=work,block_rows_per_s=work/x['prefill_s'],enqueue_s=x['enqueue_s'],completion_s=x['completion_s'],terminal_s=x['terminal_s'],decode_s=x['decode_s'],peak_GB=x['after']['peak']/1e9))
    comparisons=[]
    for n in sorted({x['length'] for x in controls['rows'] if x.get('bm8_control')}):
        base=[x for x in controls['rows'] if x['length']==n and not x.get('probe') and not x.get('bm8_control')]
        candidate=[x for x in controls['rows'] if x['length']==n and x.get('bm8_control')]
        for key in ('tokens','state_digest','continuation_tokens','continuation_state_digest','continuation_frontier'):
            assert len({json.dumps(x[key]) for x in base+candidate})==1,(n,key)
        b=statistics.median(x['prefill_s'] for x in base[1:]); c=statistics.median(x['prefill_s'] for x in candidate[1:])
        comparisons.append(dict(length=n,baseline_s=b,bm8_s=c,speedup=b/c,exact_settled_state_and_continuation=True,decode_baseline_s=statistics.median(x['decode_s'] for x in base),decode_candidate_s=statistics.median(x['decode_s'] for x in candidate)))
    # Observer barriers preserve state but change execution; compare exact state,
    # not percentages or the sum of its inclusive/exclusive stopwatch regions.
    for n in sorted({x['length'] for x in controls['rows'] if x.get('probe')}):
        rows=[x for x in controls['rows'] if x['length']==n and not x.get('bm8_control')]
        if any(not x.get('probe') for x in rows):
            for key in ('tokens','state_digest','continuation_tokens','continuation_state_digest'):
                assert len({json.dumps(x[key]) for x in rows})==1,(n,key)
    replays=[]
    for x in controls['rows']:
        for r in x.get('moe_replay',[]):
            trials={str(v['bm']):statistics.median(v['times_s'][1:]) for v in r['trials']}
            replays.append(dict(length=x['length'],repeat=x['repeat'],occupancy=r['occupancy'],seconds=trials,errors={str(v['bm']):v['max_abs_error'] for v in r['trials']}))
    suffix=json.loads((a.root/'suffix-controls.json').read_text())
    candidate=json.loads((a.root/'candidate-integrated.json').read_text())
    assert suffix['status']==candidate['status']=='PASS'
    suffix_comparisons=[]
    for n in (8192,32768,65536):
        baseline_source=suffix if n!=65536 else controls
        base=[x for x in baseline_source['rows'] if x['length']==n and not x.get('probe') and not x.get('suffix_control') and not x.get('p6_control')]
        after=[x for x in candidate['rows'] if x['length']==n]
        for key in ('tokens','continuation_frontier'):
            assert len({json.dumps(x[key]) for x in base+after})==1,(n,key)
        continuation_equal=len({json.dumps(x['continuation_tokens']) for x in base+after})==1
        def through_eos(ids): return ids[:ids.index(1)+1] if 1 in ids else ids
        eos_prefix_equal=len({json.dumps(through_eos(x['continuation_tokens'])) for x in base+after})==1
        b=base[-1]; c=after[-1]
        suffix_comparisons.append(dict(length=n,baseline_s=b['prefill_s'],candidate_s=c['prefill_s'],speedup=b['prefill_s']/c['prefill_s'],saved_s=b['prefill_s']-c['prefill_s'],baseline_completion_s=b['completion_s'],candidate_completion_s=c['completion_s'],decode_baseline_s=b['decode_s'],decode_after_s=c['decode_s'],continuation_decode_baseline_s=b['continuation_decode_s'],continuation_decode_after_s=c['continuation_decode_s'],tokens_equal=True,continuation_raw_steps_equal=continuation_equal,continuation_equal_through_first_eos=eos_prefix_equal,legacy_window_bytes_equal=False))
    assert all(x['source_and_history_exact'] for x in suffix['rows'] if x.get('suffix_control'))
    decode=json.loads((a.root/'decode-controls.json').read_text())
    assert decode['status']=='PASS'
    assert len({json.dumps(x['tokens']) for x in decode['rows']})==1
    paired=[]
    for mode in (False,True):
        rows=[x for x in decode['rows'] if x['suffix_control']==mode and x['repeat']>0]
        paired.append(dict(candidate=mode,prefill_median_s=statistics.median(x['prefill_s'] for x in rows),decode32_median_s=statistics.median(x['decode_s'] for x in rows),continuation8_median_s=statistics.median(x['continuation_decode_s'] for x in rows)))
    decode_regression=paired[1]['decode32_median_s']/paired[0]['decode32_median_s']-1
    qualification={}
    for name in ('native-state-qualification.json','r1-off.json'):
        if (a.root/name).exists():
            proof=json.loads((a.root/name).read_text())
            qualification[name]=dict(status=proof['status'],decision=proof.get('decision'))
    summary=dict(schema='ds41f.target-prefill.summary.v1',status='PASS',decision='NO_PRODUCTION_CHANGE',scaling=scaling,bm8_ab=comparisons,moe_replay=replays,suffix_reuse=suffix_comparisons,qualification=qualification,paired_decode=paired,decode_regression_fraction=decode_regression,warning='Serialized producer/replay timings are diagnostic, include dispatch/allocation, and must not be summed as production GPU time.')
    (a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for x in scaling:print(x['length'],round(x['prefill_s'],3),round(x['tokens_per_s'],1),round(x['block_rows_per_s']))
    print('BM8',comparisons)
    print('SUFFIX',suffix_comparisons)
    print('PAIRED DECODE',paired,'REGRESSION',decode_regression)
    print('QUALIFICATION (UNSELECTED CANDIDATE)',qualification)

if __name__=='__main__': main()
