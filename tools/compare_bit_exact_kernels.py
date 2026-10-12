"""Compare the finite #4360 production receipts, excluding only measured timers."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import mean


def load(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)


def compare(a, b):
    result = dict(greedy_and_default_sampling=[])
    for x, y in zip(a['cases'][0]['rows'], b['cases'][0]['rows'], strict=True):
        assert x['label'] == y['label']
        checks = {k: x['trace'][k] == y['trace'][k] for k in
                  ('canonical_generated', 'target_offsets', 'dspark_offsets', 'canonical_frontier', 'generation_policy')}
        stats = lambda r: {k: v for k, v in r['trace']['mtp_stats'].items() if not k.endswith('_ms')}
        checks.update(mtp_counts=stats(x) == stats(y), output=x['response']['choices'] == y['response']['choices'],
                      usage=x['usage'] == y['usage'], complete_retained_paired_state=x['state'] == y['state'],
                      zero_replay_repack=all(r['trace'][k] == 0 for r in (x, y)
                                            for k in ('prompt_replay', 'full_cache_repack')))
        assert all(checks.values()), (x['label'], checks)
        result['greedy_and_default_sampling'].append(dict(label=x['label'], checks=checks,
            decode_before_s=x['trace']['decode_s'], decode_after_s=y['trace']['decode_s'],
            prefill_before_s=x['trace']['prefill_handoff_s'], prefill_after_s=y['trace']['prefill_handoff_s']))
    warm = lambda r: mean(x['trace']['decode_s'] for x in r['cases'][0]['rows'] if x['label'].startswith('hit'))
    da, db = warm(a), warm(b)
    result.update(warm_decode_mean_s=dict(before=da, after=db, speed_change_percent=100 * (da / db - 1)),
                  completion_tokens_per_decode_second=dict(before=64 / da, after=64 / db),
                  warm_two_token_prefill_s=dict(before=a['cases'][0]['rows'][2]['trace']['prefill_handoff_s'],
                                               after=b['cases'][0]['rows'][2]['trace']['prefill_handoff_s']),
                  peak_memory_bytes=dict(before=a['peak_memory_bytes'], after=b['peak_memory_bytes']),
                  real_operand_audit=b['real_operand_audit']['counts'])
    assert b['real_operand_audit']['bit_exact']
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(compare(load(args.before), load(args.after)), indent=2) + '\n')
