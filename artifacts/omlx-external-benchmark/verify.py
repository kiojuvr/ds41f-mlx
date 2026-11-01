"""Verify wire usage and render published values from official result dictionaries."""
import gzip
import json
from pathlib import Path

out = Path(__file__).resolve().parent
results = json.loads((out / 'results.json').read_text())
assert [r['requested_pp'] for r in results] == [1024,4096,8192,16384,32768,65536,131072,200000]
checks = []
rows = ['| Target prompt | Actual prompt / completion tokens | TTFT (s) | Processing estimate (tok/s) | Generation (tok/s) | End-to-end (s) |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
for r in results:
    label = f"pp{r['requested_pp']}"
    request = json.loads((out / f'{label}.request.json').read_text())
    assert request['temperature'] == 0 and request['max_tokens'] == 128
    assert request['reasoning_effort'] == 'none' and request['stream_options']['include_usage']
    text = (out / f'{label}.sse').read_text()
    chunks = [json.loads(line[5:].strip()) for line in text.splitlines()
              if line.startswith('data:') and line[5:].strip() != '[DONE]']
    usage = [c['usage'] for c in chunks if c.get('usage')][-1]
    assert usage['prompt_tokens'] == r['prompt_tokens']
    assert usage['completion_tokens'] == r['completion_tokens']
    assert usage['prompt_tokens_details']['cached_tokens'] == r['cached_tokens']
    finishes = [choice['finish_reason'] for c in chunks for choice in c.get('choices', [])
                if choice.get('finish_reason')]
    deltas = [choice.get('delta',{}) for c in chunks for choice in c.get('choices',[])]
    assert 'data: [DONE]' in text
    assert all(r[k] is not None and r[k] > 0 for k in ['ttft_ms','gen_tps','processing_tps','e2e_latency_s'])
    # Inputs are already rounded by upstream: allow propagation of 0.05 ms TTFT rounding.
    assert abs(r['processing_tps'] - r['prompt_tokens']/(r['ttft_ms']/1000)) < 0.1
    checks.append(dict(target_prompt=r['requested_pp'], usage=usage, finish_reasons=finishes,
        done=True, content_deltas=sum(bool(d.get('content')) for d in deltas),
        reasoning_deltas=sum(any(d.get(k) for k in ['reasoning_content','reasoning','analysis']) for d in deltas),
        heartbeat_lines=sum(line.startswith(':') for line in text.splitlines())))
    target = {1024:'1K',4096:'4K',8192:'8K',16384:'16K',32768:'32K',65536:'64K',131072:'128K',200000:'200K'}[r['requested_pp']]
    rows.append(f"| {target} | {r['prompt_tokens']:,} / {r['completion_tokens']:,} | {r['ttft_ms']/1000:.3f} | {r['processing_tps']:.1f} | {r['gen_tps']:.1f} | {r['e2e_latency_s']:.3f} |")
(out / 'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
(out / 'published-table.md').write_text('\n'.join(rows)+'\n')
# Operational identity is separate from the benchmark's timing results.
a = json.loads(gzip.decompress((out / 'identity-during.json.gz').read_bytes()))
z = json.loads(gzip.decompress((out / 'identity-final.json.gz').read_bytes()))
h0 = json.loads((out / 'initial.health.json').read_text())
h1 = json.loads((out / 'final.health.json').read_text())
state = dict(identity_checks={k:a[k] == z[k] for k in a if k != 'captured_utc'},
             ready=h1['status'] == 'ready', active_requests=h1['active_requests'],
             queued_requests=h1['queued_requests'],
             defaults_unchanged=h0['chat_defaults'] == h1['chat_defaults'],
             generation_policy_unchanged=h0['generation_policy'] == h1['generation_policy'],
             dependency_identity_unchanged=h0['dependency_identity'] == h1['dependency_identity'],
             profile_unchanged=h0['profile'] == h1['profile'],
             glm_profile_unchanged=h0['glm_profile'] == h1['glm_profile'],
             cache_counters_delta={k:h1['prefix_cache'][k] - h0['prefix_cache'][k]
                 for k in ['hits','misses','evictions','total_queries']})
(out / 'final-state.json').write_text(json.dumps(state, indent=2) + '\n')
assert all(state['identity_checks'].values())
assert state['ready'] and not state['active_requests'] and not state['queued_requests']
assert all(state[k] for k in ['defaults_unchanged','generation_policy_unchanged',
    'dependency_identity_unchanged','profile_unchanged','glm_profile_unchanged'])
readme = (out.parents[1] / 'README.md').read_text()
assert (out / 'published-table.md').read_text().strip() in readme
print('PASS: eight official results match streamed usage and README; Production identity/state checks pass')
