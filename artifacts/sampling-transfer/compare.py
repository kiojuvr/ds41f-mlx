"""Reproduce the bounded candidate receipt comparisons; no runtime/admission hooks."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def load(name):
    return json.loads((HERE / name).read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics_equal(a, b):
    keys = ('prompt_tokens', 'cached_tokens', 'remaining_suffix_tokens', 'generated',
            'canonical_frontier', 'aligned', 'accepted_drafts', 'proposed_drafts',
            'prompt_replay', 'full_cache_repack')
    assert {k:a[k] for k in keys} == {k:b[k] for k in keys}
    return {k:a[k] for k in keys}
def choice_without_request_ids(response):
    value = json.loads(json.dumps(response['choices']))
    for choice in value:
        for call in choice['message'].get('tool_calls', []):
            # Tool call IDs derive from the unique HTTP request, not sampling.
            call['id'] = '<request-correlated>'
    return value

result = dict(status='PASS', scope='shared sampling source/maintenance transfer; native installed candidates, not stable promotion',
              start_commit='d2a24ad', behavioral_authority='DS41F-STABLE-e8c719c', packages=[], http={}, mtp={}, off={})
stable = json.loads((ROOT / 'release/stable-e8c719c/baseline.json').read_text())
assert load('baseline-mtp-acceptance.json')['ordinary']['health']['dependency_identity'] == next(
    p['runtime_execution_identity_sha256'] for p in stable['packages'] if p['profile'] == 'mtp' and p['glm'] == 'native')
assert load('baseline-off-acceptance.json')['manifest_sha256'] == next(
    p['manifest_sha256'] for p in stable['packages'] if p['profile'] == 'standard-off' and p['glm'] == 'native')
for profile, lane in [('mtp', 'mtp-native'), ('off', 'off-native')]:
    wheel = next((HERE / 'wheels' / profile).glob('*.whl'))
    baseline_wheel = next((ROOT / 'release/stable-e8c719c/wheels' / lane).glob('*.whl'))
    with zipfile.ZipFile(wheel) as z:
        manifest_bytes = z.read('ds41f_mlx/execution_manifest.json')
        manifest = json.loads(manifest_bytes)
        for name, digest in manifest['payloads']['ds41f_mlx'].items():
            assert sha(ROOT / 'ds41f_mlx' / name) == digest, name
    with zipfile.ZipFile(baseline_wheel) as z:
        before = json.loads(z.read('ds41f_mlx/execution_manifest.json'))
    assert before['payloads']['omlx']['utils/sampling.py'] == '8cd19ae59dd5802b38ae1825098f0ecdfdb13c7ed821bfcab32a24864b6fc131'
    for key in ('semantic_native', 'files', 'checkpoint', 'checkpoint_metadata', 'tokenizer_sha256', 'glm'):
        assert before[key] == manifest[key], key
    assert {k:v for k,v in before['payloads'].items() if k != 'ds41f_mlx'} == {
        k:v for k,v in manifest['payloads'].items() if k != 'ds41f_mlx'}
    delta = sorted(k for k in before['payloads']['ds41f_mlx'].keys() | manifest['payloads']['ds41f_mlx'].keys()
                   if before['payloads']['ds41f_mlx'].get(k) != manifest['payloads']['ds41f_mlx'].get(k))
    assert delta == ['sampling.py', 'serving/deepseek_recipe_backend.py', 'serving/generation_policy.py', 'serving/internal_mtp.py']
    result['packages'].append(dict(profile=profile, wheel=str(wheel.relative_to(ROOT)), wheel_sha256=sha(wheel),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), baseline_wheel_sha256=sha(baseline_wheel),
        source_payload_sha256=manifest['provenance']['source_payload_sha256'],
        source_commit_is_base=manifest['provenance']['source_commit_is_base'], delta=delta,
        dependencies_native_assets_exact=True))
    b, c = load(f'baseline-{profile}-http.json'), load(f'candidate-{profile}-http.json')
    assert b['status'] == c['status'] == 'PASS'
    assert b['health']['generation_policy'] | {'sampler_authority': 'ds41f_mlx.sampling.make_sampler'} == c['health']['generation_policy']
    assert b['health']['chat_defaults'] == c['health']['chat_defaults']
    for a, d in zip(b['requests'], c['requests']):
        assert a['body'] == d['body']
        assert a['response']['choices'] == d['response']['choices']
        assert a['response']['usage'] == d['response']['usage']
        if profile == 'mtp':
            metrics_equal(a['settlement']['metrics'], d['settlement']['metrics'])
    rows = []
    for temperature in (0., 1.):
        for field in (['elapsed_s'] if profile == 'off' else ['elapsed_s', 'decode_s', 'total_ttft_s']):
            def median(r):
                return statistics.median([x[field] if field == 'elapsed_s' else x['settlement']['metrics'][field]
                    for x in r['requests'] if x['temperature'] == temperature and not x['warmup']])
            old, new = median(b), median(c)
            assert not (new-old > .1 and new > old*1.15), (profile, temperature, field, old, new)
            rows.append(dict(temperature=temperature, metric=field, baseline_median_s=old, candidate_median_s=new))
    if profile == 'off':
        old = max(x['rss_kib'] for x in b['requests']) * 1024
        new = max(x['rss_kib'] for x in c['requests']) * 1024
        assert not (new-old > 256*1024**2 and new > old*1.02)
        result['off']['sampled_process_rss_bytes'] = dict(baseline=old, candidate=new, note='sampled RSS, not peak MLX/wired memory')
        assert b['cancel_settlement'] is c['cancel_settlement'] is None
    else:
        assert b['cancel_settlement']['finish_reason'] == c['cancel_settlement']['finish_reason'] == 'abort'
        result['mtp']['cancel_metrics'] = metrics_equal(b['cancel_settlement']['metrics'], c['cancel_settlement']['metrics'])
    assert b['recovery']['choices'] == c['recovery']['choices']
    result['http'][profile] = dict(result='exact choices/usage and MTP target/draft/settlement counts',
                                  measured_requests_per_implementation=6, output_tokens_per_request=64, medians=rows)

b, c = load('baseline-mtp-acceptance.json'), load('candidate-mtp-acceptance.json')
for label in ('call', 'continuation', 'branch'):
    assert choice_without_request_ids(b['ordinary'][label]) == choice_without_request_ids(c['ordinary'][label])
    assert b['ordinary'][label]['usage'] == c['ordinary'][label]['usage']
result['mtp']['tools_append_branch_sse_counts'] = [metrics_equal(x['metrics'], y['metrics'])
    for x,y in zip(b['ordinary_settlements'], c['ordinary_settlements'])]
for x,y in zip(b['paired_session_stages'], c['paired_session_stages']):
    for key in ('canonical_frontier', 'request_count', 'state'):
        assert x[key] == y[key]
    assert x['last_turn']['response_json']['choices'] == y['last_turn']['response_json']['choices']
    assert x['diagnostics']['m8'] == y['diagnostics']['m8']
result['mtp']['paired_session_frontiers'] = [x['canonical_frontier'] for x in c['paired_session_stages']]
load_last = []
for name in ('baseline-start.log', 'candidate-mtp-start.log'):
    lines = (HERE / name).read_text().splitlines()
    load_last.append([line.split('DeepSeek V4.1 loaded ')[-1] for line in lines if 'DeepSeek V4.1 loaded ' in line][-1])
assert load_last[0] == load_last[1]
result['mtp']['last_reported_model_load_allocation'] = load_last[0]
# First uncached tools observations are single, not repeated performance samples.
result['mtp']['uncached_single_observations_s'] = dict(
    frozen_stable=json.loads((ROOT / 'release/stable-e8c719c/evidence/performance-comparison.json').read_text())['rows'][0]['candidate']['total_ttft_s'],
    fresh_baseline=b['ordinary_settlements'][0]['metrics']['total_ttft_s'],
    candidate=c['ordinary_settlements'][0]['metrics']['total_ttft_s'],
    disposition='Fresh cold difference inconclusive; isolated to unchanged prefill, candidate within frozen stable practical observation. Not a cold-start performance PASS or SLA.')
b, c = load('baseline-off-acceptance.json'), load('candidate-off-acceptance.json')
assert b['status'] == c['status'] == 'PASS'
for step in ('p5', 'p6'):
    def reports(x):
        return [{k:v for k,v in r.items() if k != 'latency_s'} for r in x[step]['reports']]
    assert reports(b) == reports(c)
    for key in ('frontier', 'same_list', 'replay', 'repack'):
        assert b[step][key] == c[step][key]
    result['off'][step] = reports(c)
assert b['successful_model_native_calls'] == c['successful_model_native_calls']
assert b['retirement'] == c['retirement']
result['off']['native_calls_and_retirement_exact'] = True
b,c = load('baseline-off-tools.json'),load('candidate-off-tools.json')
assert choice_without_request_ids(b['call']) == choice_without_request_ids(c['call'])
assert b['call']['usage'] == c['call']['usage'] and b['result_text'] == c['result_text']
result['off']['tool_call_result_sse_exact'] = True
result['limits'] = ['Native candidates only; no portable requalification or new stable promotion.',
    'Cold observations are not a statistical performance qualification.',
    'OFF control API does not project request/cancel metrics; disconnect followed by serialized explicit recovery, direct P5/P6 and retained lifecycle tests cover settlement.',
    'No new MTP peak-memory measurement; exact source/operator graph, same final reported model-load allocation and retained stable resource evidence apply.']
result['remaining_concrete_blocker'] = None
result['boundary_complete'] = True
# Bind retained raw receipts/scripts/logs; do not recursively hash the summary itself.
result['evidence'] = {str(p.relative_to(HERE)):dict(sha256=sha(p), bytes=p.stat().st_size)
    for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'qualification.json'
    and '__pycache__' not in p.parts and p.suffix in ('.json', '.log', '.py')}
(HERE / 'qualification.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS: exact owned payload/dependencies, matched semantics/counts/state and bounded warm practical screening')
