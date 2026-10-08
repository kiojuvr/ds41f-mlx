"""Validate supplementary M50R evidence; no runtime admission or M51R approval."""
import hashlib
import json
from pathlib import Path


def qualified_prefix(receipt):
    if receipt.get('status')!='PASS' or receipt.get('archive_imports') is not False:
        return False
    if receipt.get('oracle_live_scheduler',receipt.get('live_scheduler')) is not False:
        return False
    cases=receipt.get('cases',[])
    if {(c['before'],c['accepted']) for c in cases}!={(b,m) for b in (127,128,255,256) for m in range(4)} or len(cases)!=16:
        return False
    live=receipt.get('live_cases',[])
    if [(c['cycle'],c['accepted']) for c in live]!=[(1,3),(10,2)]: return False
    for c in cases+live:
        if not all(c.get(k) is True for k in ('state_content_exact','tap_content_exact','logit_content_exact')): return False
        if c.get('state_mismatches')!=[] or c.get('ring_content_exact')!=[True]*3: return False
        pubs=c.get('index_publications_exact',{})
        if len(pubs)!=16 or not all(v is True for v in pubs.values()): return False
        n=c['accepted']+1
        if c['input_ids'][:n]!=c['oracle_ids'][:n]: return False
        if n<4 and c['input_ids'][n:]==c['oracle_ids'][n:]: return False
    return all(c.get('metadata_exact') is True and len(c['expected_slot_digests'])==40 and
               all(len(slots)==9 for slots in c['expected_slot_digests']) for c in cases) and all(
               c.get('owner_retired_before_oracle') is True and
               c.get('ring_append_input_matches_same_forward_taps') is True for c in live)


def supplement(root,identity_sha,request_sha,baseline_rows):
    required=('prefix-oracle','movement','movement-summary','shared-bridge','environment/summary')
    missing=[name for name in required if not (root/(name+'.json')).exists()]
    if missing:
        return dict(blockers=['Independent prefix-state / native movement / bounded environment closure receipts missing: '+', '.join(missing)])
    receipts={n:json.loads((root/(n+'.json')).read_text()) for n in required}
    for name in ('prefix-oracle','movement','shared-bridge','environment/summary'):
        r=receipts[name]
        tool=root/('environment/driver.tool.py' if name=='environment/summary' else name+'.tool.py')
        assert hashlib.sha256(tool.read_bytes()).hexdigest()==r['tool_sha256'],name
    for name in ('prefix-oracle','movement','movement-summary','shared-bridge'):
        assert receipts[name]['identity_sha256']==identity_sha,name
    oracle=receipts['prefix-oracle'];movement=receipts['movement'];physical=receipts['movement-summary']
    assert qualified_prefix(oracle),'prefix byte-content qualification incomplete'
    assert movement['status']=='CAPTURED' and movement['retired']
    assert movement['request_sha256']==physical['request_sha256']==request_sha
    assert movement['turn']['canonical_generated']==baseline_rows[0]['trace']['canonical_generated']
    assert receipts['shared-bridge']['status']=='PASS'
    assert physical['native_constructors']['numpy_source_count']>0
    assert physical['native_host_materializations']['shape_counts']['native array.tolist:[8]']==10
    assert physical['metal']['encoder_labels']=={'Compute':physical['metal']['encoder_intervals']}
    assert physical['metal']['resource_storage_modes']=={'Shared':sum(physical['metal']['resource_events'].values())}
    assert physical['metal']['command_buffer_ids_with_completion_points']==physical['metal']['command_buffer_submissions']
    import gzip
    for name,digest in physical['table_sha256'].items():
        p=root/'metal'/name
        data=p.read_bytes() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes())
        assert hashlib.sha256(data).hexdigest()==digest,name
    env=receipts['environment/summary']
    assert env['status']=='CAPTURED'
    assert [c['name'] for c in env['cases']]==['reload-a','pressure-a','reload-b','pressure-b']
    rows=list(baseline_rows);groups=[]
    for c in env['cases']:
        assert c['status']=='CAPTURED'
        p=root/'environment'/(c['name']+'.json');r=json.loads(p.read_text())
        assert hashlib.sha256(p.with_suffix('.tool.py').read_bytes()).hexdigest()==r['tool_sha256']
        assert r['identity']['identity_sha256']==identity_sha
        measured=[v for v in r['rows'] if not v['warmup']]
        assert r['rows'][0]['warmup'] and len(measured)>=3
        for v in measured:
            assert v['request_sha256']==request_sha
            assert v['trace']['canonical_generated']==baseline_rows[0]['trace']['canonical_generated']
            assert v['trace']['mtp_stats']['depth_drafted']==baseline_rows[0]['trace']['mtp_stats']['depth_drafted']
            assert v['trace']['mtp_stats']['depth_accepted']==baseline_rows[0]['trace']['mtp_stats']['depth_accepted']
            assert v['retired'] and v['trace']['canonical_frontier']==283
        if c['co_resident_gib']:
            assert c['co_resident_gib']==64 and c['pressure_ready']['bytes']==64*1024**3
            assert c['pressure_end_rss_bytes']>=c['pressure_ready']['bytes']
        rows+=measured
        groups.append(dict(name=c['name'],load_s=c['load_s'],available_range_bytes=c['available_range_bytes'],
            swap_used_range=c['swap_used_range'],decode=c['dispersion']['decode_tok_s']))
    from tools.freeze_m50r_candidate import dispersion
    decode=dispersion([r['decode_tok_s'] for r in rows])
    verify=dispersion([r['trace']['mtp_stats']['backbone_ms']/1000 for r in rows])
    assert (decode['max']-decode['min'])/decode['mean']<.01
    assert (verify['max']-verify['min'])/verify['mean']<.01
    return dict(blockers=[],prefix=dict(primitive_cases=16,live_cases=2,byte_exact=True,
        oracle_scheduler_live=False,arithmetic='same qualified MLX/oMLX width-4 arithmetic; independent CPU prefix/layout selectors, adversarial future and deferred actual-candidate captures'),
        movement=physical,environment=dict(groups=groups,measured_sessions=len(rows),decode=decode,verify_wall_s=verify,
            empirical_decode_full_range_fraction=(decode['max']-decode['min'])/decode['mean'],
            empirical_verify_full_range_fraction=(verify['max']-verify['min'])/verify['mean'],
            limitations='one host/day; process reload NOT filesystem/JIT/Engram-cold; bounded touched 64 GiB co-residency NOT severe OS pressure; original six resident samples unchanged'),
        residual_observability=dict(blocks_changed_physical_implementation=True,
            blocks_unchanged_delegated_primitive_graph=False,
            reason='Metal records real Shared resources, encoder/execution/completion intervals; native constructor/item/tolist and bridge/eval/sync sites are observed. Internal compute-copy kernels, per-kernel barriers and page migrations are opaque. Conformance requires unchanged qualified MLX/native binaries and oMLX physical functions, reviewed identical primitive graph/shape/dtype/ownership/async-lifetime bindings, numerical oracle and matched rates. Changes outside this preserved graph require new diagnostics and BLOCK, not inference from these intervals.'))
