"""Join actual CPU/API bridge observations and exported Metal System Trace.

Resolve xctrace ID references; filter target PID (never count WindowServer work).
Encoder intervals are NOT kernels. Device busy UNION is NOT summed kernel time.
Logical concatenation/import payloads are NOT DMA byte counters.
"""
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def table(path):
    with (gzip.open(path,'rb') if str(path).endswith('.gz') else open(path,'rb')) as stream:
        root=ET.parse(stream).getroot()
    definitions={e.get('id'):e for e in root.iter() if e.get('id') is not None}
    def resolve(e):
        return definitions[e.get('ref')] if e.get('ref') is not None else e
    def value(e):
        e=resolve(e)
        if e.tag=='sentinel': return None
        if e.tag=='process':
            p=next(resolve(v) for v in e if resolve(v).tag=='pid')
            return dict(pid=int(p.text),name=e.get('fmt'))
        if e.tag=='thread':
            return dict(tid=int(next(resolve(v).text for v in e if resolve(v).tag=='tid')),
                        name=e.get('fmt'))
        if e.tag=='tagged-backtrace':
            back=resolve(e[0])
            return [resolve(f).get('name') for f in back]
        if e.tag in ('start-time','duration','size-in-bytes','uint32','uint64','metal-command-buffer-id','gpu-hardware-trace','connection-uuid64','metal-nesting-level'):
            return int(e.text)
        return e.get('fmt') if e.get('fmt') is not None else e.text
    schemas=root.findall('.//schema')
    if not schemas: return []
    names=[c.findtext('mnemonic') for c in schemas[0].findall('col')]
    result=[]
    for r in root.findall('.//row'):
        assert len(r)==len(names), (path,len(r),len(names))
        result.append(dict(zip(names,[value(e) for e in r])))
    return result


def interval_union_ns(intervals):
    end=None;total=0
    for start,stop in sorted(intervals):
        assert stop>=start
        if end is None or start>end:
            total+=stop-start;end=stop
        elif stop>end:
            total+=stop-end;end=stop
    return total


def main():
    root=Path('artifacts/m50r');metal=root/'metal'
    movement=json.loads((root/'movement.json').read_text())
    capture=json.loads((metal/'capture.json').read_text())
    bridge=json.loads((root/'shared-bridge.json').read_text())
    assert movement['status']=='CAPTURED' and capture['status']=='CAPTURED' and bridge['status']=='PASS'
    assert movement['pid']==capture['probe']['pid']
    assert movement['identity_sha256']==bridge['identity_sha256']
    assert movement['retired'] and movement['turn']['queue_empty'] and movement['turn']['prediction_retired']
    toc=ET.parse(metal/'toc.xml')
    epoch_ns=int(datetime.fromisoformat(toc.findtext('.//summary/start-date')).timestamp()*1e9)
    trace_end_ns=int(datetime.fromisoformat(toc.findtext('.//summary/end-date')).timestamp()*1e9)
    begin=movement['observed_begin_wall_ns']-epoch_ns
    end=movement['observed_end_wall_ns']-epoch_ns
    assert epoch_ns<movement['observed_begin_wall_ns']<movement['observed_end_wall_ns']<trace_end_ns
    pid=movement['pid']
    # xctrace summary dates have millisecond resolution; object lifetimes may
    # begin before commit. Use 2 ms coverage padding, not device cost inference.
    def selected(rows):
        return [r for r in rows if (r.get('process') or {}).get('pid')==pid and
                r.get('start',r.get('timestamp',0)) <= end+2_000_000 and
                r.get('start',r.get('timestamp',0))+(r.get('duration') or 0)>=begin-2_000_000]
    paths=list(metal.glob('metal-*.xml'))
    paths+= [p for p in metal.glob('metal-*.xml.gz') if not p.with_suffix('').exists()]
    tables={p.name.removesuffix('.gz').removesuffix('.xml'):table(p) for p in paths}
    submissions=selected(tables['metal-application-command-buffer-submissions'])
    encoders=selected(tables['metal-application-encoders-list'])
    gpu=selected(tables['metal-gpu-intervals'])
    active=[r for r in gpu if r['state']=='Active']
    assert submissions and encoders and active, 'no target GPU coverage'
    assert min(r['start'] for r in submissions)<begin+500_000_000
    assert max(r['start']+r['duration'] for r in active)>end-500_000_000
    assert {r['cmdbuffer-id'] for r in active} <= {r['cmdbuffer-id'] for r in submissions}, 'unattributed target GPU work'
    allocations=selected(tables['metal-resource-allocations'])
    allocated=selected(tables['metal-current-allocated-size'])
    cpu=selected(tables['metal-application-intervals'])
    driver=selected(tables['metal-driver-intervals'])
    submit_ids={r['cmdbuffer-id'] for r in submissions}
    completed=[r for r in tables['metal-command-buffer-completed'] if r['cmdbuffer-id'] in submit_ids]
    events=movement['events']
    region_counts=Counter(e['label'] for e in events)
    by_phase={p:dict(Counter(e['label'] for e in events if e['phase']==p)) for p in {e['phase'] for e in events}}
    reads=[e for e in events if e['label']=='CPU selected-row copy/read']
    copied=sum(e['result'][0]['nbytes'] for e in reads)
    assert all(e['result'][0]['owns_data'] for e in reads)
    exports=[e for e in events if e['label']=='mlx-to-numpy export/materialization']
    decode=[e for e in events if e['label']=='CPU-to-MLX decode/import site']
    # Constructor payload size from actual rows and pinned dtype branches;
    # import ownership is separately tested. NOT a measured GPU DMA total.
    imports=Counter()
    for e in decode:
        raw,dtype=e['args'];payload=raw['nbytes']
        if dtype=='BF16': payload=raw['nbytes']*2
        elif dtype.startswith('F8_E8M0'): payload=raw['nbytes']*4
        imports[dtype]+=payload
    allocation_events=Counter(r['event-type'] for r in allocations)
    descriptions=Counter()
    for r in allocations:
        label=r['event-label'] or ''
        mode=re.search(r',\s*(Shared|Private|Managed)\s*\)',label)
        descriptions[mode.group(1) if mode else 'unclassified']+=1
    constructors=[e for e in events if e['label']=='native mlx.array constructor']
    materializations=[e for e in events if e['label'] in ('native array.item','native array.tolist')]
    assert constructors and materializations, 'native monitoring required, not sys.setprofile zero counts'
    assert all(e['native_return']=='return' for e in constructors+materializations)
    assert descriptions['Shared']==len(allocations), 'unclassified/private allocation route'
    def file_digest(path):
        data=gzip.decompress(path.read_bytes()) if str(path).endswith('.gz') else path.read_bytes()
        return hashlib.sha256(data).hexdigest()
    result=dict(schema='ds41f.m50r.movement-summary.v1',status='CAPTURED',
        identity_sha256=movement['identity_sha256'],request_sha256=movement['request_sha256'],
        pid=pid,trace_epoch_ns=epoch_ns,observation_window_relative_ns=[begin,end],
        table_sha256={p.name.removesuffix('.gz'):file_digest(p) for p in [*paths,metal/'toc.xml']},
        native_constructors=dict(count=len(constructors),
            numpy_source_count=sum(e['input_payload_nbytes'] is not None for e in constructors),
            numpy_source_payload_bytes=sum(e['input_payload_nbytes'] or 0 for e in constructors),
            source_kinds=dict(Counter(e['args'].get('kind','metadata') if isinstance(e['args'],dict) else type(e['args']).__name__ for e in constructors))),
        native_host_materializations=dict(count=len(materializations),
            shape_counts=dict(Counter(e['label']+':'+str(e['args']['shape']) for e in materializations)),
            caution='Materialization sites, not a count of actual waits/fences: already-ready shared data may need no device wait.'),
        actual_api_regions=dict(region_counts),api_regions_by_phase=by_phase,
        threads=movement['threads'],cpu_selected_row_copies=dict(count=len(reads),bytes=copied,owns_data=True),
        actual_host_exports=dict(count=len(exports),returned_payload_bytes=sum(e['result']['nbytes'] for e in exports),
            owning_cpu_copies=sum(e['result']['owns_data'] for e in exports)),
        native_import_sites=dict(count=len(decode),derived_constructor_payload_bytes_by_dtype=dict(imports),
            ownership_assay='native constructor snapshots copied CPU source; export is a non-owning shared view'),
        logical_concatenations=region_counts['mlx.concatenate'],
        metal=dict(command_buffer_submissions=len(submissions),encoder_intervals=len(encoders),
            encoder_labels=dict(Counter((r['encoder-label'] or '').split(' Command ')[0] for r in encoders)),
            gpu_interval_channels=dict(Counter(r['channel-name'] for r in active)),
            gpu_active_union_s=interval_union_ns((r['start'],r['start']+r['duration']) for r in active)/1e9,
            command_buffer_ids_with_completion_points=len({r['cmdbuffer-id'] for r in completed}),
            application_event_types=dict(Counter(r['event-type'] for r in cpu)),
            driver_types=dict(Counter(r['gpu-driver-name'] for r in driver)),
            resource_events=dict(allocation_events),resource_storage_modes=dict(descriptions),
            allocated_size_range_bytes=[min(r['current-allocated-size'] for r in allocated),max(r['current-allocated-size'] for r in allocated)]),
        limits=[
            'Native Python-to-nanobind constructor/item/tolist calls are observed with CPython monitoring, including all threads. C++-internal operations are not Python CALL events.',
            'Encoder/device intervals are observed, not kernel dispatches. Shader Timeline is disabled; compute-copy kernel bytes and individual internal barriers/page migrations are not measured.',
            'CPU/API phase boundaries may overlap device work. Device busy union is for the whole instrumented request; it is not uninstrumented device latency or additive proposal/verify/prefill cost.',
            'Native host imports copy into shared Metal storage; neither their payload sizes nor logical concatenations are discrete DMA volume.',
            'Warm identical prompt uses mapped CPU gathers/prefetch; absent pread events do not qualify cold SSD I/O.',
            'Internal MLX copy/fusion detail remains opaque: preservation requires unchanged pinned MLX/oMLX physical sources/binaries, matching bridge/eval/sync/shape path, Shared allocation route and initial matched-rate gate. A changed physical implementation or extra unclassified bridge requires remeasurement/BLOCK.'
        ])
    (root/'movement-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
