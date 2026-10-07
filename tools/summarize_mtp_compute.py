"""Summarize real R1 receipts and bounded observer data; no inferred GPU timings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import gzip
import json
from pathlib import Path
from statistics import median


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    root = a.directory
    summary = dict(schema='ds41f.mtp.compute-evidence.v1', files={}, baselines={})
    for name in ('r1-off.json', 'r1-mtp.json', 'r1-observed-mtp.json', 'r1-serialized-mtp.json', 'r1-low-overhead-mtp.json', 'r1-head-observer-mtp.json', 'r1-attention-observer-mtp.json'):
        path = root/name
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        summary['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        row = dict(status=value['status'], forced_shutdown=value.get('forced_shutdown', False))
        model = value
        if 'gates' in value:
            row['gates'] = len(value['gates'])
            lane = 'off-model.json' if 'off' in name else 'mtp-model.json'
            model_path = root/(path.stem+'-gates')/lane
            if model_path.exists():
                model = json.loads(model_path.read_text())
                summary['files'][str(model_path.relative_to(root))] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        cases = []
        for case in model.get('cases', []):
            metrics = case.get('outcome', {}).get('metrics', {})
            cases.append(dict(name=case.get('name',case.get('route')), wall_s=case.get('wall_s'),
                              metrics=metrics, outcome=case.get('outcome',{}).get('outcome_state')))
        row['cases'] = cases
        summary['baselines'][name] = row
    for filename in ('observer.json', 'serialized.json', 'low-overhead.json', 'head-observer.json', 'attention-observer.json'):
        path = root/filename
        if path.exists():
            data = path.read_bytes()
        elif path.with_suffix('.json.gz').exists():
            data = gzip.decompress(path.with_suffix('.json.gz').read_bytes())
        else:
            continue
        value = json.loads(data)
        summary['files'][filename] = hashlib.sha256(data).hexdigest()
        shapes = Counter()
        for operation in value['operations']:
            shapes[json.dumps({k:v for k,v in operation.items() if k != 'cycle'}, sort_keys=True)] += 1
        route_groups = defaultdict(list)
        for route in value['routes']:
            route_groups[(route['phase'], route['input_shape'][1])].append(route)
        routes = []
        for (phase, rows), items in route_groups.items():
            histogram = Counter()
            for item in items:
                histogram.update({int(k):v for k,v in item['rows_per_active_expert_histogram'].items()})
            total_routes = sum(item['routes'] for item in items)
            active = sum(item['active_experts'] for item in items)
            routes.append(dict(phase=phase, global_rows=rows, layer_calls=len(items),
                total_routes=total_routes, total_active_expert_calls=active,
                rows_per_active_expert_histogram=dict(sorted(histogram.items())),
                mean_rows_per_active_expert=total_routes/active,
                singleton_route_fraction=histogram[1]/total_routes,
                ideal_weight_read_reuse_bound=total_routes/active,
                max_rows=max(item['max_rows'] for item in items)))
        attention = Counter()
        for operation in value['operations']:
            if operation['operation'] == 'packed_attention':
                wi, ci = operation['shapes'][3][-1], operation['shapes'][4][-1]
                attention[(wi,ci)] += 1
        phases = defaultdict(list)
        for cycle in value['cycles']:
            for phase in cycle.get('phases', []):
                phases[phase['name']].append(phase['wall_s'])
        row = dict(status=value['status'], sync_phases=value['sync_phases'],
            cycles=value['cycles'], routes=routes, prefill=value.get('prefill', []),
            attention_candidate_shapes=[dict(window_slots=w, pooled_slots=c, calls=n,
                routing='mma' if w+c>=512 else 'fused') for (w,c),n in sorted(attention.items())],
            shapes=[dict(json.loads(key), calls=count) for key,count in shapes.items()],
            phase_totals={name:dict(calls=len(times), total_s=sum(times), median_s=median(times))
                          for name,times in phases.items()},
            microbench=value.get('microbench', []), dense_row_sweep=value.get('dense_row_sweep', []),
            sorted_route_probe=value.get('sorted_route_probe'))
        summary[filename] = row
    for name in ('dispatch-routed.json', 'dispatch-dense.json', 'dispatch-dense-r1.json', 'dispatch-head.json',
                 'dispatch-attention-fused.json', 'dispatch-attention-mma.json', 'head-prototype.json',
                 'attention-dispatch-496.json', 'attention-dispatch-544.json', 'attention-occupancy.json', 'headroom.json'):
        path = root/name
        if path.exists():
            summary[name] = json.loads(path.read_text())
            summary['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = root/'summary.json'
    path.write_text(json.dumps(summary, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
