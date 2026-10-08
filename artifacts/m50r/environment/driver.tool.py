"""Bounded M50R matched reload/co-residency assay, not an operational soak.

Original six resident trials remain untouched. Four independent loaded processes,
ABAB (ordinary/64GiB co-resident), each one excluded warm-up and three new sessions.
The inference measurement tool is unchanged; no profiler or GPU observer is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('artifacts/m50r/environment'))
    ap.add_argument('--pressure-gib',type=float,default=64)
    args=ap.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    baseline=json.loads(Path('artifacts/m50r/performance.json').read_text())
    identity=baseline['identity']['identity_sha256']
    expected=baseline['rows'][1]
    result=dict(schema='ds41f.m50r.environment.v1',status='RUNNING',cases=[],
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        conditions='process-fresh/model reload; filesystem/JIT/Engram cache uncontrolled and not called cold; touched co-resident allocation is bounded pressure, not severe stress')
    output=args.root/'summary.json'
    def save(): output.write_text(json.dumps(result,indent=2)+'\n')
    (args.root/'driver.tool.py').write_bytes(Path(__file__).read_bytes())
    (args.root/'pressure.tool.py').write_bytes(Path('tools/m50r_pressure.py').read_bytes())
    save()
    for name,pressed in [('reload-a',False),('pressure-a',True),('reload-b',False),('pressure-b',True)]:
        holder=None
        case=dict(name=name,co_resident_gib=args.pressure_gib if pressed else 0)
        try:
            if pressed:
                holder=subprocess.Popen([sys.executable,'-m','tools.m50r_pressure','--gib',str(args.pressure_gib)],
                                        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                ready=holder.stdout.readline()
                if not ready:
                    raise RuntimeError('pressure holder failed: '+holder.stderr.read())
                case['pressure_ready']=json.loads(ready)
                assert case['pressure_ready']['rss_bytes'] > args.pressure_gib*2**30*.95
            path=args.root/(name+'.json')
            command=[sys.executable,'-m','tools.freeze_m50r_candidate','--mode','performance','--repeats','3','--output',str(path)]
            case['command']=command;case['begin']=time.time()
            with (args.root/(name+'.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt=json.loads(path.read_text())
            assert receipt['identity']['identity_sha256']==identity
            assert receipt['status']=='CAPTURED_NOT_GATE_PASS'
            rows=receipt['rows'][1:]
            assert len(rows)==3
            for row in rows:
                assert row['request_sha256']==expected['request_sha256']
                assert row['trace']['canonical_generated']==expected['trace']['canonical_generated']
                assert row['trace']['canonical_frontier']==expected['trace']['canonical_frontier']
                assert row['trace']['mtp_stats']['depth_accepted']==expected['trace']['mtp_stats']['depth_accepted']
            if holder:
                import psutil
                assert holder.poll() is None, 'pressure fixture exited during inference'
                case['pressure_end_rss_bytes']=psutil.Process(holder.pid).memory_info().rss
            case.update(status='CAPTURED',load_s=receipt['load_s'],dispersion=receipt['dispersion'],
                        available_range_bytes=[min(r['after']['available_bytes'] for r in rows),max(r['after']['available_bytes'] for r in rows)],
                        swap_used_range=[min(r['after']['swap']['used'] for r in rows),max(r['after']['swap']['used'] for r in rows)],
                        backbone_s=[r['trace']['mtp_stats']['backbone_ms']/1000 for r in rows],
                        prefill_s=[r['trace']['prefill_handoff_s'] for r in rows])
        except BaseException as e:
            case.update(status='ERROR',error=repr(e));raise
        finally:
            if holder:
                try:
                    holder.communicate('release\n',timeout=15)
                except subprocess.TimeoutExpired:
                    holder.kill();holder.communicate()
            result['cases'].append(case);save()
    result['status']='CAPTURED';save()


if __name__=='__main__': main()
