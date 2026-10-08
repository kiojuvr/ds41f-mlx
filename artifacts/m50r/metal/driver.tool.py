"""Attach stock Xcode Metal System Trace to the warmed M50R diagnostic process.

No launch-time DYLD/import injection, privilege/config change or benchmark rates.
If capture is unavailable, bridge observations may run but Metal coverage is BLOCK.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--marker',type=Path,default=Path('artifacts/m50r/movement-ready.json'))
    ap.add_argument('--root',type=Path,default=Path('artifacts/m50r/metal'))
    args=ap.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    result=dict(status='WAITING',version=subprocess.check_output(['xcrun','xctrace','version'],text=True))
    path=args.root/'capture.json'
    def save(): path.write_text(json.dumps(result,indent=2)+'\n')
    save();deadline=time.monotonic()+300
    while not args.marker.exists():
        if time.monotonic()>deadline: raise TimeoutError('warmed probe marker absent')
        time.sleep(.2)
    marker=json.loads(args.marker.read_text());result['probe']=marker
    trace=(args.root/'candidate.trace').resolve()
    command=['xcrun','xctrace','record','--template','Metal System Trace','--attach',str(marker['pid']),
             '--time-limit','30s','--output',str(trace),'--no-prompt']
    result['command']=command;save()
    log=args.root/'record.log'
    with log.open('w') as f:
        recorder=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+45
        ready=False
        while recorder.poll() is None and time.monotonic()<deadline:
            text=log.read_text()
            if 'Starting recording with the Metal System Trace' in text:
                # xctrace emits this before attaching instruments. Give attach
                # setup a bounded settling interval; exported GPU timestamps,
                # not this text alone, must prove workload coverage.
                time.sleep(2)
                ready=recorder.poll() is None;break
            time.sleep(.2)
        result['recording_ready']=ready
        result['go_wall_ns']=time.time_ns()
        # Preserve useful original-call bridge evidence even if permission/tool
        # failure prevents GPU observation. Never present that as Metal coverage.
        Path(str(args.marker)+'.go').write_text(json.dumps(dict(metal_recording_ready=ready,wall_ns=result['go_wall_ns']))+'\n')
        save()
        try: result['record_exit']=recorder.wait(timeout=90)
        except subprocess.TimeoutExpired:
            recorder.terminate();result['record_exit']=recorder.wait(timeout=15)
    if trace.exists():
        toc=args.root/'toc.xml'
        export=['xcrun','xctrace','export','--input',str(trace),'--toc','--output',str(toc)]
        with (args.root/'export.log').open('w') as f:
            result['export_exit']=subprocess.run(export,stdout=f,stderr=subprocess.STDOUT).returncode
    result['status']='CAPTURED' if ready and result.get('record_exit')==0 and result.get('export_exit')==0 else 'BLOCK'
    save()


if __name__=='__main__': main()
