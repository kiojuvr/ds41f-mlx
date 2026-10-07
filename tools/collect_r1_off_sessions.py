"""Read-only bounded sampler of R1 OFF's public session timing records.

Start alongside unchanged reference.R1.off_model. Does not enable diagnostics,
patch the runtime, or change requests. Only changed records are retained.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', type=float, default=900)
    a = p.parse_args()
    result = dict(schema='ds41f.r1.off-session-observer.v1', records=[], reads=0,
                  scope='public GET only, once per second; no stateless decode timing claim')
    seen = set()
    start = time.monotonic()
    while time.monotonic()-start < a.deadline:
        for sid in ('m41_off_0', 'm41_off_restored'):
            try:
                with urllib.request.urlopen(a.url+'/v1/sessions/'+sid, timeout=2) as response:
                    row = json.load(response)
                result['reads'] += 1
                serialized = json.dumps(row, sort_keys=True)
                if serialized not in seen:
                    seen.add(serialized)
                    result['records'].append(dict(session=sid, since_start_s=time.monotonic()-start,
                                                  record=row))
            except (urllib.error.URLError, TimeoutError):
                pass
        if a.qualification.exists() and json.loads(a.qualification.read_text())['status'] != 'RUNNING':
            break
        time.sleep(1)
    result['elapsed_s'] = time.monotonic()-start
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
