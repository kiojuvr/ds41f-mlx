"""External evaluator trace retention; no production serving/cache changes."""
import gzip
import hashlib
import json
from pathlib import Path
import threading

class TraceArchive:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.count = 0
        self.lock = threading.Lock()

    def flush(self, traces):
        with self.lock:
            snapshot = list(traces)
            for i in range(self.count, len(snapshot)):
                trace = snapshot[i]
                if not trace.get('protocol_events') or trace['protocol_events'][-1].get('type') != 'message_stop':
                    raise RuntimeError(f'evidence-retention failure: trace {i} incomplete')
                data = json.dumps(trace, default=str).encode()
                path = self.directory / f'trace-{i:02}.raw.json.gz'
                temporary = path.with_suffix('.tmp')
                if path.exists():
                    raise RuntimeError('evidence-retention failure: archive already exists')
                with temporary.open('xb') as f:
                    f.write(gzip.compress(data, mtime=0))
                    f.flush()
                    import os
                    os.fsync(f.fileno())
                temporary.replace(path)
                assert gzip.decompress(path.read_bytes()) == data
                self.count += 1
            return len(snapshot)

    def final(self, traces):
        total = self.flush(traces)
        paths = sorted(self.directory.glob('trace-*.raw.json.gz'))
        rows = []
        for path in paths:
            raw = gzip.decompress(path.read_bytes())
            trace = json.loads(raw)
            if trace['protocol_events'][-1]['type'] != 'message_stop':
                raise RuntimeError('evidence-retention failure: incomplete archive')
            rows.append(dict(file=path.name, raw_sha256=hashlib.sha256(raw).hexdigest()))
        if len(paths) != total or self.count != total:
            raise RuntimeError(f'evidence-retention failure: archived={len(paths)}, memory={total}')
        return dict(status='PASS', final_in_memory=total, archived=len(paths), complete=len(rows), traces=rows)
