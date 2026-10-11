"""Acceptance-only, pass-through settlement observer for the canonical launcher.

Run with the admitted installed-wheel interpreter, from /tmp, using -I.
Does not touch request admission, sampler, model, scheduling or reconciliation.
No new runtime API or release service. Output is actual execution trace data.
"""
import json
from pathlib import Path
import sys
from threading import Lock

from ds41f_mlx.observability import Projection

output = Path(sys.argv[1]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
original = Projection.metrics
lock = Lock()
seen = set()
fields = ('diagnostic_id', 'response_id', 'generation_policy', 'execution_sampling', 'cached_tokens',
          'generated', 'canonical_frontier', 'target_offsets', 'dspark_offsets',
          'prompt_replay', 'full_cache_repack', 'cancelled', 'cache_published',
          'quiescence', 'mtp_stats')


def observed(self, rid, trace=None, prompt_tokens=None):
    original(self, rid, trace, prompt_tokens)
    if not trace or 'cancelled' not in trace or rid in seen:
        return
    with lock:
        if rid in seen:
            return
        record = {name: trace[name] for name in fields if name in trace}
        record['request_id'] = rid
        choices = (trace.get('response') or {}).get('choices', [])
        if choices:
            record['finish_reason'] = choices[0].get('finish_reason')
            record['tool_calls'] = choices[0].get('message', {}).get('tool_calls', [])
        with output.open('a') as stream:
            stream.write(json.dumps(record, default=str) + '\n')
        seen.add(rid)


Projection.metrics = observed
from ds41f_mlx.serve import main
raise SystemExit(main(sys.argv[2:]))
