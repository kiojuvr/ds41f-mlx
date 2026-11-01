"""Acceptance-only worker pause and canonical response observation.

Never installed or used by Production. Admission runs normally, BEFORE importing
lazy Transformers modules. Only after admission do we attach the test observers.
"""
import json
from pathlib import Path
import time
from ds41f_mlx import mtp_identity
from ds41f_mlx.serve import main

inspect_original = mtp_identity.inspect


def inspect(cfg, **kwargs):
    report = inspect_original(cfg, **kwargs)
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend, ProductionScheduler
    start_original = ProductionMTPBackend._start
    retire_original = ProductionScheduler._retire
    paused = False

    def start(self, rec, prepared, *args, **kw):
        nonlocal paused
        if prepared.protocol == 'messages' and not paused:
            paused = True
            print('ACCEPTANCE controlled native worker pause 16.5s begins', flush=True)
            time.sleep(16.5)
            print('ACCEPTANCE controlled native worker pause ends; native prefill follows', flush=True)
        return start_original(self, rec, prepared, *args, **kw)

    def retire(self, request, **kw):
        retire_original(self, request, **kw)
        # Observe the existing canonical response, not a new semantic accumulator.
        trace = self.last_settlement
        with (Path(__file__).parent/'native-settled.jsonl').open('a') as output:
            output.write(json.dumps(dict(request_id=request.request_id, response=trace.get('response'))) + '\n')

    ProductionMTPBackend._start = start
    ProductionScheduler._retire = retire
    return report


mtp_identity.inspect = inspect
raise SystemExit(main())
