#!/usr/bin/env python3
"""Model-free queued-preparation and occurrence allocation, not model residency."""
import argparse
import base64
import gc
import hashlib
import json
import resource
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    a = p.parse_args()
    from ds41f_mlx.execution_identity import inspect_payload
    inspect_payload()
    from ds41f_mlx.mtp_identity import config
    from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
    from ds41f_mlx.serving.paired_checkpoint import image_ranges
    import ds41f_mlx.runtime.multimodal as mm
    cfg = config(profile='mtp-serving-v1')
    tokenizer = load_v41_tokenizer(cfg.recipe_path)
    data = a.image.read_bytes()
    history = []
    for n in range(12):
        history.extend([dict(role='user', content=[dict(type='text', text=f'Observation {n}'),
            dict(type='image', source=dict(type='base64', media_type='image/png', data=base64.b64encode(data).decode()))]),
            dict(role='assistant', content='Field Notes.')])
    history.append(dict(role='user', content='Repeat the title.'))
    raw = json.dumps(dict(model=cfg.model_id, max_tokens=64, thinking=dict(type='disabled'), messages=history)).encode()
    rows, waiting = [], []
    tracemalloc.start(1)
    def sample(label):
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        rows.append(dict(label=label, python_current=current, python_peak=peak,
                         maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    sample('baseline')
    for i in range(33):
        with patch.object(mm, 'MAX_REQUEST_PATCHES', 12*4784), patch.object(mm, 'MAX_REQUEST_PIXELS', 12*921600):
            waiting.append(prepare_request('messages', raw, tokenizer=tokenizer,
                recipe_path=cfg.recipe_path, checkpoint=cfg.checkpoint_path, ordinary=True))
        assert mm.MAX_REQUEST_PATCHES == 36864
        if i in (0, 15, 31, 32):
            sample(f'{i+1}-prepared')
    # Simulated queue occupancy: real preparation/retention, NOT scheduler admission
    # or cancellation. Keep one separate request as the active/preparing proxy.
    for prepared in waiting:
        prepared.multimodal.release_sources()
    del prepared
    sample('sources-released-objects-retained')
    waiting.clear()
    sample('objects-released')
    occurrences = []
    for n in (12, 24, 600):
        identities = [dict(start=1+i*578, length=578, sha256='a'*64, grid=[52, 92]) for i in range(n)]
        tracemalloc.reset_peak()
        before, _ = tracemalloc.get_traced_memory()
        t = time.perf_counter()
        ranges = image_ranges(identities)
        current, peak = tracemalloc.get_traced_memory()
        occurrences.append(dict(n=n, tuple_references=sum(len(k) for _, k in ranges),
            retained=current-before, peak=peak-before, seconds=time.perf_counter()-t))
        del ranges
    a.output.write_text(json.dumps(dict(kind='MODEL-FREE', image_sha256=hashlib.sha256(data).hexdigest(),
        request_bytes=len(raw), prompt_tokens=None, preparation_samples=rows, occurrences=occurrences,
        exclusions='No native queue, restore, encoder, capture or model loaded; input raw body preallocated before tracing.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
