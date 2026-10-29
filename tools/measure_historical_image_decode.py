#!/usr/bin/env python3
"""Model-free companion: measure decoder residency without MLX preflight peaks.

Run with the isolated candidate's python -I. Fixtures enter a fresh worker via
stdin, never disk. Does not load models or inspect/alter Production.
"""
import base64
import ctypes
import gc
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import weakref


def worker():
    from PIL import Image
    from ds41f_mlx.runtime import multimodal as mm
    assert Path(mm.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    Image.init()
    images = [base64.b64decode(x, validate=True) for x in json.load(sys.stdin)]
    assert len(images) == 20
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    def memory():
        buf = ctypes.create_string_buffer(1024)
        assert lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0
        return dict(footprint=ctypes.c_uint64.from_buffer(buf, 72).value,
                    lifetime_max_footprint=ctypes.c_uint64.from_buffer(buf, 240).value)
    refs, max_live = [], 0
    original = Image.open
    def opened(*args, **kwargs):
        nonlocal max_live
        image = original(*args, **kwargs)
        refs.append(weakref.ref(image))
        max_live = max(max_live, sum(r() is not None for r in refs))
        return image
    Image.open = opened
    gc.collect()
    before, samples, stop = memory(), [], threading.Event()
    def watch():
        while not stop.wait(.001):
            samples.append(memory()['footprint'])
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    wall, cpu, pixels = time.perf_counter(), time.process_time(), 0
    try:
        for data in images:
            assert not any(r() is not None for r in refs)
            metadata = mm.validate_encoded_image(data)
            pixels += metadata['width']*metadata['height']
            samples.append(memory()['footprint'])
        duration = dict(wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)
    finally:
        stop.set()
        thread.join()
    assert len(refs) == 40 and not any(r() is not None for r in refs)
    print(json.dumps(dict(**duration, before=before, after=memory(),
        sampled_max_footprint=max(samples), max_live_pil_handles=max_live,
        live_pil_after=0, validation_calls=20, image_open_calls=len(refs),
        total_pixels=pixels, encoded_bytes=sum(map(len, images)),
        runtime_module=str(Path(mm.__file__).resolve()),
        runtime_sha256=sha256(Path(mm.__file__).read_bytes()).hexdigest())))


def main():
    spec = importlib.util.spec_from_file_location('workload',
        Path(__file__).with_name('qualify_historical_image_admission.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = module.fixture_images()
    result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--worker'],
        input=json.dumps([base64.b64encode(x).decode() for x in images]),
        capture_output=True, text=True, check=True, timeout=60)
    print(json.dumps(json.loads(result.stdout), indent=2))


if __name__ == '__main__':
    worker() if sys.argv[1:] == ['--worker'] else main()
