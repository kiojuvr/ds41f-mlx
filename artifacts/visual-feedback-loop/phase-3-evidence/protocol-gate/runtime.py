"""Isolated Runtime bootstrap/trace retention only. Makes no generation requests."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
from dataclasses import replace

ROOT = Path('/Volumes/SDXC-512/ds41f-mlx')
OUT = Path(os.environ['PHASE3_EVIDENCE'])
WORK = Path(os.environ['PHASE3_RUNTIME_WORKSPACE'])
PORT = 18083

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, default=str)+'\n')

def main():
    with socket.socket() as s:
        s.bind(('127.0.0.1', PORT))
    # Supported admission order: identity inspection/binding before numerical imports.
    from ds41f_mlx.mtp_identity import config, inspect
    from ds41f_mlx.execution_identity import inspect_payload, numerical_profile
    cfg = replace(config(port=PORT, profile='mtp-serving-v1'), kv_root=WORK/'kv')
    identity = inspect(cfg)
    save('runtime-identity.json', identity)
    authority, _ = inspect_payload()
    assert authority['checkpoint_variant'] == 'flash-1017'
    assert authority['provenance']['source_commit'] == '2fdea4fe7367e540300d7c81d4fed402d716a274'
    import ds41f_mlx
    installed = Path(ds41f_mlx.__file__).parent
    sources = ['serving/server.py', 'serving/production_mtp.py', 'serving/deepseek_recipe_backend.py']
    hashes = {}
    for source in sources:
        actual = (installed/source).read_bytes()
        assert actual == (ROOT/'ds41f_mlx'/source).read_bytes(), source
        hashes[source] = hashlib.sha256(actual).hexdigest()
    save('runtime-admission.json', dict(pid=os.getpid(), source_commit=authority['provenance']['source_commit'], checkpoint_variant=authority['checkpoint_variant'], numerics=numerical_profile(), source_hashes=hashes, config=cfg.to_json(), admission_before_backend_import=True))
    cfg.apply_environment()
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.server import create_app
    import uvicorn
    backend = ProductionMTPBackend(runtime_config=cfg)
    backend.dependency_identity = identity['identity_sha256']
    backend.glm_profile = identity['glm_profile']
    app = create_app(profile='mtp-serving-v1', runtime_config=cfg, backend=backend)
    async def load_model():
        await backend._call(backend.load)
    app.router.add_event_handler('startup', load_model)
    from ds41f_mlx.serving.local_h11 import LocalH11Protocol
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=PORT, log_level='warning', workers=1, proxy_headers=False, ws='none', loop='asyncio', limit_concurrency=8, backlog=8, timeout_keep_alive=5, h11_max_incomplete_event_size=16384, http=LocalH11Protocol))
    def stop(signum, frame):
        server.should_exit = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def capture():
        done = 0
        while not server.should_exit:
            traces = list(backend.serving_traces)
            for i in range(done, len(traces)):
                data = json.dumps(traces[i], default=str).encode()
                with gzip.open(OUT/f'trace-{i:02}.raw.json.gz', 'wb') as f:
                    f.write(data)
                done = i + 1
            time.sleep(.25)
    watcher = threading.Thread(target=capture, daemon=True)
    watcher.start()
    try:
        server.run()
    finally:
        server.should_exit = True
        watcher.join(timeout=2)
        # The existing backend owns canonical settlement and shutdown.
        import asyncio
        asyncio.run(backend.shutdown_serving())
        backend.close()
        save('runtime-exit.json', dict(pid=os.getpid(), trace_count=len(backend.serving_traces), server_returned=True))

if __name__ == '__main__':
    main()
