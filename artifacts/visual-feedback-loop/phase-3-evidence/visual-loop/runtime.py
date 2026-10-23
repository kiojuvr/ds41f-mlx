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
    # Passive evaluation projection: call the original preparation exactly once,
    # record only identities/origins, return the identical object unchanged.
    import ds41f_mlx.serving.server as serving_server
    original_prepare = serving_server.prepare_request
    def observed_prepare(protocol, body, **kwargs):
        prepared = original_prepare(protocol, body, **kwargs)
        row = dict(body_sha256=hashlib.sha256(body).hexdigest(),
                   image_identities=[] if prepared.multimodal is None else list(prepared.multimodal.identities()),
                   image_is_historical=prepared.image_is_historical,
                   prompt_tokens=len(prepared.token_ids))
        with (OUT/'prepared-images.jsonl').open('a') as f:
            f.write(json.dumps(row, default=str)+'\n')
        return prepared
    serving_server.prepare_request = observed_prepare
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
    sys.path.insert(0, str(ROOT/'tools'))
    from phase3_trace_retention import TraceArchive
    archive = TraceArchive(OUT)
    capture_errors = []
    def capture():
        try:
            while not server.should_exit:
                archive.flush(backend.serving_traces)
                time.sleep(.25)
        except Exception as exc:
            capture_errors.append(repr(exc))
    watcher = threading.Thread(target=capture, daemon=True)
    watcher.start()
    try:
        server.run()
    finally:
        server.should_exit = True
        watcher.join(timeout=2)
        # server.run has returned only after active serving has settled.
        # Flush before shutdown_serving(), which itself calls backend.close().
        import asyncio
        try:
            assert not watcher.is_alive(), 'evidence-retention failure: watcher alive'
            assert not capture_errors, 'evidence-retention failure: '+str(capture_errors)
            save('trace-retention.json', archive.final(backend.serving_traces))
        except Exception as exc:
            save('trace-retention.json', dict(status='FAIL', error=repr(exc), archived=archive.count, final_in_memory=len(backend.serving_traces)))
            raise
        finally:
            asyncio.run(backend.shutdown_serving())
            backend.close()
            save('runtime-exit.json', dict(pid=os.getpid(), trace_count=len(backend.serving_traces), server_returned=True))

if __name__ == '__main__':
    main()
