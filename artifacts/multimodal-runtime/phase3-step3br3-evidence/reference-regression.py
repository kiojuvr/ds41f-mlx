"""Model-free actual FastAPI/Uvicorn lifetime regression, no Runtime imports."""
import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref
import uvicorn
from fastapi import FastAPI
from references import snapshot

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior_lifecycle', OUT.parent/'phase3-step3b-evidence/lifecycle.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Model: pass
class Backend:
    def __init__(self):
        self._model = Model()
        self.scheduler = None
        self._executor = SimpleNamespace(_shutdown=False)
        self._deliveries = {}
        self.sessions = {}
    async def shutdown(self): self._executor._shutdown = True

def application(backend):
    app = FastAPI()
    app.state.backend = backend
    @app.get('/health')
    async def health(): return {'model': backend._model is not None, 'application': app.title}
    app.router.add_event_handler('shutdown', backend.shutdown)
    return app

def main():
    backend = Backend()
    app = application(backend)
    observation = module.LifecycleObservation(backend, OUT/'reference-regression-lifecycle.json', lambda: {})
    observation.install(app)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='error'))
    async def stop(): server.should_exit = True
    app.router.add_event_handler('startup', stop)
    roots = dict(app=weakref.ref(app), server=weakref.ref(server))
    server.run()
    backend = None
    gc.collect()
    retained = snapshot(observation, **roots)
    assert not retained['backend']['dead'] and not retained['model']['dead']
    app = server = stop = None
    observation.finish(lambda: None)
    released = snapshot(observation, **roots)
    assert not released['backend']['dead'] and not released['model']['dead'], released
    assert released['task_roots']['app'] and not released['task_roots']['server']
    assert '_CallIdentity' in json.dumps(released)
    # Model-free isolation only: independently establish the cache ownership.
    # Never applied to the native Runtime or installed framework configuration.
    from fastapi.dependencies import models
    caches = ['_is_gen_callable_cached', '_is_async_gen_callable_cached', '_is_coroutine_callable_cached']
    cache_info = {n: str(getattr(models, n).cache_info()) for n in caches}
    for name in caches: getattr(models, name).cache_clear()
    observation.finish(lambda: None)
    isolated = snapshot(observation, **roots)
    assert isolated['backend']['dead'] and isolated['model']['dead'], isolated
    (OUT/'reference-regression.json').write_text(json.dumps(dict(retained_by_live_application=retained, after_dropping_task_roots=released, model_free_cache_isolation=isolated, classification_caches=cache_info, passed=True, weights_loaded=0), indent=2)+'\n')
    print('PASS: app/server released; FastAPI classification LRU retains endpoint closure; model-free cache isolation releases backend/model')
if __name__ == '__main__': main()
