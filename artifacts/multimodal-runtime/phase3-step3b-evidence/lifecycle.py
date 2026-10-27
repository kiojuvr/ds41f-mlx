"""Passive ASGI lifespan observer; never closes or schedules the Runtime.

Install around the existing router lifespan before server.run(). The original
lifespan alone owns shutdown. This object keeps only weak backend/model refs.
Call finish() after releasing app/server/backend diagnostic references.
"""
from contextlib import asynccontextmanager
import gc
import json
from pathlib import Path
import sys
import weakref


class LifecycleObservation:
    def __init__(self, backend, path, memory):
        self.backend = weakref.ref(backend)
        self.model = None
        self.path = Path(path)
        self.memory = memory  # Must not capture a model/backend.
        self.receipt = {"observations": {}, "observation_errors": []}

    def observe(self, name, operation):
        try:
            self.receipt['observations'][name] = operation()
        except Exception as error:
            # Store strings, never exception/traceback objects retaining owners.
            self.receipt['observation_errors'].append({
                'name': name, 'error': repr(error)})

    def persist(self):
        try:
            self.path.write_text(json.dumps(self.receipt, indent=2) + '\n')
        except Exception as error:
            # Optional diagnostics must not replace an original serving error.
            self.receipt['observation_errors'].append({
                'name': 'receipt_persistence', 'error': repr(error)})
            print('lifecycle receipt persistence failed: ' + repr(error), file=sys.stderr)

    def before(self):
        backend = self.backend()
        model = getattr(backend, '_model', None)
        if model is not None:
            self.model = weakref.ref(model)
        self.observe('before_shutdown', self.snapshot)
        self.observe('before_shutdown_mlx', self.memory)
        self.persist()

    def snapshot(self):
        backend = self.backend()
        if backend is None:
            return {'backend_absent': True, 'scheduler_absent': True}
        scheduler = backend.scheduler
        row = dict(backend_absent=False, scheduler_absent=scheduler is None,
                   model_present=getattr(backend, '_model', None) is not None,
                   executor_shutdown=backend._executor._shutdown,
                   deliveries=list(backend._deliveries),
                   sessions=list(backend.sessions))
        if scheduler is not None:
            row.update(requests=list(scheduler.requests),
                       waiting=len(scheduler.waiting), running=list(scheduler.running),
                       uid_to_request_id=dict(scheduler.uid_to_request_id),
                       active_native_owners=[rid for rid, request in scheduler.requests.items()
                           if getattr(getattr(request, 'execution', None), 'owner', None) is not None],
                       source_occurrences={rid: len(getattr(
                           getattr(request.prepared, 'multimodal', None), '_sources', ()))
                           for rid, request in scheduler.requests.items()},
                       paired_roots=len(scheduler.checkpoints._payloads))
        return row

    def install(self, app):
        original = app.router.lifespan_context

        @asynccontextmanager
        async def observed_lifespan(subject):
            try:
                async with original(subject) as state:
                    try:
                        yield state
                    finally:
                        # Still inside original lifespan: Scheduler is live here.
                        self.observe('before_hook', self.before)
            finally:
                # Original shutdown already ran, even when it failed. Do not close again.
                self.observe('after_shutdown', self.snapshot)
                self.observe('after_shutdown_mlx', self.memory)
                self.persist()

        app.router.lifespan_context = observed_lifespan

    def finish(self, clear_cache):
        """After caller drops strong diagnostic refs; not a backend lifecycle owner."""
        self.observe('gc', gc.collect)
        self.observe('clear_allocator_cache', clear_cache)
        self.observe('gc_after_cache', gc.collect)
        self.observe('final', lambda: dict(
            backend_weakref_dead=self.backend() is None,
            model_weakref_established=self.model is not None,
            model_weakref_dead=None if self.model is None else self.model() is None))
        self.observe('final_mlx', self.memory)
        self.persist()
