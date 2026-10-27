"""Zero-weight checks of the retained diagnostic, not Production shutdown mocks."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.applications import Starlette


PATH = (Path(__file__).resolve().parents[1] / 'artifacts/multimodal-runtime/'
        'phase3-step3b-evidence/lifecycle.py')
spec = importlib.util.spec_from_file_location('compaction_diagnostic_lifecycle', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LifecycleObservation = module.LifecycleObservation


class Model:
    pass


class Backend:
    def __init__(self):
        self.scheduler = SimpleNamespace(
            requests={}, waiting=[], running={}, uid_to_request_id={},
            checkpoints=SimpleNamespace(_payloads={'old': object()}))
        self._model = Model()
        self._executor = SimpleNamespace(_shutdown=False)
        self._deliveries = {}
        self.sessions = {}
        self.shutdowns = 0

    async def shutdown(self):
        self.shutdowns += 1
        self.scheduler = None
        self._executor._shutdown = True


def setup(tmp_path, *, shutdown_error=None, memory=lambda: {'active': 0}):
    backend = Backend()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield {'original_state': True}
        finally:
            await backend.shutdown()
            if shutdown_error is not None:
                raise shutdown_error

    app = Starlette(lifespan=lifespan)
    observer = LifecycleObservation(backend, tmp_path / 'receipt.json', memory)
    observer.install(app)
    return backend, app, observer


async def run(app, error=None):
    async with app.router.lifespan_context(app) as state:
        assert state == {'original_state': True}
        if error is not None:
            raise error


def test_live_scheduler_then_normal_disposal_and_weakref_death(tmp_path):
    backend, app, observer = setup(tmp_path)
    asyncio.run(run(app))
    rows = observer.receipt['observations']
    assert rows['before_shutdown']['scheduler_absent'] is False
    assert rows['before_shutdown']['paired_roots'] == 1
    assert rows['before_shutdown']['requests'] == []
    assert rows['before_shutdown']['active_native_owners'] == []
    assert rows['after_shutdown']['scheduler_absent'] is True
    assert rows['after_shutdown']['executor_shutdown'] is True
    assert backend.shutdowns == 1
    # Re-observation is idempotent; it does not call close/shutdown again.
    assert observer.snapshot() == observer.snapshot()
    assert backend.shutdowns == 1
    del backend, app
    observer.finish(lambda: None)
    final = json.loads((tmp_path / 'receipt.json').read_text())
    assert final['observations']['final'] == dict(
        backend_weakref_dead=True, model_weakref_established=True, model_weakref_dead=True)
    assert final['observation_errors'] == []


@pytest.mark.parametrize('stage', ['body', 'shutdown'])
def test_original_exception_survives_optional_observation_failure(tmp_path, stage):
    original = RuntimeError('original ' + stage)

    def broken_memory():
        raise ValueError('optional observer failed')

    backend, app, observer = setup(
        tmp_path, shutdown_error=original if stage == 'shutdown' else None,
        memory=broken_memory)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run(app, original if stage == 'body' else None))
    assert caught.value is original
    assert backend.shutdowns == 1
    observer.finish(lambda: None)
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['observations']['after_shutdown']['scheduler_absent'] is True
    assert {e['name'] for e in receipt['observation_errors']} == {
        'before_shutdown_mlx', 'after_shutdown_mlx', 'final_mlx'}


def test_existing_event_shutdown_owner_runs_once(tmp_path):
    # Match create_app's existing on_event shutdown route, without importing
    # serving or loading weights. The observer wraps, never duplicates, it.
    backend = Backend()
    app = FastAPI()
    app.router.add_event_handler('shutdown', backend.shutdown)
    observer = LifecycleObservation(backend, tmp_path / 'receipt.json', lambda: {})
    observer.install(app)

    async def exercise():
        async with app.router.lifespan_context(app):
            assert backend.scheduler is not None

    asyncio.run(exercise())
    assert backend.shutdowns == 1
    assert not observer.receipt['observations']['before_shutdown']['scheduler_absent']
    assert observer.receipt['observations']['after_shutdown']['scheduler_absent']
    del backend, app
    observer.finish(lambda: None)
    assert observer.receipt['observations']['final']['model_weakref_dead']


def test_already_absent_scheduler_is_safe(tmp_path):
    backend, app, observer = setup(tmp_path)
    backend.scheduler = None
    asyncio.run(run(app))
    assert observer.receipt['observations']['before_shutdown']['scheduler_absent']
    assert observer.receipt['observations']['after_shutdown']['scheduler_absent']
    assert backend.shutdowns == 1


def test_failed_optional_final_cleanup_still_persists_receipt(tmp_path):
    backend, app, observer = setup(tmp_path)
    asyncio.run(run(app))
    del backend, app

    def broken_cache_observation():
        raise ValueError('optional allocator call failed')

    observer.finish(broken_cache_observation)
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['observations']['final']['model_weakref_dead']
    assert receipt['observation_errors'][0]['name'] == 'clear_allocator_cache'


def test_persistence_failure_never_masks_original_exception(tmp_path):
    backend, app, observer = setup(tmp_path)
    observer.path = tmp_path  # directory, intentionally unwritable as a receipt
    original = RuntimeError('serving failure')
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run(app, original))
    assert caught.value is original
    assert backend.shutdowns == 1
    assert all(e['name'] == 'receipt_persistence' for e in observer.receipt['observation_errors'])
