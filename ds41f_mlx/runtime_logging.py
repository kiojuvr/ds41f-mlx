"""Runtime-owned disposable logs; never an execution or persistence authority."""
from collections import deque
from datetime import datetime, timezone
import logging
from queue import Queue, Full
from threading import Lock, Thread
from uuid import uuid4

CAPACITY = 512
MESSAGE_LIMIT = 1024
BATCH_LIMIT = 128


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def display(record):
    return f'{timestamp(record["timestamp"])} #{record["sequence"]} {record["severity"]} {record["component"]}: {record["message"]}'


class History:
    def __init__(self, capacity=CAPACITY):
        self.runtime_id = uuid4().hex
        self.records = deque(maxlen=capacity)
        self.sequence = 0
        self.lock = Lock()

    def append(self, created, severity, component, message):
        # Only small scalar records under this lock; no I/O or model objects.
        with self.lock:
            self.sequence += 1
            record = dict(timestamp=created, sequence=self.sequence, severity=severity,
                          component=component[:80], message=' '.join(
                              ''.join(c if c.isprintable() else ' ' for c in message).split())[:MESSAGE_LIMIT])
            self.records.append(record)
            return record.copy()

    def retrieve(self, after=None, limit=BATCH_LIMIT):
        with self.lock:
            oldest = self.records[0]['sequence'] if self.records else self.sequence+1
            cursor = max(0, self.sequence-limit) if after is None else after
            records = [r.copy() for r in self.records if r['sequence'] > cursor][:limit]
            return dict(schema='ds41f.logs.v1', runtime_id=self.runtime_id,
                        oldest_sequence=oldest, latest_sequence=self.sequence,
                        gap=after is not None and after < oldest-1,
                        records=records, next_sequence=records[-1]['sequence'] if records else cursor)


# Uvicorn lifecycle templates are content-free. Access paths/queries and arbitrary
# third-party exception strings are NOT safe for the operator control surface.
UVICORN_LIFECYCLE = ('Started server process', 'Waiting for application startup',
    'Application startup complete', 'Shutting down', 'Waiting for connections to close',
    'Waiting for background tasks to complete', 'Waiting for application shutdown',
    'Application shutdown complete', 'Finished server process')


class RuntimeHandler(logging.Handler):
    def __init__(self, history, stream):
        super().__init__()
        self.history = history
        self.queue = Queue(maxsize=CAPACITY)
        self.stream = stream
        self.worker = Thread(target=self._write, daemon=True, name='ds41f-log-output')
        self.worker.start()

    def _write(self):
        while True:
            text = self.queue.get()
            try:
                self.stream.write(text+'\n')
                self.stream.flush()
            except Exception:
                pass
            finally:
                self.queue.task_done()

    def drain(self, timeout=.2):
        """Bounded process-exit grace only; never called by execution workers."""
        from time import monotonic, sleep
        deadline = monotonic()+timeout
        while self.queue.unfinished_tasks and monotonic() < deadline:
            sleep(.005)

    def emit(self, record):
        try:
            safe = record.name.startswith('ds41f.operator') or (
                record.name == 'uvicorn.error' and str(record.msg).startswith(UVICORN_LIFECYCLE))
            message = record.getMessage() if safe else None
            if message is None and record.levelno >= logging.WARNING:
                message = 'diagnostic warning/error (details in headless output)'
                if record.exc_info:
                    message += ' exception='+record.exc_info[0].__name__
            item = None
            if message is not None:
                item = self.history.append(record.created, record.levelname, record.name, message)
            # Preserve full existing headless diagnostics, but never wait on a
            # slow/broken stderr. Tracebacks stay headless, outside the log API.
            prefix = display(item) if safe and item else f'{timestamp(record.created)} {record.levelname} {record.name}: {record.getMessage()}'
            if record.exc_info:
                prefix += '\n'+logging.Formatter().formatException(record.exc_info)
            try:
                self.queue.put_nowait(prefix[:16384])
            except Full:
                pass
        except Exception:
            pass


def install(history, stream):
    handler = RuntimeHandler(history, stream)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    for name in ('uvicorn', 'uvicorn.error', 'uvicorn.access'):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    return handler
