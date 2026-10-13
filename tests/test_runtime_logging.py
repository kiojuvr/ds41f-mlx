"""Bounded runtime history, privacy, reconnection and dashboard follow contract."""
import asyncio
import io
import json
import logging
import os
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest

from ds41f_mlx.runtime_logging import History, RuntimeHandler, display, MESSAGE_LIMIT, timestamp, server_log_config
from ds41f_mlx.observability import Projection
from ds41f_mlx.operator_control import Control
from ds41f_mlx.operator_dashboard import render_screen, runtime_log_lines, Viewport
from ds41f_mlx.operator_tui import Operator
from test_operator_observability import free_port


@pytest.fixture(autouse=True)
def local_timezone():
    previous = os.environ.get('TZ')
    os.environ['TZ'] = 'UTC'
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = previous
        time.tzset()


@pytest.mark.parametrize('zone,expected', [
    ('UTC', '2026-10-12 15:02:17.381 UTC'),
    ('Asia/Tokyo', '2026-10-13 00:02:17.381 JST'),
    ('America/New_York', '2026-10-12 11:02:17.381 EDT'),
])
def test_human_timestamps_and_chat_access(zone, expected):
    from datetime import datetime, timezone
    os.environ['TZ'] = zone
    time.tzset()
    created = datetime(2026, 10, 12, 15, 2, 17, 381000, tzinfo=timezone.utc).timestamp()
    assert timestamp(created) == expected
    from ds41f_mlx.server_logging import ServerAccessFormatter, ServerFormatter
    formatter = ServerAccessFormatter(fmt='%(levelprefix)s %(asctime)s  %(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False)
    access = logging.LogRecord('uvicorn.access', logging.INFO, '', 0,
        '%s - "%s %s HTTP/%s" %d', ('192.168.68.54:59527', 'GET', '/api/status', '1.1', 200), None)
    access.created = created
    assert formatter.format(access) == 'INFO:     ' + expected + '  192.168.68.54:59527 - "GET /api/status HTTP/1.1" 200 OK'
    config = server_log_config()
    assert config['formatters']['access']['()'] == 'ds41f_mlx.server_logging.ServerAccessFormatter'
    coloured = ServerAccessFormatter(fmt=config['formatters']['access']['fmt'], use_colors=True).format(access)
    assert '\x1b[32mINFO' in coloured and '\x1b[32m200 OK' in coloured
    assert expected in coloured
    record = logging.LogRecord('uvicorn.error', logging.INFO, '', 0, 'Started server process [%d]', (123,), None)
    record.created = created
    record.color_message = 'Started server process [\x1b[36m%d\x1b[0m]'
    lifecycle = ServerFormatter(fmt=config['formatters']['default']['fmt'], use_colors=True).format(record)
    assert '\x1b[32mINFO' in lifecycle and '\x1b[36m123' in lifecycle
    assert expected in lifecycle
    assert server_log_config()['loggers']['uvicorn.access']['propagate'] is False
    history = History()
    item = history.append(created, 'INFO', 'scheduler', 'request started')
    assert display(item) == 'INFO    scheduler: ' + expected + '  request started #1'
    assert item['timestamp'] == created
    rows = render_screen({'state': 'READY'}, Viewport(132, 57), logs=[item])
    assert any(expected in row.text and 'request started' in row.text for row in rows)


@pytest.mark.parametrize('component,alias', [
    ('uvicorn.error', 'server'), ('uvicorn.access', 'access'),
    ('ds41f.operator.runtime', 'runtime'), ('scheduler', 'scheduler'),
])
@pytest.mark.parametrize('severity,tone', [
    ('INFO', 'green'), ('WARNING', 'yellow'), ('ERROR', 'red'),
    ('CRITICAL', 'red'), ('DEBUG', 'cyan'),
])
def test_tui_log_metadata_and_semantic_colors(component, alias, severity, tone):
    item = History().append(0, severity, component, 'Application startup complete.')
    original = item.copy()
    rows = runtime_log_lines(item, 132)
    assert len(rows) == 1
    row = rows[0]
    prefix = f'{severity:<7} 1970-01-01 00:00:00.000 UTC  '
    assert row.text == prefix + alias + '  Application startup complete.'
    assert row.accents == ((0, len(severity), tone), (len(prefix), len(alias), 'dim'))
    assert item == original
    assert '\x1b' not in row.text and '#1' not in row.text
    assert component in display(item) and '#1' in display(item)  # headless unchanged
    for width in (1, 8, 20, 40, 60):
        wrapped = runtime_log_lines(item, width)
        assert all(len(line.text) <= width for line in wrapped)
        for line in wrapped:
            assert all(0 <= start < start+length <= len(line.text)
                       for start, length, _ in line.accents)
        assert ''.join(line.text[start:start+length] for line in wrapped
                       for start, length, color in line.accents if color == tone) == severity
        assert ''.join(line.text[start:start+length] for line in wrapped
                       for start, length, color in line.accents if color == 'dim') == alias


def test_runtime_history_and_safe_logging():
    history = History(3)
    output = io.StringIO()
    handler = RuntimeHandler(history, output)
    for i in range(5):
        record = logging.LogRecord('ds41f.operator.test', logging.INFO, '', 0, 'event %d', (i,), None)
        record.created = 1234+i
        handler.emit(record)
    batch = history.retrieve(0, 2)
    assert batch['gap'] and batch['oldest_sequence'] == 3
    assert [r['sequence'] for r in batch['records']] == [3, 4]
    assert batch['records'][0]['timestamp'] == 1236
    assert display(batch['records'][0]) == 'INFO    ds41f.operator.test: 1970-01-01 00:20:36.000 UTC  event 2 #3'
    batch['records'][0]['message'] = 'mutated'
    assert history.retrieve(2)['records'][0]['message'] == 'event 2'
    handler.emit(logging.LogRecord('uvicorn.access', logging.INFO, '', 0, 'GET /SECRET', (), None))
    handler.emit(logging.LogRecord('omlx.engine_core', logging.ERROR, '', 0, 'SECRET token error', (), None))
    assert 'SECRET' not in json.dumps(history.retrieve())
    handler.queue.join()
    assert 'SECRET token error' in output.getvalue()  # headless diagnostic retained
    history.append(10, 'INFO', 'test', '\x1b\n'+ 'x'*(MESSAGE_LIMIT*2))
    assert len(history.retrieve()['records'][-1]['message']) <= MESSAGE_LIMIT
    assert '\x1b' not in history.retrieve()['records'][-1]['message']
    class BrokenHistory:
        def append(self, *args):
            raise RuntimeError('logging failure')
    handler.history = BrokenHistory()
    handler.emit(logging.LogRecord('ds41f.operator.test', logging.INFO, '', 0, 'safe', (), None))


def test_log_api_separate_bounded_and_local_only():
    history = History(4)
    for i in range(8):
        history.append(100+i, 'INFO', 'test', str(i))
    projection = Projection()
    port = free_port()
    control = Control(projection, port, logs=history)
    base = f'http://127.0.0.1:{port}'
    try:
        with urlopen(base+'/ds41f/status') as response:
            assert 'records' not in json.load(response)
        with urlopen(base+'/ds41f/logs?after=0&limit=2') as response:
            batch = json.load(response)
        assert batch['gap'] and batch['next_sequence'] == 6
        with urlopen(base+'/ds41f/logs?after=6') as response:
            assert [r['sequence'] for r in json.load(response)['records']] == [7, 8]
        for query in ('limit=129', 'after=-1', 'after=x', 'after=1&after=2', 'unknown=1'):
            with pytest.raises(HTTPError) as error:
                urlopen(base+'/ds41f/logs?'+query)
            assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base+'/ds41f/logs', headers={'Origin': 'null'}))
        assert error.value.code == 403
    finally:
        control.close()


def test_reconnect_restart_and_paused_dashboard(monkeypatch):
    from ds41f_mlx import operator_tui
    history = History()
    for i in range(80):
        history.append(1700000000+i, 'WARNING' if i == 79 else 'INFO', 'test', f'event {i}')
    def call(port, path):
        after = int(path.split('=')[1]) if '=' in path else None
        return history.retrieve(after)
    monkeypatch.setattr(operator_tui, 'call', call)
    operator = Operator()
    asyncio.run(operator.poll_logs())
    assert operator.logs[-1]['sequence'] == 80
    operator.log_end = 60
    history.append(1700000080, 'ERROR', 'test', 'latest error')
    asyncio.run(operator.poll_logs())
    assert operator.log_end == 60 and operator.logs[-1]['sequence'] == 81
    snapshot = {'state': 'READY'}
    paused = render_screen(snapshot, Viewport(132, 57), logs=list(operator.logs), log_end=60)
    assert 'PAUSED' in '\n'.join(r.text for r in paused)
    assert any('event 59' in r.text for r in paused)
    assert not any('latest error' in r.text for r in paused)
    followed = render_screen(snapshot, Viewport(132, 57), logs=list(operator.logs))
    assert len([r for r in followed if 'event ' in r.text or 'latest error' in r.text]) >= 25
    assert not any(' #' in r.text for r in followed)
    assert (0, 5, 'red') in next(r for r in followed if 'latest error' in r.text).accents
    assert (0, 7, 'yellow') in next(r for r in followed if 'event 79' in r.text).accents
    for width, height in ((40, 10), (60, 20), (80, 28)):
        before = render_screen(snapshot, Viewport(width, height))
        after = render_screen(snapshot, Viewport(width, height), logs=list(operator.logs))
        core_end = next((i-1 for i,r in enumerate(before) if 'RUNTIME LOG' in r.text), len(before))
        assert before[:core_end] == after[:core_end]
        assert len(after) <= height-1
    reopened = Operator()
    asyncio.run(reopened.poll_logs())
    assert reopened.logs[-1]['sequence'] == 81
    history = History()
    history.append(1700000100, 'INFO', 'test', 'new runtime')
    asyncio.run(operator.poll_logs())
    assert len(operator.logs) == 1 and operator.log_end is None


def test_slow_output_cannot_block_log_producers():
    from threading import Event
    from time import monotonic
    entered, release = Event(), Event()
    class SlowStream:
        def write(self, text):
            entered.set()
            release.wait(5)
        def flush(self):
            pass
    history = History()
    handler = RuntimeHandler(history, SlowStream())
    record = logging.LogRecord('ds41f.operator.test', logging.INFO, '', 0, 'safe', (), None)
    try:
        handler.emit(record)
        assert entered.wait(1)
        start = monotonic()
        for _ in range(1000):
            handler.emit(record)
        assert monotonic()-start < 2
        assert handler.queue.qsize() == 512
        assert len(history.records) == 512
        assert history.retrieve(0)['gap']
        handler.drain(.01)
    finally:
        release.set()
        handler.queue.join()
