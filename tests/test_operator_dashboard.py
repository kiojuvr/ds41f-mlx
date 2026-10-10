"""Presentation rules, without terminal screenshot coupling or model loading."""
import json

import pytest

from ds41f_mlx.observability import Projection
from ds41f_mlx.operator_dashboard import Dashboard, Viewport, render_screen, command_strip, PHASE_LABELS


def fixture():
    p = Projection()
    p.lifecycle('READY', dict(host='127.0.0.1', port=8000, context_tokens=262144, profile='mtp-serving-v1'))
    p.begin('last')
    p.metrics('last', dict(cached_tokens=36782, generated=128, decode_s=2.91,
        suffix_append_s=3.67, total_ttft_s=.183, prompt_replay=0, full_cache_repack=0,
        canonical_frontier=38549, target_offsets=[38549], dspark_offsets=[38549],
        mtp_stats={'depth_drafted': [1000], 'depth_accepted': [869]}), 38421)
    p.finish('last', 'tool_calls')
    p.cache_summary({'retained_pages': 123, 'query_hits': 42})
    return p


def screen(p, width=110, height=32, **kwargs):
    s = p.snapshot() if p is not None else None
    rows = render_screen(s, Viewport(width, height), **kwargs)
    return rows, '\n'.join(r.text for r in rows)


def busy(p):
    p.begin('current'); p.phase('current', 'ENCODING')
    p.phase('current', 'CACHE_LOOKUP'); p.phase('current', 'DECODING')
    p.metrics('current', p.snapshot()['last_request']['metrics'], 38421)


def test_ready_is_quiet_and_client_wait_is_authoritative():
    p = fixture()
    _, text = screen(p)
    assert 'IDLE' in text and 'ready for request' in text
    assert 'WAITING FOR CLIENT' in text and 'SERVER IDLE' not in text
    p.begin('a'); p.finish('a', 'stop')
    assert 'WAITING FOR CLIENT' not in screen(p)[1]


def test_busy_elapsed_and_observed_only_rail():
    p = fixture(); busy(p)
    s = p.snapshot(); now = s['current_request']['phase_started_mono']+12.84
    rows, text = screen(p, now=now)
    assert rows[0].text.endswith('BUSY')
    phase_row = next(r for r in rows if r.text.startswith('DECODING'))
    assert phase_row.bold and phase_row.tone == '' and '12.84 s' in phase_row.text
    assert phase_row.accents == ((0, len('DECODING'), 'cyan'),)
    assert '[ DECODE ]' in text and 'ENCODE' in text
    assert 'RESTORE' not in Dashboard(s).rail().text  # no guessed completed stages
    assert 'active 1 · queued 0' in text


@pytest.mark.parametrize('state,tone', [('LOADING', 'yellow'), ('FAILED', 'red'), ('STOPPING', 'yellow')])
def test_lifecycle_states_no_invented_progress(state, tone):
    p = Projection(); p.lifecycle(state)
    rows, text = screen(p)
    assert rows[0].tone == '' and rows[0].accents[0][2] == tone
    assert state in text
    if state == 'LOADING':
        assert 'LOADING MODEL' in text and '—' in text
        assert 'waiting for runtime READY' in text
    assert '%' not in text and 'spinner' not in text.lower()
    assert screen(p, now=100)[1] == screen(p, now=200)[1]


def test_conflict_and_queue():
    rows, text = screen(None, conflict=True)
    assert rows[0].tone == '' and rows[0].accents[0][2] == 'red' and 'CONFLICT' in text
    p = fixture(); busy(p)
    p.begin('q'); p.phase('q', 'QUEUED', queue_reason='http_preparation')
    s = p.snapshot(); now = s['requests'][-1]['queue_entered_mono']+6.42
    _, text = screen(p, now=now)
    assert 'queued 1' in text and '1 waiting · oldest 6.42 s' in text
    assert 'http_preparation' not in text
    assert 'http_preparation' in screen(p, mode='diagnostics')[1]


@pytest.mark.parametrize('aligned,replay,repack,tone', [(True, 0, 0, ''), (False, 0, 0, 'red'), (True, 1, 0, 'red'), (True, 0, 2, 'red')])
def test_health_anomalies_remain_visible(aligned, replay, repack, tone):
    p = fixture(); s = p.snapshot()
    s['last_request']['metrics'].update(aligned=aligned, prompt_replay=replay, full_cache_repack=repack)
    rows = render_screen(s, Viewport(60, 12))
    health = next(r for r in rows if 'replay' in r.text)
    assert health.tone == ''
    assert bool(health.accents) == bool(tone)
    assert all(color == tone and length < len(health.text) for _, length, color in health.accents)
    assert ('NOT ALIGNED' in health.text) == (not aligned)
    assert f'replay {replay}' in health.text and f'repack {repack}' in health.text


@pytest.mark.parametrize('width,layout', [(110, 'wide'), (80, 'medium'), (60, 'compact')])
def test_breakpoints(width, layout):
    p = fixture(); busy(p)
    assert Viewport(width, 36).layout == layout
    rows, text = screen(p, width=width, height=36)
    assert len(rows) <= 35
    assert all(len(r.text) <= width-2 for r in rows)
    performance = next(i for i, r in enumerate(rows) if 'PERFORMANCE' in r.text)
    context = next(i for i, r in enumerate(rows) if 'CONTEXT' in r.text)
    assert (performance == context) == (layout == 'wide')
    assert 'retained_pages' not in text and 'recipe convert' not in text
    assert any(r.values for r in rows)  # separate bright numeric columns
    assert 'Q Quit' in command_strip('dashboard', width)


@pytest.mark.parametrize('width,height', [(110, 32), (80, 28), (60, 20), (40, 10)])
def test_healthy_dashboard_color_is_localized(width, height):
    p = fixture()
    for active in (False, True):
        if active:
            busy(p)
        rows, _ = screen(p, width=width, height=height, message='LLM already running')
        assert all(row.tone in ('', 'dim') for row in rows)
        colored_text = [row.text[start:start+length] for row in rows for start, length, _ in row.accents]
        assert all(text in ('READY', 'BUSY', 'DECODING', '[ DECODE ]') for text in colored_text)
        health = next(row for row in rows if 'replay' in row.text)
        assert not health.accents and health.values
        if active and width >= 70:
            rail = next(row for row in rows if '[ DECODE ]' in row.text)
            assert rail.tone == 'dim'
            assert rail.accents == ((rail.text.index('[ DECODE ]'), len('[ DECODE ]'), 'cyan'),)


def test_low_height_degrades_without_main_scrolling():
    p = fixture(); busy(p)
    for height in (24, 20, 16, 12, 10, 5, 2, 1):
        rows, text = screen(p, height=height)
        assert len(rows) <= max(0, height-1)
        if height >= 10:
            assert 'DECODING' in text and 'tok/s' in text
            assert 'replay' in text and ('hit' in text or 'Cache hit' in text)
        if height <= 12:
            assert 'REQUEST TIMINGS' not in text
    assert 'CACHE SUMMARY' not in screen(p, height=16)[1]
    assert '[ DECODE ]' not in screen(p, height=12)[1]


def test_composition_retains_metrics_and_all_observed_phases():
    p = fixture(); busy(p)
    for phase in ('RESTORING', 'SUFFIX_APPEND', 'CHECKPOINT_CAPTURE', 'P5_HANDOFF', 'DECODING', 'SETTLING'):
        p.phase('current', phase)
    s = p.snapshot()
    for width in (110, 80, 60, 40):
        rows = render_screen(s, Viewport(width, 70))
        text = '\n'.join(row.text for row in rows)
        assert all(len(row.text) <= width-2 for row in rows)
        assert 'CHECKPOINT' in text and '[ SETTLE ]' in text
        assert 'REQUEST TIMINGS · current' in text
        for value in ('38,421', '262,144', '36,782', '1,639', '95.7%', '86.9%'):
            assert value in text
        for phase in s['current_request']['completed_phase_durations']:
            label = dict((key.lower()+'_ms', label.lower()) for key, label in PHASE_LABELS)[phase]
            assert label in text
        for row in rows:
            for start, length in row.values:
                assert 'tok/s' not in row.text[start:start+length]


def test_medium_metric_column_does_not_stretch_with_terminal():
    p = fixture(); busy(p)
    endpoints = []
    for width in (70, 80, 99):
        rows, _ = screen(p, width=width, height=45)
        decode = next(row for row in rows if row.text.startswith('Decode '))
        endpoints.append(len(decode.text))
    assert endpoints == [56, 56, 56]


def test_received_and_queue_are_explicit_in_phase_rail():
    p = Projection(); p.lifecycle('READY'); p.begin('a')
    assert Dashboard(p.snapshot()).rail().text == '[ RECEIVED ]'
    p.phase('a', 'QUEUED', queue_reason='http_preparation')
    rail = Dashboard(p.snapshot()).rail()
    assert rail.text == 'RECEIVED ─ [ QUEUED ]'
    assert rail.accents[0][2] == 'yellow'


def test_details_and_unknowns_are_honest():
    p = fixture()
    rows, text = screen(p, height=15, mode='diagnostics')
    assert len(rows) > 15  # secondary view may scroll
    assert '/ Diagnostics' in text and 'RUNTIME' in text
    assert 'STATE INTEGRITY' in text and 'LAST REQUEST TIMINGS' in text
    assert 'recipe convert' in text and 'retained_pages' in text
    unknown = Dashboard(Projection().snapshot())
    assert all(value == '—' for _, value in unknown.performance)
    assert unknown.health().text.startswith('alignment —')
    assert 'replay —' in unknown.health().text
    assert 'Quit TUI; runtime remains running' in screen(p, mode='help')[1]


def test_content_never_enters_ui_projection():
    p = fixture(); p.begin('opaque')
    p.metrics('opaque', {'prompt': 'PROMPT_SENTINEL', 'response': 'RESPONSE_SENTINEL',
        'tool_calls': 'TOOL_SENTINEL', 'canonical_generated': [1234], 'decode_s': .5}, 5)
    s = p.snapshot()
    assert all(secret not in json.dumps(s) for secret in ('PROMPT_SENTINEL', 'RESPONSE_SENTINEL', 'TOOL_SENTINEL'))
    for mode in ('dashboard', 'diagnostics', 'help'):
        _, text = screen(p, mode=mode)
        assert all(secret not in text for secret in ('PROMPT_SENTINEL', 'RESPONSE_SENTINEL', 'TOOL_SENTINEL'))
        assert 'opaque' not in text and 'canonical_generated' not in text


def test_pty_resize_and_view_keys_do_not_control_runtime():
    # Runs real curses on a disposable PTY. No runtime is started or contacted.
    import os
    import pty
    import select
    import signal
    import struct
    import subprocess
    import sys
    import termios
    import fcntl
    import time
    master, slave = pty.openpty()
    source = '''import asyncio, curses
from ds41f_mlx.operator_tui import Operator
from ds41f_mlx.observability import Projection
operator = Operator()
p = Projection(); p.lifecycle('READY'); p.begin('a'); p.phase('a', 'DECODING')
operator.snapshot = p.snapshot()
async def poll(): pass
operator.poll = poll
curses.wrapper(lambda screen: asyncio.run(operator.run(screen, curses)))
print('PTY_EXIT_OK')
'''
    def resize(width, height):
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', height, width, 0, 0))
    resize(110, 32)
    def controlling_terminal():
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
    child = subprocess.Popen([sys.executable, '-c', source], stdin=slave, stdout=slave, stderr=slave,
                             preexec_fn=controlling_terminal,
                             env={**os.environ, 'TERM': 'xterm-256color'})
    output = bytearray()
    def drain():
        until = time.monotonic()+.35
        while time.monotonic() < until:
            if select.select([master], [], [], .05)[0]:
                output.extend(os.read(master, 65536))
    try:
        drain()
        for width, height, keys in ((80, 28, b'd'), (60, 18, b'\x1bOB'), (40, 10, b'd'), (110, 32, b'?')):
            resize(width, height); child.send_signal(signal.SIGWINCH)
            os.write(master, keys); drain()
        os.write(master, b'q'); drain()
        child.wait(timeout=5); drain()
        assert child.returncode == 0
        assert b'DECODING' in output and b'Diagnostics' in output and b'KEYS' in output
        assert b'PTY_EXIT_OK' in output and b'Traceback' not in output
    finally:
        if child.poll() is None:
            child.kill(); child.wait()
        os.close(master); os.close(slave)
