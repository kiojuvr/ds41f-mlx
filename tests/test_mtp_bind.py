"""Profile transport admission remains separate from executable identity."""
import pytest
from ds41f_mlx.mtp_identity import config


@pytest.mark.parametrize('host', ['0.0.0.0', '192.168.68.56', '::', 'fd12::56', 'mac-studio.local'])
def test_only_ordinary_mtp_accepts_nonloopback_bind(host):
    with pytest.raises(ValueError, match='literal 127.0.0.1'):
        config(host, 8000)
    cfg = config(host, 8000, profile='mtp-serving-v1')
    assert (cfg.host, cfg.port, cfg.max_live_sessions, cfg.trace_history_limit) == (host, 8000, 1, 32)


@pytest.mark.parametrize('profile', ['mtp-serving-v1', 'mtp-singleton-v1'])
def test_transport_change_keeps_worker_and_port_admission(monkeypatch, profile):
    with pytest.raises(ValueError, match='fixed port'):
        config('127.0.0.1', 0, profile=profile)
    monkeypatch.setenv('WEB_CONCURRENCY', '2')
    with pytest.raises(ValueError, match='one worker'):
        config('127.0.0.1', 8000, profile=profile)
