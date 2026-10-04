"""R1 has no optional semantic gates: a skipped test is not a pass."""
import pytest


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    if reporter and reporter.stats.get('skipped'):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
