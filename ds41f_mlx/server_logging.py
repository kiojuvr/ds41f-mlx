"""Uvicorn's native colours with a local timestamp at the start of the body."""
import sys
from uvicorn.logging import DefaultFormatter, AccessFormatter
from .runtime_logging import timestamp


class LocalTime:
    def formatTime(self, record, datefmt=None):
        return timestamp(record.created)


class ServerFormatter(LocalTime, DefaultFormatter):
    def __init__(self, *args, use_colors=None, **kwargs):
        super().__init__(*args, use_colors=sys.stderr.isatty() if use_colors is None else use_colors, **kwargs)


class ServerAccessFormatter(LocalTime, AccessFormatter):
    def __init__(self, *args, use_colors=None, **kwargs):
        super().__init__(*args, use_colors=sys.stdout.isatty() if use_colors is None else use_colors, **kwargs)
