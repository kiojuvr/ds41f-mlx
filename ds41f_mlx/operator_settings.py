"""Persist operator launch preferences, not live runtime state."""
import json
import os
from pathlib import Path

from .config import REASONING_EFFORTS


def settings_path():
    return Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'ds41f' / 'operator.json'


def load_reasoning():
    configured = os.environ.get('DS41F_DEFAULT_REASONING_EFFORT')
    if configured is not None:
        if configured not in REASONING_EFFORTS:
            raise ValueError('unsupported default reasoning effort')
        return configured
    try:
        value = json.loads(settings_path().read_text()).get('default_reasoning_effort', 'max')
    except (OSError, ValueError, AttributeError):
        value = 'max'
    return value if value in REASONING_EFFORTS else 'max'


def save_reasoning(value):
    if value not in REASONING_EFFORTS:
        raise ValueError('unsupported default reasoning effort')
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps({'default_reasoning_effort': value}) + '\n')
    temporary.replace(path)
