"""Resources from the provisioned test interpreter, never donor installations."""
from dataclasses import replace
from pathlib import Path
import sys
import sysconfig

from ds41f_mlx.config import load_runtime_config

SITE = Path(sysconfig.get_paths()['purelib'])
RECIPE = Path(sys.prefix) / 'share/ds41f-mtp/recipe'


def runtime_config():
    return replace(load_runtime_config(), omlx_path=SITE, recipe_path=RECIPE)


def tokenizer():
    from ds41f_mlx.serving.server import load_v41_tokenizer
    return load_v41_tokenizer(RECIPE)
