#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
import uvicorn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend, DEFAULT_RECIPE, DEFAULT_MODEL_ID
from ds41f_mlx.serving.server import create_app
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX

ap=argparse.ArgumentParser()
ap.add_argument('--host', default='127.0.0.1')
ap.add_argument('--port', type=int, default=8000)
ap.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
ap.add_argument('--omlx-path', default=str(DEFAULT_OMLX))
ap.add_argument('--recipe-path', default=str(DEFAULT_RECIPE))
ap.add_argument('--model-id', default=DEFAULT_MODEL_ID)
args=ap.parse_args()
backend=DeepSeekRecipeRuntimeBackend(checkpoint=Path(args.checkpoint), omlx_path=Path(args.omlx_path), recipe_path=Path(args.recipe_path), model_id=args.model_id)
app=create_app(backend=backend, recipe_path=Path(args.recipe_path), model_id=args.model_id)
uvicorn.run(app, host=args.host, port=args.port)
