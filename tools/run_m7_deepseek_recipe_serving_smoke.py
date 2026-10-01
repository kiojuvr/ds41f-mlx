#!/usr/bin/env python3
"""Compatibility entrypoint for the real M7 qualification harness.

The historical smoke recorder was pre-selector-promotion evidence and is no
longer authoritative.  This wrapper delegates to
``run_m7_deepseek_recipe_serving_qualification.py``.
"""
from __future__ import annotations
from tools.run_m7_deepseek_recipe_serving_qualification import main

if __name__ == '__main__':
    raise SystemExit(main())
