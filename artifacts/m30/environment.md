# M30 isolated diagnostic environment

No ds41f dependency or external source file was changed. No site-packages were
vendored into ds41f. Commands executed from the repository root:

```sh
uv venv --python .venv/bin/python /tmp/ds41f-m30-qual
# First attempt: source build + exact pytest 9.0.2, offline: pytest not cached.
# Second attempt: source build + cached pytest: OpenCV metadata missing.
# Preserve the existing release binary for corroborative Python tests instead:
cp -R /Users/kioju/.venvs/omlx-0.7.0.release/lib/python3.13/site-packages/deepseek_recipe \
  /tmp/ds41f-m30-qual/lib/python3.13/site-packages/
cp -R /Users/kioju/.venvs/omlx-0.7.0.release/lib/python3.13/site-packages/deepseek_recipe-0.1.1.dist-info \
  /tmp/ds41f-m30-qual/lib/python3.13/site-packages/
uv pip install --python /tmp/ds41f-m30-qual/bin/python --offline \
  jsonschema==4.26.0 pytest tokenizers numpy regex pydantic
PYTHONPATH=/Users/kioju/omlx-0.7.0.release /tmp/ds41f-m30-qual/bin/python \
  -c 'import omlx.api.tool_calling; print("real oMLX tool import OK")'
/tmp/ds41f-m30-qual/bin/python tools/run_m30_recipe_preview_audit.py
PYTHONPATH=. /tmp/ds41f-m30-qual/bin/python -m pytest -q \
  tests/test_m30_recipe_preview_audit.py tests/test_m11_tool_boundary_contract.py \
  tests/test_m29_mtp_lifecycle.py tests/test_m25_mtp_boundary.py \
  tests/test_prefill_fp8_mlx_p5.py
```

This is an isolated parser/tool-import diagnostic environment, NOT a complete
real-model qualification environment. It has no MLX runtime installed. That is
not hidden by skipping the tool import: the real jsonschema-dependent oMLX tool
module imports successfully. Model execution remains blocked by the protocol API.
`runtime-identities.json` records the entire package inventory, native binary
hash and clean authoritative source hashes. For a future live gate, install the
pinned full release graph rather than treating this small diagnostic env as live
qualification. Cached versions used here are diagnostic identities, not a new
release lock.

The authoritative-source probe compiles `state_machine.rs` directly with rustc
into a temporary executable. It does not need OpenCV, change the dependency, or
copy its grammar. Python binary build-revision provenance is not independently
attested in this milestone; its origin is the previously qualified release venv.
Native Cargo version 0.1.0 and distribution version 0.1.1 are both intentional in
the pinned source.

The recipe source build failure was OpenCV crate 0.93.7's inability to find
`opencv4.pc`/`OpenCVConfig.cmake`. It is not classified as the protocol blocker.
`jsonschema>=4.0.0` is declared in the pinned oMLX `pyproject.toml`; 4.26.0 satisfies
it here without changing ds41f's pyproject. Historical Engram runner is unrelated.
