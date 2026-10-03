# M31 build/source qualification environment

Exact authorities and all package/source/binary identities are in
`runtime-identities.json`. This environment has no system-site-packages and never
borrows the old recipe binary to satisfy new extension tests.

Construction performed:

```sh
git -C /Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe \
  worktree add --detach /tmp/ds41f-m31-recipe 8cadfede7063c896b944e7bae05daa3549ae97ea
git -C /Volumes/SDXC-512/deepseek-v41-flash-mlx/third_party/deepseek-recipe \
  worktree add --detach /tmp/ds41f-m31-base 8cadfede7063c896b944e7bae05daa3549ae97ea
uv venv --python .venv/bin/python /tmp/ds41f-m31-qual
uv pip install --python /tmp/ds41f-m31-qual/bin/python --offline \
  maturin pytest jsonschema==4.26.0 tokenizers numpy regex pydantic
```

Candidate changes were made only in the first worktree and committed separately:
`066d2ef2ed0deb574a0e6b5e6136316d6f296d55`. The durable format-patch artifact applies
to the exact pin; `git apply --check` passed against the pristine second worktree.
A patch applied elsewhere can have a different commit identity; compare source
hashes and record that identity explicitly rather than silently reusing this run.
No ds41f dependency manifest or original recipe/oMLX source was changed.

Executed gates:

```sh
cd /tmp/ds41f-m31-recipe
CARGO_TARGET_DIR=/tmp/ds41f-m31-cargo cargo test --offline -p deepseek-recipe
# Compile-only documentation bindings, separate target, NEVER installed:
DOCS_RS=1 CARGO_TARGET_DIR=/tmp/ds41f-m31-cargo-check \
  PYO3_PYTHON=/tmp/ds41f-m31-qual/bin/python \
  cargo check --offline -p deepseek-recipe-python
cd /Volumes/SDXC-512/ds41f-mlx
PYTHONPATH=. /tmp/ds41f-m31-qual/bin/python tools/run_m31_recipe_source_parity.py
# Real native build on final candidate: fails for missing OpenCV dev files.
CARGO_TARGET_DIR=/tmp/ds41f-m31-cargo uv pip install \
  --python /tmp/ds41f-m31-qual/bin/python --offline --no-build-isolation \
  /tmp/ds41f-m31-recipe/deepseek-recipe-python
# Both suites attempted; no native module exists, so collection explicitly fails.
PYTHONPATH=. /tmp/ds41f-m31-qual/bin/python -m pytest -q \
  /tmp/ds41f-m31-recipe/deepseek-recipe-python/tests/test_bindings.py \
  /tmp/ds41f-m31-recipe/deepseek-recipe-python/tests/test_semantic_preview.py
PYTHONPATH=/Users/kioju/omlx-0.7.0.release /tmp/ds41f-m31-qual/bin/python \
  -c 'import omlx.api.tool_calling; print("real oMLX tool import OK")'
/tmp/ds41f-m31-qual/bin/python tools/run_m31_recipe_evidence.py
PYTHONPATH=. /tmp/ds41f-m31-qual/bin/python -m pytest -q \
  tests/test_m31_recipe_extension_evidence.py tests/test_m29_mtp_lifecycle.py \
  tests/test_m25_mtp_boundary.py tests/test_prefill_fp8_mlx_p5.py \
  tests/test_prefill_fp8_mlx_p6.py
```

JSON source parity normalizes only chat timestamp metadata; raw OutputChunks are
compared exactly. Source preview timing uses optimized Rust, 200 repeat samples
per diagnostic row, not a Python/MTP cycle timing. It is not a throughput gate.

Pinned oMLX genuinely requires jsonschema>=4.0.0. Installed 4.26.0 satisfies that
requirement here and the real tool module imports. This small diagnostic/build
environment is NOT a full pinned MLX model runtime environment. The next live
qualification environment needs the complete pinned runtime graph and the newly
built native recipe dependency, not a DOCS_RS artifact or an old binary fallback.

Build blocker: opencv 0.93.7 cannot find OpenCV development headers/configuration.
The preserved wheel contains runtime dylibs, not a verified development contract.
No unsupported stubs, removed image exports, or mismatched runtime ABI were used.
No newly built native module/package identity is claimed. Python/native parity,
MTP hook installation and live protocol qualification remain deferred. The
historical missing Engram runner remains unrelated.
