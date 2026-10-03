# M32 isolated build commands

Initial audit: ARM64; Homebrew Rust 1.98.1, Apple clang 21, CMake 4.3.3,
pkgconf 3.0.7. `brew list --versions` had no OpenCV; `pkg-config --modversion
opencv4` failed. A find in /opt/homebrew, /usr/local, Xcode and CLT found no
OpenCVConfig.cmake/opencv4.pc. Real libclang existed in Xcode/CLT and
/opt/homebrew/Cellar/llvm@22/22.1.8/lib/libclang.dylib. No relevant environment
overrides existed except PATH. Final audit is build-audit.log.

```sh
HOMEBREW_NO_AUTO_UPDATE=1 brew install opencv   # 5.0.0_10; incompatible major
HOMEBREW_NO_AUTO_UPDATE=1 brew install opencv@4 # 4.14.0, keg-only, no link switch
# Complete package/dependency installation transcripts: opencv*-install.log.
git -C /tmp/ds41f-m31-recipe worktree add --detach /tmp/ds41f-m32-recipe \
  066d2ef2ed0deb574a0e6b5e6136316d6f296d55
git -C /Users/kioju/omlx-0.7.0.release worktree add --detach /tmp/ds41f-m32-omlx \
  4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40
uv venv --python /opt/homebrew/bin/python3.13 /tmp/ds41f-m32-qual
uv pip install --python /tmp/ds41f-m32-qual/bin/python maturin pytest jsonschema
# Apply candidate diagnostic patch/commit in the isolated worktree.
# Final build, all normal/default Python/image features intact:
env -u DOCS_RS \
  PKG_CONFIG_PATH=/opt/homebrew/opt/opencv@4/lib/pkgconfig \
  OpenCV_DIR=/opt/homebrew/opt/opencv@4/lib/cmake/opencv4 \
  CMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv@4 \
  LIBCLANG_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib \
  CARGO_TARGET_DIR=/tmp/ds41f-m32-cargo \
  /tmp/ds41f-m32-qual/bin/maturin build --release --locked --skip-auditwheel \
  --manifest-path /tmp/ds41f-m32-recipe/deepseek-recipe-python/Cargo.toml \
  -i /tmp/ds41f-m32-qual/bin/python -o /tmp/ds41f-m32-wheels
uv pip install --reinstall --python /tmp/ds41f-m32-qual/bin/python /tmp/ds41f-m32-wheels/*.whl
```

The first uv/maturin backend attempt compiled but wheel repair could not find
abseil @rpath; a second repair attempt with explicit DYLD_LIBRARY_PATH located
abseil but failed on libgcc. Those attempts are native-build.log and
native-build-repair.log. The successful final build/import uses no DYLD override.
Skip-auditwheel leaves a host-linked, nonportable wheel, not documentation stubs.

Pristine build repeats the successful command using /tmp/ds41f-m31-base,
/tmp/ds41f-m32-base-cargo, /tmp/ds41f-m32-base-qual and
/tmp/ds41f-m32-base-wheels. The old release recipe module is never substituted.

The model-probe environment initially installed generic MLX dependencies, then
was explicitly aligned to the authority's runtime pins before any model execution:

```sh
uv pip install --python /tmp/ds41f-m32-qual/bin/python \
  'mlx-vlm==0.7.1' 'mlx==0.32.2' 'numpy==2.3.5' 'transformers==5.17.0' \
  'mlx-lm @ git+https://github.com/ml-explore/mlx-lm@94cdcae13b266c337bcaca09b97b9c5a9c0e2cde'
uv pip install --python /tmp/ds41f-m32-qual/bin/python --no-deps /tmp/ds41f-m32-omlx
```

Additional installed diagnostic dependencies and exact final package versions are
in setup logs/runtime-identities.json. This is not a claim that every optional
oMLX serving extra is installed or qualified. Real scheduler and tool imports
succeed. No system-site-packages or release environment mutation.

```sh
/tmp/ds41f-m32-qual/bin/python tools/m32_native_parity_driver.py \
  artifacts/m31/parity-fixtures.json /tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json
# Repeat with /tmp/ds41f-m32-base-qual and pristine tokenizer for comparison.
/tmp/ds41f-m32-qual/bin/python tools/run_m32_native_preview.py \
  /tmp/ds41f-m32-recipe/static/tokenizers/v41/tokenizer.json
/tmp/ds41f-m32-qual/bin/python tools/run_m32_init_boundary_probe.py
/tmp/ds41f-m32-qual/bin/python tools/record_m32_recipe_evidence.py
(cd /tmp/ds41f-m32-recipe && CARGO_TARGET_DIR=/tmp/ds41f-m32-cargo \
  cargo test --offline --locked -p deepseek-recipe)
/tmp/ds41f-m32-qual/bin/python -m pytest -q /tmp/ds41f-m32-recipe/deepseek-recipe-python/tests
PYTHONPATH=. /tmp/ds41f-m32-qual/bin/python -m pytest -q \
  tests/test_m31_recipe_extension_evidence.py tests/test_m29_mtp_lifecycle.py \
  tests/test_m25_mtp_boundary.py tests/test_prefill_fp8_mlx_p5.py tests/test_prefill_fp8_mlx_p6.py \
  tests/test_m7_production_selector.py tests/test_m11_tool_boundary_contract.py \
  tests/test_m14_termination_contract.py tests/test_m20_generation_dependency.py tests/test_runtime_config.py
PYTHONPATH=. /tmp/ds41f-m32-qual/bin/python -m pytest -q tests/test_m32_recipe_native_evidence.py
```
