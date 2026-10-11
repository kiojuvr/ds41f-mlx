# M23 — Relocatable Release Bundle and Clean-Room Installation Qualification

## Distribution-boundary audit

M22 made release identity explicit but still assumed a source checkout for several operations:

| Component | M22 behavior | M23 classification |
| --- | --- | --- |
| `ds41f_mlx` runtime package | imported from repository root | required for normal runtime operation; copied into bundle |
| `release/ds41f-release.json` | resolved as `repo/release/...` | required package resource; copied once into bundle and remains logical authority |
| `ds41f_mlx.release_acceptance` | ran `cargo run` from workspace | source/development behavior only; installed mode uses bundled Rust acceptance binary |
| `ds41f_api` crate source | root Cargo workspace | required for Rust consumers; copied as local crate source in bundle |
| `m21_real_acceptance` | built by Cargo in source tree | required for installed acceptance; bundled as prebuilt local binary |
| `tests/`, `tools/`, `native/` CMake tests | repository-relative qualification | development/full qualification only; not required for normal installed operation |
| historical artifacts/docs | source evidence | historical evidence/source only; not required for installed runtime |
| checkpoint, oMLX, recipe, KV root | external configured assets | external; verified, not bundled |

Normal installed operation must not require Git, Cargo, CMake, historical artifacts, or the original checkout. Development qualification from source may still use those tools.

## Selected bundle architecture

M23 uses a local tarball/directory bundle rather than public registry publication. The bundle is small relative to the model and contains:

- `ds41f_mlx/` Python runtime and operator modules;
- `release/ds41f-release.json` and `release/bundle-record.json`;
- `pyproject.toml`, root `Cargo.toml`, `Cargo.lock`;
- `rust/ds41f_api/` crate source for local Rust consumers;
- `bin/m21_real_acceptance` prebuilt acceptance binary for the target Apple Silicon host;
- `bin/ds41f` and `bin/ds41f-accept` wrappers;
- `config/ds41f.env.example`;
- README and current operations/distribution docs.

The bundle excludes large and mutable external assets: official checkpoint, oMLX checkout/environment, deepseek-recipe checkout/package, and KV/Engram storage.

## Build command

From the source repository:

```bash
~/.venvs/omlx-0.7.0.release/bin/python -m ds41f_mlx.build_release --output-dir dist
```

The build verifies pyproject/Cargo version consistency with the release manifest, builds the Rust acceptance binary, copies the required runtime/boundary files, writes `release/bundle-record.json`, and creates `ds41f-mlx-0.23.0.tar.gz`.

The build does not mutate external checkpoint, oMLX, recipe, or KV assets.

## Install/configure commands

```bash
mkdir -p /opt/ds41f
cd /opt/ds41f
tar -xzf /path/to/ds41f-mlx-0.23.0.tar.gz
cd ds41f-mlx-0.23.0
cp config/ds41f.env.example config/ds41f.env
# edit config/ds41f.env for this machine
source config/ds41f.env
```

Required configuration remains the M22 `RuntimeConfig` environment model:

- `DS41F_PYTHON` — qualified Python, normally `~/.venvs/omlx-0.7.0.release/bin/python`;
- `DS41F_CHECKPOINT`;
- `DS41F_OMLX_PATH`;
- `DS41F_RECIPE_PATH`;
- `DS41F_KV_ROOT`;
- optional host/port/session settings.

## Installed operation

```bash
./bin/ds41f inspect
./bin/ds41f start
./bin/ds41f quick      # source-style quick if dev gates are present; normally use accept in bundle
./bin/ds41f-accept     # installed-release acceptance
```

`bin/ds41f` sets `PYTHONPATH` to the bundle root and delegates to `ds41f_mlx.ops`. `bin/ds41f-accept` additionally points `DS41F_ACCEPTANCE_BIN` at the bundled Rust binary and runs installed acceptance mode. Installed acceptance does not require Cargo or CMake.

## Manifest and bundle identity

`release/ds41f-release.json` remains the single logical release/dependency authority. `release/bundle-record.json` records:

- release manifest;
- source identity available at bundle build time;
- bundle file hashes and aggregate bundle digest;
- statement that large external assets are configured, not bundled.

A copied manifest is not independently edited; it is copied from the source release manifest by the bundle builder after consistency checks.

## Clean-room qualification

M23 clean-room qualification unpacked the tarball under `/tmp/ds41f-m23-clean/ds41f-mlx-0.23.0`, outside the development checkout, and ran:

```bash
DS41F_PYTHON=$HOME/.venvs/omlx-0.7.0.release/bin/python ./bin/ds41f inspect
DS41F_PYTHON=$HOME/.venvs/omlx-0.7.0.release/bin/python DS41F_OMLX_PATH=/tmp/missing-omlx ./bin/ds41f inspect
DS41F_PYTHON=$HOME/.venvs/omlx-0.7.0.release/bin/python ./bin/ds41f-accept --output /tmp/ds41f-m23-clean/installed-acceptance.json
```

Evidence:

- `artifacts/m23/build-result.json`;
- `artifacts/m23/clean-install-inspect.json`;
- `artifacts/m23/clean-install-bad-omlx.json`;
- `artifacts/m23/clean-install-acceptance.json`.

The bad oMLX path returned a failing provenance report and non-zero command status. Installed acceptance passed real server startup, `alive`/`ready`, models, stateless generation, SSE cancellation/recovery, stateful continuation, session close, error propagation, and graceful shutdown.

## Evidence inheritance

Inherited from M20: model/runtime correctness, P5 handoff, oMLX GenerationBatch decode, dependency baseline, persistence and long-context performance.

Inherited from M21: Rust HTTP/SSE boundary behavior and lifecycle semantics.

Inherited from M22: release manifest authority and operator command model.

Fresh in M23: bundle construction, manifest/package resource resolution outside Git, installed wrappers, bundled Rust acceptance binary use, bad dependency failure behavior, and clean-room installed real-runtime acceptance.

## Limitations

- The bundle is local Apple Silicon oriented; no public PyPI/crates.io publication is performed.
- The bundled Rust acceptance binary is built for the local target architecture; Rust consumers can also use the included crate source.
- Full development qualification still belongs in the source repository.
- External dependencies are verified but not created or upgraded automatically.
