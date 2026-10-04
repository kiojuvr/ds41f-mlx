# Source delivery and attribution

This source tree contains the current ds41f runtime owners, Python-hosted native
and Metal sources, the native structural build/test foundation, and the OFF Rust
HTTP/SSE boundary. Native tests are source build/ownership gates, not a new backend
or claim that every native experimental component is production-selected.
Production remains DENSE_P0_P7/P5/P7 plus attributed oMLX GenerationBatch decode.

The default OFF environment installs the R1-owned base execution source export
(`reference/R1/off-source.tar.gz`). MTP installs the M41-delivered bounded patched
export (`third_party/mtp/omlx-source.tar.gz`). Both build the delivered recipe
source (`third_party/mtp/recipe-source.tar.gz`) with its own Cargo.lock and normal
OpenCV image features. They are separate environments, not dynamic backend toggles.
Archive identities/revisions/patches are in `third_party/mtp/sources.json`, R1 and
the promotion manifest. The official checkpoint remains external.

The oMLX licenses are inside the OFF archive and `third_party/mtp/omlx-LICENSE`;
recipe license is `third_party/mtp/recipe-LICENSE` and inside its export. The
Python prefill mathematics carries `ds41f_mlx/prefill_fp8_mlx/OMLX_MATH_LICENSE`
and attribution in `omlx_suffix_math.py`. Native nlohmann JSON carries its embedded
copyright/license in `native/third_party/nlohmann/json.hpp`. Dependency packages
retain their installed licenses. No upstream ownership replacement is claimed.

Setup seals actual installed Python payload/native link/source identity. The seal
is local operator-owned, not a signature or semantic qualification. Import origins
must be provisioned environment resources (plus this projected source repository),
never development/donor checkouts. When updating an installed runtime at an
explicit release checkpoint, changed executable payload requires rebuild and
affected acceptance under M43's release rules. This is not a requirement to
rebuild independent release environments or promote ds41f-runtime after every
ds41f-mlx development milestone. Development qualification covers affected tests,
applicable R1 conformance and necessary real-model/lifecycle/performance evidence;
qualified milestones may accumulate before release promotion. Runtime-only source
changes are rejected by projection checks; features and semantic fixes belong
in ds41f-mlx and flow one-way through explicit qualified releases.

For structural source qualification, keep build products outside the repository:

```sh
cargo test --locked
cmake -S native -B /absolute/native-build
cmake --build /absolute/native-build
ctest --test-dir /absolute/native-build --output-on-failure
```

The generic native structural foundation is not additional model-math
qualification. R1 plus affected authority/ownership qualification remains required.
Historical M22/M23 bundles depended on configured external checkouts and sometimes
prebuilt acceptance binaries. Their evidence is preserved in ds41f-mlx, but their
builder is not the canonical repository projection or its supported setup route.
