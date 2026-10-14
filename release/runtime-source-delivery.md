# Source delivery and attribution

This source tree contains the current ds41f runtime owners, Python-hosted native
and Metal sources, the native structural build/test foundation, and the OFF Rust
HTTP/SSE boundary. Native tests are source build/ownership gates, not a new backend
or claim that every native experimental component is production-selected.
Production retains DENSE_P0_P7/P5/P7, first-party OFF target execution and the
separately admitted attributed oMLX MTP topology.

The default OFF environment installs the R1-owned base execution source export
(`reference/R1/off-source.tar.gz`). MTP installs the M41-delivered bounded patched
export (`third_party/mtp/omlx-source.tar.gz`). OFF provisions the exact retained
M45/M47 qualified recipe wheel bound by `third_party/mtp/standard-off-recipe.json`;
MTP provisions its separately qualified patched M52R wheel bound by
`third_party/mtp/normal-local.json`. Neither setup recompiles the recipe or admits
same-version substitute bytes. Delivered recipe source/Cargo.lock remain build
provenance. Both profiles default to the single frozen GLM native bundle;
portable remains an explicit separately admitted alternative. They are separate
environments, not dynamic backend toggles. See
`docs/standard-off-native-publication.md` for bounded OFF acceptance and receipts.
Archive identities/revisions/patches are in `third_party/mtp/sources.json`, R1 and
the promotion manifest. The official checkpoint remains external.

The oMLX licenses are inside the OFF archive and `third_party/mtp/omlx-LICENSE`;
recipe license is `third_party/mtp/recipe-LICENSE` and inside its export. The
Python prefill mathematics carries `ds41f_mlx/prefill_fp8_mlx/OMLX_MATH_LICENSE`
and attribution in `omlx_suffix_math.py`. Native nlohmann JSON carries its embedded
copyright/license in `native/third_party/nlohmann/json.hpp`. Dependency packages
retain their installed licenses. No upstream ownership replacement is claimed.

Setup constructs and installs a profile-specific execution wheel. Production
requires installed package origins; the projected/source repository is build and
promotion provenance only. There is no local seal or startup source-export/whole
host-closure equality. Exact semantic/executable payload and assets are checked
separately from host ABI compatibility; see `docs/runtime-admission.md`. When updating an installed runtime at an
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
