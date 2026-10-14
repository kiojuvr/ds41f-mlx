# Standard-OFF native publication — COMPLETE

Classification: **CONFIRMED_AUTHORITY_DRIFT — CLOSED BY BOUNDED PUBLICATION**.
The earlier “No further authority drift” conclusion was incorrect for OFF.
Portable is still an admitted official-precision alternative, not a correctness
defect or a permanent product decision to demote historical native capability.

## Recipe blocker: resolution A, not re-admission

Qualified OFF `deepseek_recipe._native` identity remains unchanged:

- SHA-256 `c4b17870812f0f03f1d026b102e21e870335a7717768f3847256de8a6f7ee5ca`
- size 14,011,072 bytes; arm64 UUID `92F54693-C8D1-39FE-8A88-2BEB5DF0BF22`
- retained M45 wheel SHA-256
  `c0bf2586a210e3d1ff8fc3f9cbfd37d025dc0d470b75cba963ac5b5f728eed91`

The exact wheel survived in `artifacts/tmp/m45-deps/wheels/`. M45 recipe-build
and installation logs identify that wheel, and M45/R1 and M47/R1 protocol/preview
receipts record the same native SHA. M47 commit `bdd9e9c` first added the resource
pin. Later OFF installed inspection records the same wheel digest. The current
OFF production binary already matched it. This is retained qualified provenance,
not harvesting an unqualified ambient binary.

All files of the retained M45 recipe source tree match the attributed repository
`recipe-source.tar.gz` members byte-for-byte, including Rust/C++ build inputs,
Python wrappers and Cargo.lock. Source authority is `sources.json`: upstream
`8cadfede7063c896b944e7bae05daa3549ae97ea`, attributed exported candidate
`29dabb5a55b7b2c6a68e18bbb3eb14495623e81a`, source archive SHA
`5b71ea6837ad3eb54a07da2b7ba1dd0a4ce658b585cdd2db24f4b83d9f879c22`.
Cargo.lock SHA is
`0897f68c7220e3fa11b11b03fe45245ab88bd2515586af057d00bf64111c9328`.

Historical build log records optimized Release, CPython 3.13 from development
`.venv`, PyO3 abi3-py3.10, pyproject features and deployment target 11.0.
Wheel metadata records maturin 1.15.0; embedded Rust paths identify compiler commit
`48a229ceaefd4985c50990b14116b6d856af0985` (Rust 1.98.1). Historical Python patch
version and exact compiler/linker invocation were not retained in that log;
no byte-reproducibility guarantee was found. These gaps do not authorize a new
build, nor invalidate the retained exact qualified wheel.

The rejected fresh binary SHA was
`1c2637dba52a89288266cc288c36bfc5920b42bb284c15dad51f5088fb820a83`.
Its source and Cargo.lock match, Rust commit and maturin match, and linked install
names match. A concrete differing build input is the absolute generated OpenCV
binding path embedded in the binary:

- historical: `/Volumes/SDXC-512/ds41f-mlx/target/release/build/opencv-3fc2c45feb0110bd/out/opencv/core.rs`
- fresh: `/private/tmp/ds41f-off-native-publication/cargo/release/build/opencv-3fc2c45feb0110bd/out/opencv/core.rs`

UUID also differs. This demonstrates path-sensitive output; it is not claimed to
explain every differing byte or exclude all other linker/build-metadata effects.
No attempt to reproduce or accept that fresh binary was needed (resolution A).

## One delivery authority, unchanged admission pins

`third_party/mtp/standard-off-recipe.json` binds the retained wheel now published
under `third_party/mtp/standard-off/`. `off_recipe.artifact_authority` verifies its
exact wheel/payload and source receipt and cross-checks the existing OFF native
resource pin. Packaging and provisioning consume this same delivery owner.
The resource pin is unchanged, not duplicated as an independent new acceptance.
MTP's separately qualified patched recipe binary remains unchanged.

OFF setup no longer recompiles the recipe. It provisions the frozen qualified
wheel and uses existing OFF dependency-version authority rather than the older
source-export numerical lock. Exact inventory checks still reject substitutes.
Recipe image/OpenCV transitive links retain existing installed semantic-native
hash/ABI admission and OFF resource ownership. No linked dependency was enrolled
from local observation. R1 source/reference is immutable.

Retained recipe validation: isolated installed import/load and ABI admission;
64 R1 protocol records exactly equal the official frozen fixture; native preview
source-agreement/nonmutation/determinism fixture passes. This is delivery
validation of the same bytes, **not new recipe artifact re-admission**.

## OFF native gates and actual publication

The single GLM artifact authority is unchanged: `glm-native.json` and archive SHA
`f8bad84b71567588677e8e9bdf30cee42f67dd5b0f32e958369bc601d6f269ec`.
No second native bundle owner, rebuild or numerical/dispatch implementation change.

Fresh native and portable environments were provisioned from repository sources
and frozen artifacts. Builder inventory and wheel membership passed. Native
installed admission requires exact members/bytes, origins, extension/ops/MLX
UUIDs, required symbols, array ABI and GPU/metallib loading. All 11 bounded
load/negative/restoration cases pass; native failure refuses rather than falls
back. Model-lifetime OFF admission consumes the selected installed GLM profile.

`accept_native_off.py` executed production backend/P5/P6 owners using the final
installed wheel, then repeated in normal production
`/Volumes/SDXC-512/ds41f-mlx_off-venv`:

- 256-token prompt + 8 committed decode tokens: frontier 264, P5 handoff count 1;
- 17-token idle suffix + 8 committed decode tokens: frontier 289;
- actual native P6 attention query shape `[1,64,16,512]` (16 appended rows,
  terminal row held out for P5); native MoE/grouped-expert calls observed;
- exact same live list, all 40 cache frontiers aligned, owned cache classes;
- history equals committed prompt/suffix/generated tokens, replay=0, repack=0;
- generation relinquishes cache/pending/wired handles at settlement;
- model retirement leaves zero active sessions and restores prior allocator-cache
  and wired limits. Resource policy itself is unchanged.

Canonical installed `serve.main --profile standard-off` also served two tiny
production HTTP session turns. Forwarding observers counted 80 successful native
multirow packed-attention and 720 grouped-expert calls (not admission smoke).
HTTP frontiers align and replay/repack remain zero. Session deletion and canonical
control shutdown completed with launcher result 0 and “runtime retired”.
No observer was added to production numerical implementation.

Default authority now explicitly selects `standard-off = native`;
`release/execution-package-standard-off.json` binds the final published package.
`release/execution-package-standard-off-portable.json` binds the explicit portable
alternative. Existing MTP native package/default is untouched. Checkout manifest
continues to describe the existing MTP package; each OFF wheel carries its own
exact OFF manifest. Startup failures never switch profiles.

Minimal capability guard: existing `execution-profiles.json` distinguishes
`production_default`, `explicit_alternative`, and pending-acceptance meaning;
default selection requires `production_default`. Temporary distribution omission
alone does not justify permanent capability deletion/demotion. This adds no
runtime selector, API or governance subsystem.

## Evidence, tests and release impact

`artifacts/off-native-promotion/` retains recipe provenance, earlier blocked
attempts, final provisioning, exact recipe fixtures, installed admission negatives,
final and production P5/P6 acceptance, canonical selected-path/HTTP proof and
shutdown. Release receipts bind artifact/package/manifest digests.

Focused tests: **130 passed, 26 subtests passed, 1 deselected**. An initial run's
one failure uses historical `DEFAULT_RECIPE` directly and points at an absent
external tokenizer; retained and not counted PASS. Installed recipe fixtures use
the exact provisioned tokenizer. Two HTTP harness assumptions were corrected
(OFF loads lazily and its legacy health does not expose GLM); canonical installed
identity/dispatch proof was used, not an API change. A P6 shape assertion was
corrected to account for its existing held-out terminal token. These were harness
errors, not runtime defects or gate bypasses.

No model precision/cache/state/topology changes, performance campaign or
200K/512K/768K/near-1M requalification. Existing native numerical/long-context/
Vision evidence is reused. OFF native distribution is release-ready within this
bounded supported-host scope. General-runtime release approval is unchanged.
**Remaining blockers: none for OFF-native publication.**
