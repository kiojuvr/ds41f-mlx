# Attributed source exports for the bounded local MTP profile

The unified `merge` Runtime is **MTP-only**. Standard-OFF production delivery is
retired; its retained exports, locks and receipts below are archival, not setup
instructions. `--checkpoint-variant flash|flash-1017` fixes the exact checkpoint
inventory and numerical identity per package (default 1017). Both variants use the
same Recipe 0.2 wheel, overlays, GLM owner and Scheduler; no installation migration.
The base archives remain repository-owned attribution and reference material.
Ordinary MTP obtains the repository-delivered Recipe **0.2.0** repaired native
wheel, bound by the existing `normal-local.json`, not an operator's investigation
environment. Its clean `recipe-02-source.tar.gz` is upstream
`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`; provisioning applies the single
`docs/recipe-02-native-semantic-boundary.patch` forward-only. `sources.json` binds their exact SHA256, official upstream bases,
historical candidate evidence commits and delivered patch identities. No public
setup command clones/checks out those candidate commits or applies undocumented
patches. `ds41f_mlx.mtp_setup` verifies/extracts them in operator-owned build scratch,
installs oMLX and the pinned native wheel into a fresh venv. Historical OFF
artifacts and Cargo.lock remain attribution inputs; the unified provisioner and
package builder no longer offer OFF selection.

- oMLX: https://github.com/jundot/omlx (Apache-2.0; `omlx-LICENSE`, also inside
  archive). Full clean source export of the M33 candidate on the pinned 0.7.0 base;
  M33 generic semantic-horizon patch is included in exported source. ds41f owns the
  admission/lifecycle/recovery contracts. This remains a temporary implementation
  substrate, not the final ds41f execution-ownership destination.
- deepseek-recipe: https://github.com/deepseek-ai/deepseek-recipe (MIT, copyright
  DeepSeek; `recipe-LICENSE`, also inside archives). Ordinary MTP uses the clean
  official 0.2 export plus the consolidated native-session patch. The older
  M32/M33/M52R exports remain historical/OFF attribution, not a second MTP
  dependency. No guessed parser or tokenizer is substituted. Both exports carry
  their Cargo.lock and official V41 tokenizer.
- `requirements.lock`: exact ordinary Python package versions from the qualified
  candidate namespace (including a normal pinned upstream mlx-lm package build).
  It contains **no local filesystem URLs/development checkouts**. Build tools and
  unused transitive packages remain explicitly locked for reproducibility; this
  does not promote their optional audio/vision/network/oMLX-server capabilities.

`normal-local.json` is still the ordinary dependency/artifact owner. Its active
Recipe source/patch/wheel/build/version fields now describe the Python 3.13.15
release build with **auditwheel repair**, not the earlier `--skip-auditwheel`
artifact. Abseil and GCC were resolved from their existing Homebrew library
locations during repair; all 264 non-system native images are delivered in the
wheel. Package construction binds those exact wheel-native resources, and installed
admission checks their hashes, ABI, loaded UUIDs and complete loader-relative
closure. It does not enroll the local host graph or seal Python sources.
Historical numerical/checkpoint inventories, old host links and acceptance
records are unchanged; they are not new Recipe 0.2 execution evidence.
`normal-local.json` is **not** read at production startup.
Historically, for standard-OFF only, `standard-off-dependencies.json` overrode the MTP
NumPy/Transformers inventory with the exact cached payloads used by the existing
200K OFF qualification (NumPy 2.5.3 / Transformers 5.18.0). Its provenance records
the historical environment receipt digest. All numerical module/native hashes
are cross-checked against the unchanged `runtime/admitted_resources.json` before
packaging; full installed payload equality remains required. Neither MTP lock
is an OFF numerical authority. No runtime enrollment or automatic reseal is added.

`runtime-support-payload.json` preserves previously sealed candidate hashes for
additional packages actually imported by the model/ordinary protocol path
(Pillow, regex/Jinja, HTTP support and their helpers), not a new observation seal.
See `docs/runtime-admission.md` for classification and installed authority.
`requirements-normal-local.lock` reproduces the complete qualified namespace,
including the already-present PDF helper packages; their presence does not admit
Web/binary/Vision capabilities to MTP. The native wheel and patched export are
MIT-licensed under `recipe-LICENSE`; oMLX remains Apache-2.0. The unchanged base
archives, base lock and OFF environment are not promoted to the M52R native.

Historical `/tmp` trees and candidate commit labels remain evidence/attribution,
never runtime lookup authorities. Archives can be inspected with `tar -tzf` or
extracted normally. OFF uses its preserved release dependencies and does not import
these candidates implicitly. See `docs/mtp-local-release-candidate.md` and M41's
identity/native-build/clean-clone evidence. New source/build/link identities require
affected qualification; equal package version numbers are not equal execution.

## Explicit GLM native / portable distributions

`execution-profiles.json` records **native as the supported M3 Ultra local MTP default**, with portable a separately admitted explicit alternative. `glm-native.tar.gz` contains the exact
current-built GLM extension, ops dylib and metallib. `glm-native.json` binds their
sizes/SHA256 and the admitted source archive, 43 GLM/common inputs, toolchain and
repository-owned `tools/build_glm_bundle.py` invocation. These oMLX-derived
artifacts retain the attribution/license of their source export, distinct from
the MIT recipe wheel. Provisioning copies the frozen accepted bundle rather than
rebuilding or resealing local binaries. Recipe migration does not change the
GLM bundle authority or the preserved OFF base archives.

Setup and package construction consume the same explicit bundle/profile owner.
`--glm-profile portable` provisions no GLM native members. Installed admission
requires the selected profile's exact identity and native usability; load/ABI/
symbol/resource failure refuses startup, never silently switches to portable.
Current publication and bounded real-model selected-path evidence are recorded
in `release/execution-package*.json`, `artifacts/native-promotion/promotion.json`
and `artifacts/off-native-promotion/publication.json`. OFF recipe delivery reuses
`standard-off/deepseek_recipe-0.1.1-cp310-abi3-macosx_11_0_arm64.whl`; it does not
re-admit or replace the historical binary. See [OFF publication](../../docs/standard-off-native-publication.md)
and [installed admission](../../docs/runtime-admission.md#glm-execution-distributions).

## Post-completion compressed growth overlay (#4365)

`compressed-growth.patch` is the isolated MIT-licensed DeepSeek V4.1 delta from
upstream `09be4f420c7409298656bcfa13699dee3a0ed938`, adapted to retain an unchanged
index-key handle on ratio=2 tokens without a publication. `compressed-growth.json`
binds the patch and the four before/after source identities. MTP provisioning
applies it to the unchanged 0.7.0 source export before building; packaging inherits
the old dependency inventory with only these four explicit replacements and
records the overlay in the new installed manifest. Neither historical
`normal-local.json` nor base archives nor OFF admission is rewritten.

This is post-completion performance improvement, not runtime completion or an
upstream release upgrade. Slots 2/3 keep logical views over two capacity buffers;
singleton extraction shares the advisory registry, partial native rollback keeps
its chain, external replacement breaks it, and paired cloning intentionally does
not export it. MLX copy-on-write preserves held snapshots. The extra resident
packed storage is deliberate, not a zero-memory-cost optimization. See
[performance evidence](../../docs/performance.md#post-completion-compressed-growth-4365).
The growth overlay itself includes no #4354/#4360/#4363 changes.

## Post-completion bit-exact kernel overlay (#4360)

`bit-exact-kernels.patch` / `bit-exact-kernels.json` select only MXFP8 GEMV and
short-row lane Sinkhorn from MIT-licensed upstream
`f0d4cb53271456a474948edcdb38e0d2da347d79`. Apply AFTER #4365 to the same 0.7.0
source export, for MTP only. Three explicit source identities are inherited at
package build; the operator's installed payload and OFF sources are untouched.
The GEMV byte decoder repeats pinned MLX 0.32.2 `fp8.h`, including reserved
`0x7f/0xff` codes (+/-480); blindly copying upstream would change those values.
Sinkhorn retains the pinned serial fallback's `ROWS` template/guard.

FP8 packing was rejected: the current compiled graph encodes zero-group scale
as byte 2, upstream's fused pack as byte 1. Native sparse-attention changes do
not reach the qualified BF16 rounded path. Long-prefill radix top-k is not
adopted: it adds grouped score materialization and host evaluation/scheduling,
not an in-place kernel replacement within the retained P6 topology.
See [bounded evidence](../../docs/performance.md#post-completion-bit-exact-kernels-4360).
The #4360 overlay itself includes no #4354/#4363 work or broad upstream upgrade.

## Post-completion template identity overlay (#4354)

`template-reuse.patch` / `template-reuse.json` selectively adopt MIT-licensed
upstream `63917f5f6a9bd429621d8e615ba958269a2fcee6`, AFTER #4365 and #4360.
Four hash-bound replacements remove token-dependent template counts in current
index selection/merge, FP8 activation rounding, unpaired SwiGLU, Sinkhorn,
hyperconnection post and sorted expert combine. Arithmetic/order, existing
geometry/dispatch, output/state and pinned dependency/native versions stay intact.

#4360's lane Sinkhorn already reads runtime rows but retained an unused caller
`ROWS` template; this overlay removes it and adapts only the serial fallback's
guard. GEMV and the rejected packing/radix/attention decisions are untouched.
Unobserved paired-concat SwiGLU keeps its original generated source and `N`
specialization; the upstream patch is not applied wholesale. Provisioning and
package construction verify the full ordered overlay chain. OFF and historical
execution authorities remain unchanged; no #4363 work is included.

See [finite cold/shape-transition/warm evidence](../../docs/performance.md#post-completion-template-reuse-4354).
Source adoption is CLOSED, as a post-completion performance improvement. An
installed update is recommended through normal package construction/admission;
only isolated acceptance environments were built here, not the operator payload.
