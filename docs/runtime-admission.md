# Installed execution admission

## Unified MTP-only admission

Production supports only `mtp-serving-v1` and the explicit singleton MTP profile.
Standard-OFF startup/admission/provisioning/package selection is retired. OFF
migration descriptions and receipts below are historical, not current instructions.
Installations are not migrated automatically. The explicitly authorized
[Multimodal-Runtime promotion](../artifacts/multimodal-runtime/production-completion.md)
updated only the existing Flash-1017 Runtime wheel, preserving the admitted
dependency/native/checkpoint identities and a verified rollback snapshot.

`--checkpoint-variant flash|flash-1017` fixes one variant per newly built package
(default: `flash-1017`). Its manifest binds the exact approved 48-shard inventory,
official revision, metadata, numerical profile and execution dependencies. Original
Flash uses `flash-original-woa-v1` (unchanged inverse-RoPE output); 1017 retains
`flash-1017-pre-woa-fp8-v1`. Tuple validation rejects unknown/mixed inventories or
profiles. The selection is bound once by installed admission, cannot change within
the process and has no hot-path filesystem access. Both checkpoint inventory and
numerical profile participate in paired-cache execution identity.


DeepInfer ordinary MTP permanently discontinues source/archive/patch and generated
Python-package SHA sealing. Version/source dependency locks, strict ordered patch
application, installed non-editable module/support-file origins and exact native,
tokenizer and selected official checkpoint integrity remain mandatory. No installed file
may be manually edited; no bypass or future source-seal finalization is required.
The source/payload equality descriptions below retain the original admission
migration / preserved OFF boundary, not ordinary DeepInfer Python-source checks.

Recipe 0.2.0 comes from the retained auditwheel-repaired Chat/Messages wheel and
one consolidated semantic patch. Its 264-image non-system native closure is
wheel-owned with zero Homebrew links; exact bytes, origins, resolved loaded images
and Mach-O ABI still admit execution. The older Homebrew-linked Recipe discussion
below describes the original/preserved distribution, not this new wheel.
Build/source commit and wheel SHA are provenance, not equality with today's checkout.
[Operations](operations.md#deepinfer-daily-operation-and-harness) owns the separate
persistent installation; [development evidence](deepinfer-development.md) records
installed identity and acceptance without broad release qualification.

## Trust boundary

The release publisher and local operator are trusted. Deployment packages and
admitted assets must remain static for the process lifetime. This is not signing,
hostile-host attestation, or protection from an administrator replacing both code
and its admission manifest. Production imports an installed, non-editable
`ds41f_mlx` package inside the selected environment; `ops start` launches Python
in isolated mode. Direct checkout execution, external import overrides, and
`--no-validate` fail closed. No source tree or local observation seal is required.

| Responsibility | Authority / treatment |
| --- | --- |
| Execution identity | `execution_manifest.json`: DeepInfer MTP dependency versions and installed module/support-file origins plus exact native/asset payloads; preserved OFF also retains full Python payload/bytecode checks |
| Semantic assets | Exact checkpoint metadata, 48 shard byte hashes and recipe tokenizer; existing file-identity cache avoids rescanning unchanged weight files, never trusts LFS metadata alone |
| Semantic native | Recipe 0.2 wheel-owned native graph is exact; original/preserved Recipe uses the qualified Homebrew semantic subgraph. Native hashes come from artifact authorities, not today's observation |
| Host compatibility | Apple Silicon, supported macOS 26 ABI family, Python 3.13 >= 3.13.15, M3 Ultra/512 GB MTP capacity; successfully loaded native recipe with required semantic methods; resolved native graph and consumer/provider Mach-O ABI contracts. No OS patch equality, unrelated distribution inventory or host-closure SHA equality |
| Build provenance | Source commit plus package-input digest (including uncommitted input), attributed exports/patches, native wheel/toolchain provenance and lockfiles. Retained for audit/reproduction; never compared to today's checkout, compiler or SDK at startup |
| Qualification evidence | R1, historical `normal-local.json` runtime/closure/environment hashes and support-package candidate evidence. Build qualification inputs only; absent from runtime admission paths |

MLX is platform support **and** model execution. Its executed package, native
library and kernels remain exact; calling it a host dependency does not excuse
numerical drift. Tokenizers, Transformers, Pillow and protocol Python dependencies
remain version-locked with installed origins in ordinary MTP; this is not a
Python-source hash seal. Native numerical payloads, tokenizer assets and Recipe
image-codec libraries remain byte-exact. Preserved OFF retains its stronger Python
payload/bytecode equality checks. Unrelated packages, compiler tools and test runners are not inventoried
as executable authority. Stdlib/system frameworks use the supported host ABI
contract rather than an inventory of the macOS shared cache.

## Native call/link audit

`deepseek-recipe-image/src/opencv.rs` in the delivered recipe export calls Mat
operations, imdecode, color/alpha conversion, INTER_CUBIC resize, border creation
and WebP imencode. This is semantic image preparation, not an incidental library.
The Rust binding's broad link feature also loads unrelated OpenCV modules.
`libopencv_freetype -> HarfBuzz -> GLib -> pcre2` supports text shaping/rendering
and GLib regular expressions on that unused drawing path. The recipe does not
call FreeType/HarfBuzz drawing, GLib regex, OCR, video capture/FFmpeg, VTK rendering
or OpenCV DNN inference; model execution is MLX/first-party code. These unused
link paths remain host loadability/ABI obligations, **not** semantic byte pins.
Classification follows the actual recipe calls and their dependency subgraph,
not library names, link distance, or everything observed by dyld.

Host scanning resolves the provisioned Homebrew install names, validates each
provider's ABI version against its consumer, requires successful native loading,
and checks that the resolved graph is actually loaded without redirected
non-system images. Semantic images additionally require exact bytes and a matching
loaded Mach-O UUID (including universal arm64 slices). New reachable native
features require reclassification and qualification before package publication.
No pcre2-specific exception, approved replacement hash, or Homebrew rollback is
involved. Compatibility failure still refuses startup.

## GLM execution distributions

Both unified **MTP checkpoint variants default to GLM native** on the supported M3 Ultra deployment.
`third_party/mtp/execution-profiles.json` records the explicit decisions and status;
portable is an admitted explicit alternative.
MTP serving mode, checkpoint variant and GLM execution profile are distinct.
Provision with `--glm-profile portable` for the alternative; there is no request
selector, hot switch or implicit fallback. Inspect, startup logs, operator status
endpoint and MTP health expose `glm_profile` without performance inference.

`tools/build_glm_bundle.py` owns the GLM-only Release build from the SHA-pinned
current oMLX export (43 unchanged GLM/common inputs). It removes transient absolute
rpaths and freezes `_ext.cpython-313-darwin.so`, `libomlx_glm_kernel_ops.dylib` and
`omlx_glm_kernels.metallib` in `third_party/mtp/glm-native.tar.gz`. The adjacent
`glm-native.json` binds exact sizes/hashes and source/toolchain/build provenance.
Independent builds need not reproduce binary hashes: provisioning installs the
frozen accepted bytes, never observes or reseals a local rebuild. Portable ships
none of these members. Package inventories consume the same bundle authority.

Installed admission extends the existing `resource_admission.check_glm_bundle`:
selected profile, exact bundle membership/bytes, imported extension origin,
loaded extension/dylib and MLX core/library UUIDs, unique installed MLX and
sibling GLM dylibs, all 23 declared extension
symbols, array-caster ABI probe and bounded packed-attention GPU/metallib load.
Native load/ABI/symbol/resource failure **refuses startup**, even if portable
operations would be possible. Historical OFF resource pins are retained; installed
OFF can consume the same publisher bundle owner, not an independent native owner.
The subsequent [bounded OFF publication](standard-off-native-publication.md)
reuses this same GLM owner. Its recipe artifact is the exact retained qualified
M45/M47 wheel, delivered by `standard-off-recipe.json` without changing resource
pins. `release/execution-package-standard-off.json` binds OFF native; the
`-portable.json` sibling binds its explicit alternative.

`artifacts/native-promotion/` binds current artifact provenance, four exact
same-input primitive checks against the historical admitted native bundle,
installed negative gates, actual model native-entry/P6 dispatch observation,
bounded ordinary tools/append/branch/SSE and paired-session acceptance. Final
publication is `release/execution-package.json`; the separately identified
portable alternative is `release/execution-package-portable.json`. P5/P6,
paired-frontier ownership, replay=0/repack=0, numerical precision, allocator/wired
policy and long-context semantics are unchanged. Reusable historical qualification
is not relabeled as a new numerical/near-1M or general-runtime release campaign.

## One-way delivery

1. Develop and qualify in `ds41f-mlx`; retain attributed source/lock/toolchain inputs.
2. `tools/build_execution_package.py` constructs a profile-specific wheel. It
   inherits admitted dependency/native identities. DeepInfer MTP stages ordinary
   source and delivery metadata without Python-source sealing; preserved OFF still
   verifies exact declared payload equality.
3. Publish the wheel with its SHA and affected qualification. Install non-editably
   into a provisioned environment; never copy a mutable checkout into import paths.
4. Startup verifies installed execution and assets separately from host compatibility.
   Updating or deleting the development checkout/build scratch cannot change admission.

`mtp_setup` now follows this route for MTP only. The former OFF provisioning path
is retired; its R1 base exports and qualified artifact records remain archival.
Cross-profile startup is refused. This migration's fresh real-model acceptance is
**ordinary MTP**, not fresh OFF or full long-context/release qualification.
One-way repository promotion remains a source/build audit, not an installed
startup equality. R1's private OFF export verifier remains isolated qualification
with exact archive attribution, installed shared payload checks and exact weights;
it cannot admit an ordinary server. Historical evidence is not rewritten.

## Removed and retained machinery

Removed: MTP runtime source inventory, source-export rehash at startup, global
site-packages/distribution inventory equality, OS/Python patch equality, all-closure
SHA authority, native build provenance equality, MTP/OFF local `seal` actions,
and the old M41 seal-based negative driver. OFF ordinary startup also uses the
installed boundary, with no validation bypass.

Retained: source/projection audit tools for provenance and one-way promotion;
R1 qualification exports/fixtures; exact numerical resource/lifetime admission;
full checkpoint byte verification/cache; recipe capabilities and tokenizer pins;
canonical-state/frontier, no replay/repack, paired MTP/DSpark and recipe-authority
guards. State, scheduling, model math and protocol ownership were not changed.

## Qualification

See `artifacts/execution-admission/`: actual old/new pcre2 byte drift with the
**same** admitted recipe extension; installed Python/recipe/tokenizer drift
rejection; simulated semantic-native hash drift rejection; unchanged execution
digest with absent source exports and unrelated package additions; bounded
ordinary real-model tools/continuation/branch/SSE acceptance and settlement logs.
The old historical closure stays at pcre2 10.48. The new manifest contains no
pcre2 digest. These are boundary tests, not a claim of all-host or all-ABI coverage.
