# Installed execution admission

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
| Execution identity | `execution_manifest.json`: complete installed ds41f, attributed oMLX/recipe, numerical/tokenizer and reachable protocol-support package payloads, including bundled extensions/Metal kernels; exact content and Python bytecode checks |
| Semantic assets | Exact checkpoint metadata, 48 shard byte hashes and recipe tokenizer; existing file-identity cache avoids rescanning unchanged weight files, never trusts LFS metadata alone |
| Semantic native | Recipe image calls reach OpenCV core/imgproc/imgcodecs; their Homebrew subgraph is exact, conservatively including codecs, compression, OpenBLAS, TBB and Fortran support. Hashes inherit the previous qualified candidate, not today's observation |
| Host compatibility | Apple Silicon, supported macOS 26 ABI family, Python 3.13 >= 3.13.15, M3 Ultra/512 GB MTP capacity; successfully loaded native recipe with required semantic methods; resolved native graph and consumer/provider Mach-O ABI contracts. No OS patch equality, unrelated distribution inventory or host-closure SHA equality |
| Build provenance | Source commit plus package-input digest (including uncommitted input), attributed exports/patches, native wheel/toolchain provenance and lockfiles. Retained for audit/reproduction; never compared to today's checkout, compiler or SDK at startup |
| Qualification evidence | R1, historical `normal-local.json` runtime/closure/environment hashes and support-package candidate evidence. Build qualification inputs only; absent from runtime admission paths |

MLX is platform support **and** model execution. Its executed package, native
library and kernels remain exact; calling it a host dependency does not excuse
numerical drift. Likewise tokenizers, Transformers, Pillow and recipe image
codecs can alter model inputs, so their installed payloads remain exact.
Transport/protocol Python dependencies are exact where they execute the admitted
behavior. Unrelated packages, compiler tools and test runners are not inventoried
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

Normal-local **MTP defaults to GLM native** on the supported M3 Ultra deployment.
`third_party/mtp/execution-profiles.json` records the explicit decision; portable
is an admitted explicit alternative. OFF publication/default is unchanged.
Serving mode (`mtp-serving-v1`/`standard-off`) and GLM execution profile are distinct.
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
No OFF-native package is published by this MTP promotion.

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
   inherits admitted dependency/native identities, refuses candidate drift, and
   verifies that the wheel ships precisely its declared ds41f payload.
3. Publish the wheel with its SHA and affected qualification. Install non-editably
   into a provisioned environment; never copy a mutable checkout into import paths.
4. Startup verifies installed execution and assets separately from host compatibility.
   Updating or deleting the development checkout/build scratch cannot change admission.

`mtp_setup` follows this route for both MTP and standard-OFF; OFF uses its separate
R1 base oMLX export and built base recipe wheel, not the patched MTP payload.
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
