# GLM native profile promotion assessment

Status: **CLOSED — NATIVE IS VIABLE, WITH BOUNDED PROMOTION WORK; IMPLEMENTATION DEFERRED**.

This is the frozen pre-implementation assessment. Its statements about installed
portable defaults and missing promotion gates describe that assessment revision,
not today's distribution. The subsequent bounded promotion is complete: native
is the local MTP default; portable remains explicit. Current authority is
[installed distribution admission](runtime-admission.md#glm-execution-distributions)
and `release/execution-package.json`; the assessment evidence below is unchanged.

OFF follow-up: **CONFIRMED_AUTHORITY_DRIFT — CLOSED BY BOUNDED PUBLICATION**,
not “No further authority drift”. The [OFF publication](standard-off-native-publication.md)
reuses the MTP GLM authority and exact retained M45/M47 recipe wheel (no recipe
re-admission). Installed native/P5/P6 and production selected-path gates passed;
standard-OFF now defaults to native, with portable explicit. The frozen assessment
below still describes its historical revision.

This is a distribution decision, not a continuation of the closed
[long-frontier suffix investigation](long-context-suffix-investigation.md).
No default, installed environment, execution manifest, resource pin, model math,
state/replay/repack semantics or release authority was changed.

## Why portable is the current distribution authority

The repository provides a concrete chain, rather than a native reliability policy:

1. **M41 (`03516fe`)** delivered `third_party/mtp/omlx-source.tar.gz` and
   `mtp_setup.py`. Setup ran an ordinary `uv pip install ... build/omlx`.
   The exported `setup.py` builds extensions only with
   `OMLX_WITH_CUSTOM_KERNEL` or `--with-custom-kernel`; neither was enabled.
   Current setup additionally removes ambient `OMLX_*` variables from its child
   environment. The default source installation therefore builds no GLM bundle.
2. **M42 (`08690aa`)** froze the private OFF source in R1.
   `tools/finalize_r1.py` explicitly says “not ... prebuilt kernels”, selects
   Git-tracked `.py/.metal/.h/.cpp/.c/.json` files, and excludes native binaries.
   It also omits CMakeLists/common build material from this private OFF export.
   This was a scoped reference-source selection, not evidence of native failure.
3. **M53R (`30710aa`)** changed recipe delivery to an exact M52R wheel and froze
   the normal-local dependency inventory. It did not enable GLM compilation.
   `normal-local.json['omlx']` contains no native GLM members.
4. **Installed admission (`0550922`)** inherited that MTP inventory. OFF inherits
   the R1 source-only oMLX export and its separately admitted numerical dependency
   payload. `build_execution_package.py` rejects additional/deviating dependency
   files; it does not build or enroll optional native kernels. Current manifest,
   `normal-local.json` and `standard-off-dependencies.json` all omit the bundle.
   Later MTP overlays preserve this delivery choice.
5. **M47** explicitly admitted both native and extension-absent portable resources;
   **M48** retained that distinction in first-party OFF model execution.
   Historical 200K/very-long-context qualification retained the exact native
   bundle in a separate resource environment. It was not publication of that
   bundle through the ordinary provisioning route.

Thus the positive explanation is **opt-in build not selected + source-only OFF
export + exact inheritance of the accepted portable candidate**. No reviewed
repository decision found in this chain weighs native instability, target-host
portability, maintenance cost or numerical inferiority against portable. Do not
turn the absence of such a decision into a claim about its author's intentions.
“Portable native wheels” in older release exclusions describes packaging/deployment
scope, not a judgment against the GLM native execution profile.

The source/reference scoping remains valid for frozen R1. Exact dependency
inheritance remains valid admission policy. Neither is a current architectural
reason to exclude native forever on the supported M3 Ultra local runtime.
Portable is nevertheless the **accepted installed authority today**: historical
native admission is not permission to modify that authority by observation.

## Implementation, build and identity: distinct results

| Boundary | Finding |
| --- | --- |
| Implementation | Present in the current attributed oMLX export: C++, Metal, nanobind bindings, GLM CMake and shared common sources. Source-only R1 is not itself a complete native build distribution. |
| Historical/current source | 43 GLM/common build-input files in the current export match the preserved historical native source byte-for-byte, including CMake. This is source identity, not proof of the old compiler invocation. |
| Current build | Two independent GLM-only Release CMake builds succeeded from the delivered export with Python 3.13.15, MLX 0.32.2, nanobind 2.15.0, CMake 4.3.1, Apple clang 21 / SDK 26.5, deployment target 15.0. No other optional model kernels were built. |
| Native ABI/kernel loading | Fresh scratch MLX dylib/metallib match current OFF resource pins. Native ABI probe passes and all 23 declared symbols exist. Installed-layout packed-attention GPU/metallib smoke passes with original source/build/output paths hidden. No model was loaded. |
| New build vs admission | **All three new file hashes differ from the historical native pin.** They are not admitted resources. No policy refresh was performed. |
| Reproducibility | Dylib and metallib hashes match across the two build/output paths; extension hashes differ. Build/load reproducibility is demonstrated, byte-reproducible extension publication is not. Its Mach-O rpaths include relative installed-layout entries and absolute build/environment entries. Freeze one accepted artifact rather than require local recompilation to recreate its exact identity. |
| Dependency graph | GLM links sibling ops dylib, admitted `libmlx`, system frameworks/libc++; no new OpenCV/Homebrew numerical subgraph was observed. Recipe/image dependency policy is unchanged. |
| Ordinary installed admission | Exact inventories currently reject insertion of this bundle. Merely adding filenames to a manifest neither attributes a build nor accepts its numerical/selected-path behavior. |

The new three-file bundle is approximately **3.54 MB**. No evidence supports a
large distribution-size or new external-dependency cost. The real recurring cost
is ABI/toolchain-bound artifact maintenance and affected acceptance when changing
MLX/nanobind/native code, not a runtime optimization program.

The upstream `OMLX_WITH_CUSTOM_KERNEL=1` switch builds **five** optional extension
families. It is not the minimal promotion implementation: provision only the
selected GLM bundle, with exact attribution and identity, rather than adding
unrelated native models to execution inventory.

## Admission and selection gap that matters

OFF `resource_admission.check_native` already verifies the exact three members,
actual extension origin, loaded Mach-O UUID and unique dylib. A failed import or
ABI-disabled `_ext` with files present is rejected as a mixed/broken portable
bundle. New hashes must undergo an explicit admitted-policy update; deleting the
old native pin or loosening exact inventory is unnecessary.

Ordinary MTP uses **installed payload admission**, not the OFF resource capability.
`execution_identity.inspect_payload` verifies complete oMLX/MLX bytes, but its
native load/UUID/host checks target recipe/OpenCV. It does not require the GLM
native profile or probe GLM loadability. The attributed `fast.py` catches import
errors and disables symbols on nanobind ABI mismatch. That is harmless for the
present extension-absent portable delivery, but **a native MTP manifest alone
would not prevent admitted files from executing portable branches after a native
load failure**. `InternalMTPQualificationBackend.load` uses the oMLX loader directly;
OFF's GLM guard cannot be presumed to run there.

Promotion must bind the selected GLM profile explicitly in package authority and
use a shared, exact loaded-bundle/ABI/symbol check for installed MTP as well as
OFF. Inspect/start/health/operator presentation should disclose `GLM native` or
`GLM portable` separately from `mtp-serving-v1` / `standard-off`. This is a small
admission/selection-UX responsibility, not a new model execution owner. A startup
native failure must refuse, never automatically downgrade. No per-request native
selector, hot switching, execution override or resource-pin bypass is needed.

## Evidence reusable without requalification

- Official-model numerical/precision qualification, P5 held-out terminal handoff,
  P6 append, state ownership, frontier/replay/repack and resource-lifetime policy
  remain authorities for their unchanged source/execution boundaries.
- M46/M47 native matched 4K/32K tokens and all cache slots; M48 first-party source
  ownership; historical standard-OFF 200K and very-long native continuation and
  wired-budget repair remain reusable. No new kernel or resource-policy defect
  was found that would reopen these campaigns.
- The closed suffix investigation's **current core + exact old native bundle**
  matched historical 32 probe tokens, 280 post-probe slots and preceding 768
  generated tokens. The matched 2,049-token suffix was 5.636 s versus historical
  8.636 s. This establishes current OFF native viability, not fresh installed
  native/MTP acceptance. Current portable OFF/MTP's approximately 10.17/10.20 s
  is a different matched workload/frontier, not a precise native speedup ratio.
- Current portable ordinary paired continuation/settlement and release acceptance
  remain valid for portable. Historical singleton or native OFF receipts do not
  establish current native ordinary MTP target/draft verification with today's
  overlays. There is no need to reprove profile-independent protocol/state
  machinery, but the newly selected dispatch must actually be exercised.

Native and portable are **intentionally separately admitted numerical trajectories**
under M47/R1's backend-local fidelity policy, not bit-equivalent settings. The
observed token divergence is not itself evidence that either profile is incorrect.
A rebuilt artifact is not accepted solely because its source matches or its ABI
probe passes. Nor should new generated prose be compared to a portable golden as
if bit equality across profiles were required.

## Bounded promotion requirements and owners

1. **Dependency/artifact owner:** select a repository-owned exact GLM artifact
   with source/toolchain/build provenance and MLX/nanobind ABI identity. Either
   explicitly deliver the already admitted historical artifact (after attribution
   and relocation acceptance), or qualify/freeze the current rebuilt bundle.
   The old bundle is currently supplied by an external historical environment,
   not by the ordinary repository distribution; copying its path is not closure.
   For a rebuild, reuse unchanged source evidence and perform only affected native
   primitive/dispatch equivalence or fidelity checks against the admitted native
   implementation, including packed attention and used MoE/projection entries.
   No full-model numerical campaign or benchmark ladder is justified.
2. **Package/provisioning owner:** add an explicit GLM execution-profile choice
   at package/provision time and an exact profile-specific oMLX bundle overlay
   authority. Preserve the existing Python/MTP overlays and portable inventories.
   OFF must consume the bundle as an attributed addition to immutable R1 sources,
   not rewrite R1. Native policy and installed manifest must refer to the same
   accepted bytes. Test wheel membership, additional/missing/altered members and
   fresh install without donor/build imports. Do not observe/reseal the shell's
   dependency state. Existing wheel/receipt machinery is sufficient.
3. **Admission/operator owner:** require and report actual selected GLM execution
   profile; check loaded origin/UUID, ABI array caster and required symbols.
   Negative tests cover present-but-unloadable, ABI-disabled and partial bundles,
   not just inventory drift. Share existing guards without relaxing identities.
4. **Selected-path acceptance owner:** use the exact installed native candidate
   for bounded ordinary MTP preparation/P6/P5, target verification with native
   drafts, paired continuation/branch and protected settlement; reuse the existing
   ordinary acceptance fixture. Exercise one eligible multi-row suffix and record
   native dispatch. If publishing OFF too, check its exact installed native
   admission/terminal handoff/idle continuation. Do not rerun 200K–1M ladders.
   Bind artifact/package digests to receipts; tiny ABI smoke is not model acceptance.
5. **Publication/operator owner:** stop/settle before switching packages and start
   new sessions. Retain a separately selectable portable package; no implicit
   failure fallback. MTP has no persistent pair migration to perform. OFF M9
   artifacts do not carry an enforced GLM-profile field, so portable/native restore
   must not be presented as same-trajectory continuity. Use separate artifact
   namespaces/explicit same-profile restore operationally; no cache conversion,
   replay, repack or persistence-schema redesign is required for MTP promotion.

These are the promotion boundary, not authorization for general release expansion.
The new-build identity and current native MTP selected-path evidence are genuine
missing requirements, even though broad native correctness qualification is not.

## Default and distribution recommendation

**Promote native as an explicitly admitted normal-local package once the bounded
gates above pass; then prefer native for this M3 Ultra/512 GB single-operator
runtime. Do not change default in this investigation.**

The supported hardware is already narrow, so generalized portability provides
little local benefit. Preserved numerical/state/native evidence, materially lower
observed long-frontier latency, unchanged dependency graph and a small artifact
favor native for the main interactive use. Exact artifact publication avoids
operator recompilation surprises; explicit startup refusal avoids silent trajectory
changes. Maintaining both profiles adds one bounded artifact/acceptance branch,
not another runtime design. No observed current correctness/resource issue requires
native to remain permanently non-default.

However installation/artifact maintenance and native ordinary MTP acceptance have
not yet closed. Portable retains the current default because it is the actual
accepted installed candidate, **not because it is known to be safer or because
native latency is unimportant**. This is viable/promotion-work-defined, not
“production-ready” or “promotion-ready with all gates passed”. A native package
should become the next local default only with its publication receipt. Portable
remains a supported **explicit alternative**, not a hidden execution fallback;
standard-OFF versus MTP is a separate serving-profile choice.

## Work performed, validation and release impact

Only this assessment/navigation and bounded scratch evidence were added.
`artifacts/native-promotion/` retains source hashes/comparison, two configure/build
logs, build/ABI identities, relocation/metallib smoke and test results. Scratch
builds were outside repository/runtime environments; no model process or installed
runtime was modified. New binaries were not published or silently enrolled.

Initial boundary tests: **49 passed, 2 failed**, both because the development
configuration still names an absent historical recipe tokenizer path. With explicit
current recipe configuration: **50 passed, 1 deselected**; the remaining test uses
`DEFAULT_RECIPE` directly and is not environment-configurable. Failures are retained,
not relabeled as acceptance. No native numerical or ordinary-model PASS is claimed
from these unit tests. Repository documentation/containment check and **46
documentation-authority tests pass**; `git diff --check` passes. Receipts are
retained alongside the build evidence.

Current portable release readiness is unchanged. Native installed/default readiness
is blocked only by the enumerated artifact/admission/provisioning and selected-path
acceptance boundaries. This closes the decision and stops here: no performance
optimization, MTP redesign, full qualification campaign or release infrastructure
expansion follows from the latency observation.
