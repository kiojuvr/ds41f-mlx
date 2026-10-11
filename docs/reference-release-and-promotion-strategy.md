# Reference Release and Promotion Strategy

## Authority and scope

This page owns **development authority, semantic reference and prospective source
projection rules**. [Qualification](qualification.md) owns evidence interpretation;
[R1](reference-release-r1.md) owns its frozen executable contract;
[operations](operations.md) and [installed admission](runtime-admission.md) own
package publication/deployment. None of these is proof that later HEAD is releasable.

`ds41f-mlx` is the sole development repository. The existing `ds41f-runtime`
directory is unused old generation, not deployment or accepted release history.
A permanent second repository is not required. Future projection, if explicitly
selected, must follow the one-way rules below; documentation edits do not trigger it.

## Repository roles

### `ds41f-mlx`

Own implementation, semantic/numerical/lifecycle development and qualification,
reference/negative evidence, experiments, release surface and promotion tooling.
It may retain more historical/investigation material than an end-user runtime.
Do not delete useful evidence just to make delivery visually minimal.

### `ds41f-runtime`

This name may identify a future qualified source projection, not an independent
implementation, peer development line or continuously synchronized mirror.
All projected source must have authoritative origin in `ds41f-mlx`; the existing
directory does not fulfill that role. Source-clone operation/package publication
can be supported without maintaining it.

## One-way development authority

```text
requested change / release defect
  -> ds41f-mlx implementation + affected qualification
  -> qualified development milestones may accumulate
  -> separate explicit release/promotion checkpoint
  -> qualified source projection / published payload
```

No runtime-only semantic fixes, reverse merges or release-only implementation
branch. Return defects upstream, qualify there, then replace the selected payload.
Emergency scope may be narrowed with evidence, never reverse authority or bypass
required ownership/numerical gates. A release-surface edit does not force immediate
clean-room setup, receipt renewal or source projection.

## Semantic authority hierarchy

Official checkpoint/model/protocol semantics outrank implementation qualification,
which establishes a scoped Reference Release oracle for candidate conformance and
later release decisions. A Reference Release is not permission to preserve a
"golden bug". Changed official semantics, backend/platform or evidence of a defect
requires investigation and affected qualification; explicitly revise/version the
reference where needed, not expected outputs patched to fit a candidate.

## M42 — Semantic Closure and Reference Release

This is a **completed historical boundary**, not the next work request.
M42 established repository-owned R1 material, independent of historical donor
checkouts for its declared conformance scope. It performed no M43 extraction.
The [earlier strategy edition](archive/editorial-baseline/reference-release-and-promotion-strategy.md.txt)
retains the original proposed gate, scope inventory, examples and chronology.

### Reference Release R1

The current [R1 contract guide](reference-release-r1.md) names immutable material,
identity and verifier. R1 binds its checkpoint/tokenizer/precision, execution/state/
protocol/lifecycle/negative fixtures and bounded profile scopes. Exact canonical
and ownership predicates are distinct from universal floating tensor identity.
Owned generalized fixtures do not expand public capability or later release scope.

### Reference conformance

The implemented interface is `python -m ds41f_mlx.reference`; flags and outcomes
are described by R1. The old proposed CLI names are history, not missing work.
Seam conformance and real-model conformance are different decisions. Neither
alone is installed startup admission, maximum-workload qualification or promotion.

### Semantic-closure gate

Historical closure applies to declared R1 inputs and owned observations. A later
implementation must satisfy applicable unchanged contracts and affected model/math
qualification; a passing fake-cache seam is not mathematical backend fidelity.
Absent dependency, skipped predicate or insufficient scope must not be called PASS.

## M43 — Release Repository Extraction

M43 implemented and scoped-qualified **deterministic Git-object projection**.
It was not runtime reimplementation and did not certify later source states or
today's old runtime directory. Its existing mechanism is `tools/promote_release.py`
over `release/runtime-surface.json`; no new promotion infrastructure is requested.

At a selected checkpoint the mechanism requires clean declared HEAD, exact R1
material/verifier/dependency archives and explicit inclusion/exclusion for owned
source roots. Required entries must be regular files with safe destinations.
Projection reads Git blobs, not ambient checkout mtimes/build junk. Output must
be empty and outside development; it never reconciles runtime edits back upstream.
It does not build binaries or establish model qualification by copying files.

### Promotion manifest

`ds41f.promotion.v1` binds source commit, reference/semantic contract, surface/tool
identity, per-file origin/hash/size/mode, canonical payload-tree identity, dependency/
checkpoint identities, declared nondeterministic artifacts and inherited receipts.
The derived envelope is checked separately from its canonical files-map digest.
Identical authoritative input reproduces the source payload.

Qualified promotion must bind the selected payload and reference through a committed
PASS receipt; an ancestor commit alone is insufficient if the payload-tree hash
differs. The tool labels a candidate `NOT_RELEASE_QUALIFIED` when no qualification
receipt exists and rejects a present mismatched receipt. Do not erase or regenerate
old receipts merely to make later HEAD extractable. Current added owners still
need reviewed surface classification and selected-payload acceptance; historical
mechanics tests do not make HEAD promotable.

## Release repository contents

An explicitly selected projection must deliver source/native/Metal/build/dependency
material, required semantic assets, public contracts, configuration/provenance and
its supported conformance surface. Development-only investigations, obsolete
harnesses, donor comparison machinery and large evidence can remain in development.
The actual allowlist is the reviewed surface, not a generic file-name heuristic.
New owners must be explicitly classified; do not expand it merely because a checker
or frozen receipt rejects later HEAD.

## Relationship to temporary implementation substrates

Source closure is not implementation-ownership closure. Attributed reproducible
dependencies may remain without external developer checkouts. The OFF first-party
core and MTP pinned-oMLX physical authority are separate current architectures.
Delivery/dependency-count reduction does not authorize rebuilding qualified MTP
topology and recovering speed later. [MTP doctrine](mtp-architecture-reset.md)
governs structural changes; no automatic donor-removal milestone follows from R1.

## Future development after R1

Development needs affected tests, applicable reference contracts and necessary
model/lifecycle/performance evidence. Stable-boundary full R1 regression is a
separate decision under [current priorities](implementation-plan.md), not a full
campaign after every milestone. Source-scoped old receipts remain valid for their
inputs; neither deferring regression nor passing mechanics qualifies later HEAD.

A future promotion needs deterministic projection, fresh independent setup,
release-profile acceptance and bound receipt. Broader feature/security/default-MTP
approval remains separate. New semantics outside R1 require authority review and
potential R2, not invented inherited coverage.

## Release-repository change policy

1. Fix and qualify in `ds41f-mlx`.
2. Preserve source/reference identities and historical receipts.
3. Select projection/publication as an explicit checkpoint, not synchronization.
4. Qualify the exact chosen payload and independent setup under its admitted scope.
5. Replace derived delivery; never introduce release-only runtime semantics.

Installed-wheel verification and host ABI compatibility are current startup
boundaries, not equality with a developer checkout, SDK or local source seal.
Source projection remains a source/build audit; it cannot admit ordinary startup.

## Relationship to the final runtime target

[Final target](final-runtime-target.md) is prospective source closure and documented
reproducible setup, not mandatory installers/vendor removal or a second repository.
If projection is chosen, users must not need the development evidence environment
to operate it. Developers still work upstream. Scoped package/source-clone release
approval is not completion of that destination or a general-runtime release.

## Planned progression

The original M41→M42→M43 sequence is historical and retained in the edition.
M54R/M55R later approved a bounded normal-local singleton source-clone release
with projection deferred. Ordinary MTP subsequently has its own implemented
production semantics. None of those decisions automatically promotes default MTP,
certifies later installed payloads or closes the general-serving release boundary.
Current work is not a new extraction or repeat qualification campaign.

## Architectural invariant

**One development authority, explicit semantic qualification and one-way derived
delivery; no direct release-only implementation line.** Reference versions,
packaging forms and supported envelopes may evolve without reversing that invariant.
