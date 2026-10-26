# Phase 2 / Step 2E — changed-projection reconstruction

**A–E PASS within the model-free selection/integration scope; C2 PARTIAL;
UNDEPLOYED. No Production change or full-model load.** No Harness offload,
compaction, reacquisition, numerical reference, push, deployment or Phase 3 work.

## 1. Authority and conclusion

Started on clean `Multimodal-Runtime`, actual/expected HEAD
`f9fcc59fd6a7287cde1dc41c5199319cbba3e838`. Read the requested CURRENT contracts
and Phase 1 / Steps 2C, 2D and 2D-R1 evidence. Reused the completed DeepInfer audit
at `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`; no upstream research, ordinary
full-forward comparison or kernel investigation was repeated.

Inspected `paired_checkpoint.py`, `production_mtp.py`, `runtime/multimodal.py`,
`internal_mtp.py` and their existing tests. **No missing Production capability was
found.** Image-aware upstream lookup, exact restore checks, whole-span admission,
current-source preparation and the already qualified C=0/C>0 lifecycle paths
satisfy these bounded changed-projection cases without a source repair.
[Source identities](phase2-step2e-source-identities.json) bind the four unchanged
Runtime owners and the two changed test files.

For an independent Messages request, the **current Recipe projection** supplies
all tokens, image occurrences, source bytes and protocol content. Prior input is
not a source-recovery store. A reusable checkpoint must match execution identity,
exact current token prefix, all committed image identities and complete Target40 /
DSpark3 state at the same whole-image-safe C. The same authority selects an
eligible earlier pair or returns a miss; no legitimate pair is discarded to force
C=0. Missing current bytes remain an error, not permission to borrow old images.

## 2. Reused test owners and actual inputs

Extended `test_genuine_recipe_changed_history_never_false_hit` in
`tests/test_partial_messages_reconstruction.py`, rather than creating another
framework. Its existing `two_images` Recipe fixture now supplies shortening and
rewriting variants. The existing `run_start_scheduler_route` helper accepts an
already queued request with old snapshots present. That is test setup injection,
not patched lookup metadata: Scheduler performs its own normal acquisition.
The expected snapshot lookup in the helper is only a physical-state assertion.

Before/after pairs use the existing complete controlled Target/ring fixture and
**real authority capture/publication helpers** over actual Recipe token prefixes.
No successful hit is forged or arbitrary image coordinates inserted. These are
model-free paired states, not claimed real-model numerical checkpoints.

[Environment](phase2-step2e-environment.json): existing disposable
`/tmp/ds41f-phase1-1017-tests`, Python 3.13.15, installed Recipe **0.2.0**, oMLX
0.7.0, MLX 0.32.2, MLX-LM 0.31.4.dev132+g94cdcae13, Pillow 12.3.0. Recipe/native
conversion, rendering and encoding are actual installed behavior. The existing
pytest fixture independently admits installed dependencies in isolated Python,
then binds source-test numerics for controlled interfaces; this is **not installed
serving admission or a model numerical qualification**. No package was installed,
edited or replaced for this task.

The common old wire projection is the existing finite fixture:

1. User corn image + `Identify this food in one short sentence.`
2. Assistant `Corn.`
3. User carrots image + `Identify this food.`
4. Assistant `Carrots.`
5. User `State one visible feature.`

Old canonical N=669, images corn `[2,191)`, carrots `[206,650)`. Ordinary old pairs
are available at **2, 191, 667**. Shortening starts with only the old image-covered
pairs **191, 667** retained: no available valid text pair is removed to obtain a
miss. Rewriting uses the same history plus an `inspect` declaration and a completed
`tool_use` / `tool_result` pair, exact opaque ID **`Read:Exact-A`**, input `{}` and
result `inspection complete`, before the final feature request. Its old N=938,
images `[236,425)` / `[440,884)`, available pairs **236, 425, 936**. Tool declaration
prefix expansion is Recipe-owned, not manually positioned.

[Projection receipts](phase2-step2e-projections.json) retain the actual current
wire text/roles/tool IDs with base64 replaced by authenticated digest, canonical
int32-LE token SHA256, exact image digest/grid/start/length, old checkpoint
coverage, selected frontier, fresh/covered work, absolute fresh rows, controlled
transitions and published identities. Receipts come from pytest JUnit properties,
not another driver or cache owner. All current images remain historical under the
actual Recipe wire-origin rule; this matrix never relabels them.

## 3. Positive matrix and actual selection

All seven rows complete controlled continuation, settlement-fenced publication
and request/source retirement. No Cartesian fault/partition campaign was added;
the same lifecycle helper and existing negative controls are shared.

| Case / actual current change | Current N and spans | Actual C | Fresh / covered patches | New compatible publication |
| --- | --- | ---: | ---: | --- |
| **A1 — changed corn bytes**: RGB invert, re-save JPEG at original dimensions | 669; `[2,191)`, `[206,650)` | **2** | **5325 / 0** | 667, 670 |
| **A2 — changed carrots bytes**: same operation on the later image | 669; same spans | **191** | **3774 / 1551** | 667, 670 |
| **B1 — reorder**: swap distinct image blocks between actual user messages; do not reorder text | 669; carrots `[2,446)`, corn `[461,650)` | **2** | **5325 / 0** | 667, 670 |
| **B2 — move**: prepend actual text block `Look carefully.` before first image | 672; corn `[5,194)`, carrots `[209,653)` | **2** | **5325 / 0** | 670, 673 |
| **C — shorten**: remove first corn user/assistant segment, retain later valid messages | 465; carrots only `[2,446)` | **0 miss** | **3774 / 0** | 463, 466 |
| **D — rewrite**: replace first assistant `Corn.` by `A different prior description.`; retain later images and completed tool pair | 941; corn `[236,425)`, carrots `[443,887)` | **425** | **3774 / 1551** | 939, 942 |
| **E — unchanged**: exact common old projection | 669; original spans | **667** | **0 / 5325** | Existing 667 remains reusable; final 670 |

### Why stale longer pairs cannot cross the change

- **A1/A2:** canonical token IDs are **exactly identical** to the old IDs; grids
  and lengths are unchanged. The old/new token digest is
  `a5548e2d7212f60454a23e076409d56693e051cbe294e2f635ce65c1a73b5528`.
  Different source digest excludes the image-covered longer pair. A1 keeps the
  valid text C=2; A2 preserves the unchanged corn pair at C=191 and encodes only
  replacement carrots. This is not token-only reuse.
- **B1/B2:** Recipe recalculates expanded IDs, source order and absolute positions.
  Old image-covered keys/prefixes cannot restore. The genuine text C=2 remains
  eligible; no image ordering or positions are normalized for cache reuse.
- **C:** old C=667 exceeds the new N-2 lookup boundary. Old C=191 contains corn,
  whereas the current first occurrence is carrots and would intersect that
  frontier. Image identity excludes it; the actual result is C=0, never fractional
  image reuse. A source observer at **current preparation** asserts removed corn
  is never supplied; current `_sources` contains only carrots. Old sources are
  explicitly released before continuation. No old source/embedding/history store
  participates in recomputation, and no obsolete source is required by execution.
- **D:** actual earlier token divergence and shifted later span exclude C=936.
  The unchanged whole corn pair C=425 remains eligible and is preserved. Recipe
  conversion retains the exact tool-use ID and ordered completed result in both
  projections. This is an ordinary text rewrite test, not a fabricated summary,
  tool-effect replay or compaction implementation.
- **E:** longest eligible C=667 is retained, not degraded to a shorter prefix or
  forced miss. Both images skip actual `encode_new` work (zero fresh rows).

Healthy old pairs with different image keys may remain valid for the old branch;
exclusion from the current lookup need not evict them. In cases with equal N,
old/new C=667 roots can coexist with **different image identities**. Every asserted
new publication is matched by both current tokens and current image identities,
not by frontier number alone. Ordinary bounded upstream LRU eviction still applies.
Damaged-pair fallback/invalidations use the existing authority, not a new selector.

## 4. Controlled lifecycle and work evidence

The real `_schedule_waiting`, backend `_start`, Scheduler stepping, N-2 capture,
publication gates and retirement methods execute. P6 numerical work, native P5 /
BatchGenerator and downstream advance/settlement witnesses use the same controlled
interfaces as Step 2D. Thus **controlled settlement/publication integration** is
proven; no row is relabeled as fresh native generation or a Recipe numerical
accuracy certificate. Real Recipe semantic completion/failure guards are also
covered separately in the focused suite. Prior Steps 2B-R2 and 2D-R1 retain their
separate bounded native lifecycle evidence.

The helper asserts exact restored current prefix and all controlled Target/ring
values before append, independent immutable selected-pair values after append,
complete fresh source authentication and whole-image embedding row lengths at
absolute positions. Actual CPU patch preparation/`encode_new` is used; only the
span encoder's numerical result is deterministic complete-length output. No saved
embedding is reused. Covered occurrences produce no row/encoder work.

Each reconstructing row executes **C→N-2→N-1**, all **40 Target / 3 DSpark**
frontiers align at pending capture and completing append, and P5 receives the
held-out terminal through the sole Scheduler batch. Both UID maps agree during
controlled advance. Canonical current tokens plus the controlled output reach the
settlement witness; only then are complete immutable prompt/final pairs published.
Requests, queue, maps, batch, active owner, pending capture and sources retire.
For unchanged E, **C=N-2** already: existing capture followed by the genuine
**667→668** append and P5 is correct, not a synthetic zero-length handoff. The
helper's former two-append assumption was generalized in tests only.

Corn is 1,551 patches / 138,600 pixels; carrots is 3,774 patches / 717,824 pixels.
Inversion preserves geometry. A1/B charge both (**5,325 patches / 856,424 pixels**);
A2/C/D charge only current uncovered carrots; E charges zero fresh work. All totals
fit the normal **36,864-patch** envelope. Current input-container/body/source-pixel
admission remains separate even for covered images. No aggregate historical
patch-discount callback, 12 MiB projected-pair / 192 MiB retained-root allowance,
source-byte/pixel/context/output/Files/queue limit or lease policy changed.

## 5. Negative and failure containment

The focused suite reuses existing controls rather than multiplying every fault
across all seven projections:

| Protected boundary | Evidence / safe result |
| --- | --- |
| Same placeholders, different image digest | A1/A2 actual Recipe identity exclusion and positive fresh reconstruction |
| Missing/corrupted current fresh sources | Pre-native refusal; sources/queue retire, no publication or healthy-engine burn |
| Changed committed identity | Direct restore/prefix and image-aware key controls exclude stale state |
| Image-intersecting frontier | Capture/prefix/workload refuse; incompatible current image cannot become a fractional hit |
| Incomplete Target/DSpark | Real authority invalidates damaged candidate to an actual earlier pair or miss |
| Lost exact retained logical Chat lease | Exact retained acquisition fails; no ordinary shorter-hit/cold reconstruction fallback |
| Ordinary independent projection | All seven rows have no logical session and use ordinary lookup, not retained-session failure policy |
| Fresh patch/pixel overflow | Existing pre-native physical refusal; no wire-age discount |
| Conditional aggregate discount / paired-byte refusal | Existing callback historical-coverage and pair/root byte requirements retained |
| Queued cancellation | No Vision/P6/P5 transition; transient sources/request retire |
| Post-native mutation/handoff failure | Existing burn/fatal fence clears reuse and retires; no replay or retry |
| Incomplete Recipe/settlement boundary | Semantic guard negatives and controlled missing-quiescence test refuse publication |
| Rejected input with healthy retained pair | Strengthened existing pre-native test asserts identical retained payload, every Target/ring byte unchanged, no publication |

No new refusal, production bypass, alternate owner, fallback selector or protocol
was introduced. Initial test development found a collection parenthesis typo and
an assertion assuming plain assistant `tool_calls` was iterable rather than None;
both were corrected in tests. Neither was a Production or reconstruction defect.

## 6. Validation, integrity and remaining scope

[Focused log](phase2-step2e-tests.log): **228 passed**, 73 existing deprecation
warnings, **52.05 seconds**. Includes actual Recipe Messages projections, paired
identity/fallback/restore, fresh/cached Vision, controlled P6/P5 ownership,
settlement/publication/retirement, exact session policies, cancellation, source /
resource refusal and failed-state fences. No unrelated full test campaign,
new environment, model load, ordinary-forward reference or installed-package edit.

[Operational integrity](phase2-step2e-integrity.json): read-only postflight checks
all **817** existing inventory paths against the retained Step 2B-R2 baseline
(also used by Step 2D-R1). SHA256/size/mtime/inode match exactly. The original
Runtime was not imported for execution or modified; no operational weight
admission/startup. This reuses the inventory's atime/bytecode exclusions, not a
whole-volume attestation. No new task-owned installation, model process or
Supervisor was created; no model-memory/resource-peak claim follows.

Final `git diff --check` and `python tools/check_repository_self_containment.py`
pass. Diff review: only two existing test files, scoped CURRENT summaries and new
evidence. Four inspected Production owners match starting HEAD byte-for-byte.
Historical reports/dispositions remain unchanged.

**Disposition:** A–E model-free selection and controlled continuation **PASS**;
no case in this finite matrix is BLOCKED. **Native changed-projection numerical
and physical-resource qualification remains PARTIAL/unqualified**, not silently
inferred from controlled state or prior native schedules. Same-schedule controls
reuse the same MTP path; differing partitions require compatible committed state
and suffix alignment, not bitwise cold equality. Prior MTP/Ordinary and
shape-dependent numerical observations remain unresolved; no `wkv` research,
quantization change, fitted tolerance or speculative normalization.

The smallest remaining Phase 2/C2 boundary is qualification beyond this bounded
selection/integration evidence: changed-projection native numerical/resource
scope, broader conditional-discount reconstruction and true image-intersecting
execution remain unqualified. The latter is not enabled by whole-image fallback.
C2 is not globally PASS. C3–C5, C8 and Harness-specific C9 remain unqualified;
these artificial Runtime projection edits are **not official Harness offload or
compaction**. No automatic next task. Stop after Phase 2 / Step 2E.
