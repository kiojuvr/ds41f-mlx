# Phase 2 / Step 2D-R1 — native partial-prefix continuation

**INSTALLED_ADMISSION_PRESERVED / NATIVE_C_GT_0_CONTINUATION_EXECUTED /
PARTIAL_PREFIX_RESTORED / FRESH_SUFFIX_VISION_VERIFIED /
CANONICAL_SETTLEMENT_VERIFIED / PAIRED_PUBLICATION_VERIFIED /
NUMERICAL_UNRESOLVED / PARTIAL.**

One admitted model load, two actual independent Messages requests, one genuine
C=203 hit. Both requests generated, settled, published and retired. C2 remains
PARTIAL; no cross-partition equality, ordinary-forward reference, deployment,
retry, push or Phase 3 work. Historical Step 2D evidence is unchanged.

## 1. Starting authority and scope

Started on clean `Multimodal-Runtime`, actual/expected HEAD
`ae78f7c84da1b2741d4012053f41a66b7130ed64`. Read the requested policy, plans,
Runtime admission/ordinary execution contracts and Steps 2D, 2B-R and 2B-R2;
inspected their workers/receipts and the installed admission/import path. No
upstream re-audit, numerical-kernel investigation or Production change.

The operational `/Volumes/SDXC-512/ds41f-1017` was not modified or used for weight
loading. Constructed `/tmp/ds41f-step2dr1` non-editably through official
`mtp_setup`, Flash-1017/native GLM, from the starting HEAD. Task changes are only
external diagnostics/tests, evidence and scoped CURRENT documentation.

## 2. Confirmed import-order cause and correction

Step 2D's exact failure was `ValueError: non-installed executable origin:
transformers` in `inspect_payload` → `check_installed_packages`, after its worker
had imported serving modules and prepared Recipe projections.

Fresh isolated zero-weight processes characterize both orders:
[early](phase2-step2dr1-import-early.json), [late](phase2-step2dr1-import-late.json).
Before imports, Transformers' spec origin is the disposable installed
`site-packages/transformers/__init__.py`, with one **str** search-location entry.
After the production-serving import graph it is the same origin and resolved
directory, with a **PosixPath** entry. Module type becomes `_LazyModule`; resolved
locations and the actual `__init__.py` SHA256 are unchanged. Late normal admission
reproduces exactly the prior refusal. This is metadata representation drift, not
observed package-origin redirection.

Installed Transformers `processing_utils.py` calls
`direct_transformers_import(Path(__file__).parent)`; that helper passes `[path]`
to `spec_from_file_location(..., submodule_search_locations=...)`. Existing Runtime
admission compares the list literally against strings. Importing Transformers
alone did not reproduce every transition: preliminary import-only probes passed
late admission, and an initial combined zero-weight probe command timed out.
The full production-serving import order is the qualified reproducer, not a
universal claim about every lazy import.

The new [worker](phase2-step2dr1-worker.py.txt) imports only the minimal admission
diagnostic before calling `prove_installed_admission`. That existing gate performs
normal installed payload and Runtime/checkpoint admission **before** importing
serving/numerical consumers for origin checks and materializing the BF16
`pre_woa` fixture. Application serving/Recipe imports and canonical preparation
follow the completed gate. It then checks authenticated module/function objects,
profile and installed origins remain unchanged, without another `inspect_payload`.
The actual load/serving path required no redundant post-import inspection.

No `execution_identity.py` change, spec normalization, origin-check replacement,
`_admitted_variant` injection, checkout overlay, module purge or diagnostic reload.
Ordinary Transformers behavior is allowed only after normal admission; the same
authenticated numerical module graph is retained throughout.

## 3. Admission, zero-weight gate and installation identities

[Admission](phase2-step2dr1-admission.json) records full actual installed origins,
native closure, checkpoint inventory and lifecycle signatures. Full 48-shard
byte verification ran in the real worker before loading. Checkpoint:
`/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, official
revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

- Runtime admission identity:
  `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
- Numerical profile: `flash-1017-pre-woa-fp8-v1`, GLM native.
- Python 3.13.15, oMLX 0.7.0, MLX 0.32.2, MLX-LM
  0.31.4.dev132+g94cdcae13, Recipe 0.2.0; macOS 26.5.2, M3 Ultra/512 GB.
- Actual zero-weight BF16 `[1,2,1,128]` `pre_woa` materialized, finite, same
  dtype/shape; result float32 digest
  `185d7692ae717f721a3e33ed4661b857c3582ca9ad4eb01691c8415c05bb7d99`.
- Early-admission characterization also executes actual `pre_woa` after the
  imports, preserving the authenticated identity/function objects and profile.
- [Preflight](phase2-step2dr1-preflight.json) proves the unchanged two-request
  projection template, N=202 / eligible C=200, real image sources, normal workload
  envelope and preserved installed Scheduler/internal/identity module objects.
  The real worker repeats this gate with full checkpoint admission before the
  Supervisor authorizes loading. The placeholder projection is never submitted.

[Identities](phase2-step2dr1-identities.json) retain installed-vs-checkout hashes
for seven unchanged execution owners, exact diagnostic hashes, canonical token
digests, manifest and wheel provenance. Runtime wheel SHA256:
`b60c0087dbb9f6226851ff37598d3e0d932babef611acd5da2c8a7efedf95bce`.
[Provision log](phase2-step2dr1-provision.log) records the normal pinned exports,
ordered overlays and retained native assets. No mutable execution overlay.

## 4. Actual native two-request sequence

[Observation](phase2-step2dr1-observation.json), [run log](phase2-step2dr1-run.log).
Both requests use the same Production Scheduler and paired authority, greedy
sampling and one output-token budget. No tools or external effects are executed.

| Boundary | Request 1: seed | Request 2: partial-prefix |
| --- | --- | --- |
| Actual Recipe canonical input | N=202; full IDs retained | N=662; full IDs retained |
| Paired lookup | Genuine C=0 miss | Genuine **C=203** hit, restore time 0.001022 s; 459-token suffix |
| Historical classification | Corn false | Corn true, carrots true; no relabeling |
| Vision | One corn occurrence | One carrots occurrence; no corn encoding |
| P6 schedule | 0→200→201 | **203→660→661** |
| Pending N-2 capture | Target40/DSpark3 at 200 | Target40/DSpark3 at 660 |
| Registered post-P5 native state | Exact 202 prompt IDs; 40/3 at 202 | Exact 662 prompt IDs; 40/3 at 662 |
| UID/native owner | UID 0, same Scheduler batch and both maps | UID 0 in the next retired-owner lifetime, same Scheduler batch and both maps |
| Actual native generation | Token **2337**, text **This** | Token **671**, text **The** |
| Messages completion | `max_tokens`, output usage 1, `message_stop` | `max_tokens`, output usage 1, `message_stop` |
| Canonical Settlement | Target40/DSpark3 at **203** | Target40/DSpark3 at **663** |
| Complete paired publication | **200, 203** | Existing pairs retained; new **660, 663** |
| Retirement | Native closed/quiesced; UID/batch null; requests/maps/sources empty | Same; no fatal engine error |

The first actual assistant content `[{'type':'text','text':'This'}]` is passed
verbatim into Request 2, between the identical first corn user message and the
unchanged carrots inspection/current-feature user messages. No synthetic reply,
conversation variant or alternative generation was substituted. Actual carrots
position and whole-image-safe overlap were recalculated before enqueue.

The actual selected C=203 is the longer **settled generated pair**, not merely the
preflight's eligible prompt C=200. A read-only normal authority acquisition before
Request 2 compares all **283** present Target/ring arrays exactly to that retained
pair, including shape/dtype, metadata and independent cache/ring containers.
Tokens and committed corn identity match exactly. Its SHA256 over physical state
is unchanged after suffix execution. The production lookup itself is unwrapped,
selects the same C=203 and reports it in both trace and Recipe cached usage. No
checkpoint was constructed, inserted, patched or manually published by diagnostics.

P6 plan fields are the existing algorithm's selected boundaries, not an alternate
executor. N-2 is observed after registration; N-1 completing append is certified
by the unchanged genuine-append/P5 path reaching the registered post-terminal N
state, not separately sampled between native operations. No native insert/decode,
generation, settlement or publication wrapper was introduced. Optional registered
observations are non-throwing; mandatory assertions run after retirement. Error
cleanup preserves the first failure (none occurred).

## 5. Covered/fresh images and physical accounting

| Occurrence | Actual span | Encoded-byte SHA256 | Work |
| --- | --- | --- | --- |
| Corn | `[2,191)` | `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1` | Grid 33×47, **1,551 covered patches**, **138,600 covered pixels** at C=203; zero second-request encode |
| Carrots | `[205,649)` | `5df896a4a07e127281c60fc957f8b3d73f4735b3258a0bf762b4383557f8fa9a` | Grid 51×74, **3,774 fresh patches**, **717,824 fresh pixels**; one complete absolute-position occurrence |

Total projected workload is **5,325 patches**, within the ordinary **36,864**
envelope. Current sources are authenticated before native start; only uncovered
carrots is fresh work. Existing `encode_new`/P6 alignment and the production
`vision_encoded_count=1` trace establish whole carrots consumption, without
retaining embeddings as a second cache owner. The aggregate historical-discount
callback and its 12 MiB / 192 MiB qualifications are unchanged and not generalized.

Both native runs have one init emission, three proposals, no verification cycles
under the one-token budget. This proves bounded native execution, not extended
speculative acceptance rates or visual-answer quality. Both report replay=0,
repack=0, drained queues, retired predictions and settled authority connections.
Quiescence introduces no new verify/proposal/forward/replay/repack/DSpark work.

## 6. Resources, cleanup and operational integrity

Reused the existing [Supervisor](phase2-step2dr1-supervisor.py.txt), changing only
task paths. [Resources](phase2-step2dr1-resources.json) and
[Supervisor log](phase2-step2dr1-supervisor.log):

- Worker **24272**, Supervisor **24269**, both exit 0 and absent postflight.
- One model load; no reload/retry or competing inference worker. Supervisor
  samples directly; no auxiliary Python statistics process during inference.
- Preload available **493.916 GiB** ≥455.226. Minimum host reserve
  **198.504 GiB** >64. Pressure **1** throughout, swap used/swapouts **0**, no growth.
- Sampled lifetime physical-footprint peak **300.595 GiB**, below 425 GiB stop
  and 430 GiB absolute ceiling. Samples do not prove every unsampled transient.
- Total **375.24 seconds**, including ~292 seconds pre-load gate; no 10-minute
  no-progress interval or 30-minute timeout; no resource stop/unrelated kill.
- Loaded MLX active/cache/peak: **309151331768 / 5590977820 / 311557023684 bytes**.
  After both requests: active/cache **309168129394 / 12464316 bytes**, same peak.
  Host and MLX measures overlap; they are not additive.
- Native/request retirement is explicit in both receipts. Scheduler closes and
  clears authority, model closes, executor shuts down, references/allocator cache
  release. Model weakref dead; **1,064 active bytes / zero cache** before exit;
  numerical profile remains valid. Disposable installation/build/handshake files
  removed after retaining identities. No model or task worker remains.

[Operational integrity](phase2-step2dr1-integrity.json) reuses the exact existing
**817-path** inventory. Before/after hash/size/mtime/inode and the retained
Step 2B-R2 baseline all match. Original checkpoint=False inspection identity
`6234d1e8a59f98cfb6e9f7d7e9de050b1c72973421ac94774da665ed88493681`
is unchanged, recorded separately from the disposable full-model admission.
No original weight load, installation modification or claimed identity equivalence.

## 7. Verification and disposition

[Focused suite](phase2-step2dr1-tests.log): **46 passed**, 261.39 seconds, using the
new coherent disposable installation. Includes exact import-order subprocesses,
actual zero-weight worker preflight, unadmitted numerical refusal, C>0 paired
selection and native-interface lifecycle controls, cancellation, mutation/failed
handoff and incomplete settlement/burn fences. Source-test numerical bindings do
not supply admission to the isolated subprocesses. Reused Step 2D's prior **315
passed / 26 subtests** as historical unaffected evidence, not rerun or relabeled.

Postflight, the zero-weight worker regression redirects only its diagnostic receipt
directory to a temporary path, so future tests cannot overwrite retained native
evidence. A first check using the older Phase 1 installation refused because that
installed `MultimodalInput` predates `validate_sources`
([exact refusal](phase2-step2dr1-output-isolation-refusal.log)); no weights loaded.
The final isolated-output check uses a fresh coherent official `mtp_setup`
installation at `/tmp/ds41f-step2dr1-postflight-tests`: **1 passed**
([log](phase2-step2dr1-postflight-tests.log)). This changes only the test, not the
executed worker or any Runtime module. That test-only installation/build was
removed. There was no model reload or generation retry.

Final `git diff --check` and `python tools/check_repository_self_containment.py`
pass. Diff review finds no Scheduler, P6/P5, numerical, admission, cache ownership,
protocol or settlement implementation change. Historical reports are unchanged;
CURRENT summaries add only this bounded undeployed evidence.

The Step 2D installed gate blocker is closed for this diagnostic sequence. The
smallest remaining C2 gap is **case-specific qualification beyond this preserved
whole-image-safe prefix**: changed/reordered/shortened projections and broader
reconstruction cases remain unqualified. True image-intersecting execution and
broader conditional-discount reconstruction are not enabled. Prior MTP/Ordinary
and cross-partition numerical questions remain unresolved; no reference rerun,
`wkv` investigation, fitted tolerance or bitwise cold-append claim. Harness
offload/compaction/reacquisition and broader resource/lifecycle qualification remain
separate. C2 is PARTIAL, not globally PASS. Stop after Step 2D-R1.
