# Phase 2 / Step 1 — ordinary numerical prefill reference

**Disposition: BLOCKED for real-model qualification.** Source-level construction
and model-free observation tests are available; the numerical reference is **not
qualified**. No historical-image reconstruction, C2 completion, guard change,
release admission or production behavior change is implied.

## 1. Preflight and exact identities

Started on `Multimodal-Runtime`, HEAD
`43728c1cf68958345f0ac821f67250858efbd3b5` (the expected HEAD), clean tree, one
worktree. One coding agent (`pi`, PID 2432) was observed, no other coding agent or
Python inference worker. This is process inspection, not a repository/service
lock. No concurrent tracked edits were observed during this task.

The broader listener inspection found **LM Studio PID 1039 listening on
`*:1234`**, with its connector PID 1248. Read-only `lms ps --json` returned `[]`;
`GET /api/v1/models` confirmed no loaded instances, but advertised
`glm-5.3-mlx-mixed-4`, **427,767,580,950 bytes**. Thus there was no observed resident
LLM, but there was a live serving endpoint capable of receiving work. A task-
exclusive maintenance window/exclusion of an on-demand load was not established.
A concurrent load of that model cannot safely coexist with the prior Flash
359.226 GiB footprint observation on this 512 GiB host. That historical footprint
is not a current allocation guarantee. **Do not start a model beside this service
on the strength of an idle snapshot.** No service was stopped, disabled or changed;
no generation request was made to it. This is a conservative concurrency safety
blocker, not evidence of current memory exhaustion or a model numerical defect.

Preflight host: Apple M3 Ultra, 549,755,813,888 bytes physical memory;
`memory_pressure` reported 98% free, 29,228,724 free 16-KiB pages at the retained
sample; swap allocated/used zero, swapins/swapouts zero. Free temporary storage:
789 GiB on the system volume, 190 GiB on the repository volume. These are
point-in-time observations; they do not reserve memory against serving requests.

[Machine-readable preflight](phase2-reference-preflight.json) retains package
origins, source hashes, checkpoint metadata, fixture identity, serving queries and
resource samples. Relevant admitted existing isolated environment:

| Item | Observed identity |
| --- | --- |
| Interpreter | `/Volumes/SDXC-512/ds41f-1017/bin/python`, Python 3.13.15 |
| Installed ds41f | 0.23.0; build source `aa8c5717b42aa0a216fef3134e72c6db7ca49287` |
| oMLX | 0.7.0, installed from `/Volumes/SDXC-512/ds41f-1017-build/omlx` |
| MLX / mlx-lm | 0.32.2 / `0.31.4.dev132+g94cdcae13` |
| Recipe | 0.2.0, native source `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` + repository native semantic patch |
| Numerical selection | `flash-1017-pre-woa-fp8-v1`, `glm_profile=native` |
| Installed identity SHA256 | `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2` |
| Installed compatibility inspection | PASS, with explicit checkpoint environment variable; **no weights loaded** |

The installed identity retains its seal-era `mtp-singleton-v1` profile field;
inspection was requested for `mtp-serving-v1`. This is not ordinary capacity
qualification. Checkout tooling/production sources are not equated to installed
0.23.0 sources. The test fixture authenticates the installed dependency closure
and selects its immutable numerical variant while testing checkout code.

Development `.venv` exists (Python 3.13.14, MLX 0.32.2, oMLX 0.7.0), but has Recipe
0.1.1 and **no installed ds41f payload**. It is not the numerical execution
environment. The existing 1017 environment was reused without installation,
wheel replacement, dependency rebuild or duplicate environment creation.

Checkpoint, read-only access only:
`/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`.
All 48 declared shards were readable. Existing installed admission succeeded;
its full-byte verification may use its unchanged-file-identity cache. No claim of
a fresh independent rehash of every weight byte is made. Shard source revision:
`1bae0e1088993b9b82de08f59de5be5a34e42170`. SHA256:

- config: `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- index: `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- tokenizer: `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`

Repository-pinned oMLX source archive SHA256:
`26cc224a5fa77d8576589764a56ba8053ac31fe9f4904d8ba60305ed841f33e6`, upstream
base `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, exported candidate
`fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`. Ordered compressed-growth,
bit-exact-kernels, template-reuse and context-copy overlays remain part of the
installed provenance; their repository hashes are retained. Source-archive
identity alone is not a claim that every installed file equals the archive.

## 2. Inspected owners and reference construction

Read the governing plan, Phase 1 audit (including §§3/5/6/8), ordinary serving,
Vision integration, architecture, correctness, qualification, production prefill
substrate and P3/P4 numerical qualification documentation. Inspected the requested
four tools and `tests/test_production_mtp.py`, plus current paired, P6, lifecycle
and multimodal owners. Actual installed oMLX owners are under
`/Volumes/SDXC-512/ds41f-1017/lib/python3.13/site-packages/`:

| Owner | Interface / numerical role |
| --- | --- |
| `omlx/patches/deepseek_v41/model.py` | `Model.get_input_embeddings`, ViT/aligner/delimiters; SHA256 `77c7012b5ce2aa757157b350eb38e1a3694c0b3fb7730ce1be2ac4fe99b43a24` |
| `.../language.py` | `LanguageModel._forward`, ordinary 40-layer loop, image-aware Engram, post-Engram/pre-layer mean taps; SHA256 `702b8a1c872eee645e6633140787a5947dc4bb7aace16384475a90151a4c5276` |
| `.../mtp.py` | `dspark_append_context`, shared projection/norm and three stage attention append operations |
| `omlx/patches/mlx_lm_mtp/deepseek_v4_dspark.py` | `DSparkContextCache.append`, absolute-modulo physical rings; `install/validate/take_committed_context` |
| `.../cache.py`, `growth.py` | Seven Target slots, metadata, logical slices/backing growth, extract/merge padding |
| `.../quantization.py` | Existing `unpack_activation` for bounded selected-row diagnostics, not cache reconstruction |
| Checkout `serving/internal_mtp.py::_start` | Candidate taps, P6 segments, N-2 pending capture, N-1 seal, P5 |
| Checkout `serving/production_mtp.py` | Candidate image adapter, bounded tail selection, Scheduler admission |
| Checkout `serving/paired_checkpoint.py` | Existing 40/3 structural validation and immutable snapshot/restore contract |

Candidate:

```text
Recipe canonical preparation -> Scheduler paired lookup/fresh cache
 -> encode_new -> pinned Model.get_input_embeddings (per new image span)
 -> DeferredPrefillAppend / P6 -> candidate layer taps
 -> native dspark_append_context -> N-2 capture -> N-1 seal
 -> existing P5/native BatchGenerator -> canonical settlement
```

Reference (test-only helper, no production selector):

```text
same canonical IDs + independently prepared current source pixels/metadata
 -> Model.get_input_embeddings (recomputed; never candidate sparse embeddings)
 -> LanguageModel._forward(cache=make_cache(), return_dspark_hidden=True)
 -> dspark_append_context(hidden, make_mtp_cache(), start_offset=0)
 -> observations of Target/rings/logits/image rows
```

`tools/qualify_prefill_fp8_mlx.py::ordinary_prefill_reference` now invokes these
existing APIs directly, accepts no candidate cache or embeddings, enforces fresh
40/3 factories and a bounded singleton input. It does not call paired lookup,
restore, P6, `_omlx_prefill`, CED, DSpark implicit priming, sampling or settlement.
No second model loader, inference implementation, queue or cache authority was
added. This helper is not selected by that tool's historical P3/P4 CLI. The caller
must first admit model/profile/resources, restore any temporary candidate layer
wrappers, and own evaluation and cleanup. Real invocation remains unexecuted.

**Shared numerical components, not independent end-to-end proof:** checkpoint,
tokenizer/Recipe encoding, CPU image recipe, ViT/aligner/delimiter operations,
MLX/Metal primitives, loaded layer modules and quantization/routing/attention
operations, SSD Engram data and native DSpark projection/append/ring operations.
The independent boundary is **fresh Target state and ordinary layer-loop dataflow
versus candidate scheduling**, not independent implementations of these kernels.
Reference taps must be recomputed, not borrowed from candidate Target hidden
state. Shared DSpark append can establish integration consistency of separately
computed hidden streams, not the correctness of DSpark append itself. Sharing
prepared pixels is an isolation diagnostic only; baseline image preparation and
embeddings must each be recomputed and compared.

## 3. Canonical observations and full state design

Freeze exact Recipe request/rendering and IDs (ordered little-endian int32 digest
plus the short IDs themselves), absolute positions, numerical identity and source
origins. For each occurrence record source byte SHA256, occurrence index,
`(start,length,grid,kinds)`, expanded IDs and meaningful masks. Independently
prepare patches from those same bytes; compare pixels/layout exactly. Compare
embeddings including delimiters at absolute positions. Candidate's per-span adapter
and reference's full-prompt adapter may share operations but not their produced
embedding arrays. No model input IDs/expanded spans or embedding values were
measured in this task; the cases below are **not run**.

`paired_physical_metrics` traverses **every layer 0..39 and every slot 0..6**, plus
`left_padding`, `lengths`, `meta_state`, compression ratio and logical offsets.
It traverses rings **0,1,2**, records capacity/offset and absolute positions in
physical-slot order, and compares keys without rotation. Native rings have one
`keys` tensor used as K and V; there is no separate persistent `values` field.
Diagnostic output is `OBSERVED_ONLY`, never PASS.

| Target slot | State / required meaningful interpretation |
| --- | --- |
| 0 | Exact integer consumed offset, plus layer `size()` |
| 1 | Packed local window KV; chronological retained positions `[max(0,C-window),C)`, not DSpark modulo ordering |
| 2 | Packed compressed KV; source-layer rows through `floor(C/ratio)`, respecting actual source layout |
| 3 | Packed index K; source/index ownership and complete published rows; not transient query top-k candidates |
| 4,5 | Incomplete compressor KV/gate tails, `C % ratio` for ratio >1; otherwise applicable empty state |
| 6 | Exact Engram lookback (layer 0), applicable empty histories elsewhere, including image masking semantics |

Compare full shape/dtype and presence, even for empty arrays. Keep unused padding
separate from meaningful values; do not trim an unexplained layout mismatch.
Growth backing capacity is not automatically meaningful logical state. Derive
meaningful selectors from the admitted cache/config, not the observed discrepancy.
The new physical collector deliberately **does not yet implement semantic masks
or decode packed values**: packed byte mismatch counts are not numerical KV error.
Next collector extension should slice rows before existing `unpack_activation`:
slot 1 `(bits=8,group_size=32)`, slot 2 `(4,16,e4m3_scale=True)`, slot 3 `(4,32)`;
record those shared diagnostic-decoder semantics. Never unpack an entire persistent
cache or compare integer encodings as if they were floating magnitudes.

At C, ring meaningful positions are `[max(0,C-128),C)`, stored at absolute position
modulo 128 once full. Physical slot-to-absolute-position maps must match exactly.
No chronological sorting, circular alignment search or rotation normalization is
allowed. Capacity is 128; keys shape is `[1,1,min(C,128),head_dim]` in this topology.

Comparable observation boundaries:

1. **N-2 pending capture:** observe candidate through existing capture callback;
   fresh ordinary reference runs that exact canonical prefix. Compare live and
   immutable snapshot/restore separately; do not publish diagnostic state.
2. **N-1/P6 seal:** observe immediately before P5; continue ordinary reference by
   the same canonical text row, with native append at its old absolute offset.
   Independent fresh-current-prefix reference is also required for append case C.
3. **Cold completion / P5 terminal N:** ordinary reference consumes terminal once;
   candidate observes existing `OMLXMTPGenerationSession.start` bootstrap. Inspect
   native pending-main/priming ownership: do not assume rings already at N when
   Target reaches N. `take_primed` explicitly permits the prior ring frontier.
   Compare only matched consumed Target and committed ring frontiers, or the
   existing canonically quiesced complete pair. Never append/replay/settle candidate
   state merely for observation. Test-only P5 observation adapter remains unwired.
4. **Short continuation:** use one fixed teacher-forced canonical suffix, record
   matched-position logits before consuming each next ID. Compare committed state
   at matched frontiers; proposal/verify stash and sampled lookahead are not paired
   truth. Wiring a teacher-forced candidate control through existing interfaces
   (without replacing native scheduling) remains unresolved. Greedy text alone
   cannot substitute for this comparison.

Exact physical snapshot equality and cross-schedule numerical equivalence are
different assertions. The collector retains complete physical diagnostics, but
cannot presently certify all meaningful cross-schedule state or these boundaries.

## 4. Acceptance policy — fixed before any future model observations

**Exact:** canonical IDs and hashes, bytes/occurrence order/spans/grids/kinds,
checkpoint/profile identity, masks, presence/shape/dtype/layout, integer offsets,
histories, metadata and ring indexing. Same-operation immutable capture/restore
requires exact value equality and independent mutable containers, including
wrapped rings. No float tolerance may excuse these structural mismatches.

**Floating:** measure maximum absolute error, RMS-relative error
`sqrt(sum((a-b)^2)/sum(b^2))`, cosine when both norms are nonzero, and matched-
position full-logit errors. Nonfinite pairs are explicit unresolved failures to
compare; zero reference norm gives undefined RMS-relative/cosine, not a fabricated
perfect score (use absolute error). The metric collector reduces in bounded
65,536-element chunks with FP64 host accumulators and at most four mismatch
samples; BF16 converts only the current chunk to FP32 before host transfer.
It retains no complete host-state duplicate. Candidate/reference live state must
still coexist within the declared resource budget. Ordinary `_forward` computes
full call-local logits; this helper does not introduce a logit-suppression kernel.

**No defensible general Target/DSpark floating bound is established yet.** Existing
P3/P4 and P6 controls prove exact packed equality only for selected same-input,
same-operation source/index/window checks (P6 matrix primarily layer 20 slots
2/3). They do not calibrate a 40-layer different-dataflow trajectory. Existing
Vision Torch/MLX thresholds remain Vision-only and are not imported. HTTP Vision
acceptance proves integration/frontiers, not all paired numerical values.

Calibration order: unchanged accepted cold text, cold image and supported append
versus ordinary fresh-current-prefix reference; separate same-operation exact
controls from different schedules and packed logical-value comparisons. Retain
per-layer/per-slot/per-ring/logit maxima, RMS-relative and cosine, independent
structural outcomes and first differing meaningful rows. Investigate deviations
against operator/profile semantics before defining an envelope. These three small
controls may still be insufficient to justify an envelope: then retain PARTIAL,
do not widen a bound to fit results. This report establishes **no numeric tolerance
value** for future reconstruction and no automatic numerical acceptance decision.

## 5. Finite execution plan (not activated)

Only after service concurrency is resolved without interfering with production,
repeat safety preflight and installed admission. Reuse **one** model load from the
existing 1017 environment. Candidate loader settings resolved from
`InternalMTPQualificationBackend.load`: pinned `loading.load(checkpoint,
preserve_mtp=True, engram_ssd_offload=True)`; retain admitted full-backbone/native
profile and existing `MTPWiredLimitLease`/generation stream. Do not copy historical
P3/P4 runners' `preserve_mtp=False` or original-Flash checkpoint defaults. Those
runners are not drop-in Flash-1017 paired controls.

Budget: only A/B/C; each at most 4096 expanded positions and one image; at most
8 fixed teacher-forced continuation positions per case (never more than 32 output
positions). At most one model-loading process; 30-minute total wall budget,
10-minute no-progress stop, no sweeps/endurance/retry campaign. A: Recipe text
`Explain immutable checkpoints in one short sentence.` B: existing `corn.jpeg`
with `Identify this food in one short sentence.` C: accepted compatible text
prefix from A, append `Then state one benefit.`; bind actual restored C and compare
against fresh complete current-prefix reference, not candidate's restored state.
Recipe owns actual framing/IDs; do not handcraft them. B fixture SHA256:
`251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1`.
Actual expanded IDs/grids and teacher-forced suffix IDs must be frozen before run.
These short cases cannot qualify the deferred bulk branch: current
`P6_FINAL_TAIL_THRESHOLD` is 8192. No larger case is authorized by this task.

Proposed conservative operational stop conditions: do not start without exclusive
model-load availability and at least 96 GiB pre-load available host memory above
the historical 359.226 GiB footprint; stop on any new inference loader, non-normal
host pressure, available host reserve below 64 GiB, any swap-use increase or
swapout delta, model error/cancellation, 430 GiB task physical footprint, or budget
exhaustion. These are task stop conditions, not production resource-limit changes
or a measured capacity guarantee. Monitor host/process state externally; a
post-segment check alone cannot catch a stuck GPU command. On stop, cancel only
this task's process, preserve partial evidence, close owned model/resources, wait
for task worker exit, and verify no task-owned loader remains. No arbitrary retry
or parameter tuning after mismatch.

Before launching, finish semantic packed/mask comparisons, P5 and teacher-forced
observation wiring and supervisor-compatible settings. The helper alone is not a
ready-to-run qualification campaign.

## 6. Results, deviations and resources

| Case | Real Target numerical coverage | Real DSpark numerical coverage | Continuation |
| --- | --- | --- | --- |
| A cold text | **0/40**, all layers unexecuted | **0/3**, all rings unexecuted | NOT RUN |
| B cold image | **0/40**, all layers unexecuted | **0/3**, all rings unexecuted | NOT RUN |
| C supported append | **0/40**, all layers unexecuted | **0/3**, all rings unexecuted | NOT RUN |

Model-free collector regression traverses **40 x 7 slots / 3 rings**, catches a
last-layer last-slot mutation and a rotated ring 2, and records physical wrapped
positions without normalization. It covers zero/empty/scalar/nonfinite metrics,
chunked reductions, structure/dtype rejection, actual MLX BF16 conversion,
reference API selection, fresh factories and refusal of precomputed Target or
embedding input. This is synthetic coverage, **not any real-layer numerical
proof**. Production snapshot regressions also exercise native MLX arrays/wrapped
rings without loading weights.

Two tooling/environment deviations were preserved: development `.venv` failed
all 57 test setups because its isolated interpreter cannot import installed
ds41f; use of the existing admitted environment resolved that without fixture
bypass. A new BF16 collector test exposed MLX dtype string prefix handling; the
collector was corrected, not any model operation or tolerance. See the compact
[initial setup log](phase2-reference-tests.log),
[collector correction](phase2-reference-collector-correction.log) and
[final test log](phase2-reference-installed-tests.log).

There was **no real-model numerical mismatch**, because no comparable model
execution took place. Baseline floating tolerances, semantic packed comparisons
and handoff/teacher-forced observations remain incomplete independently of the
resource blocker. Their likely owners are the existing verification collectors
and native lifecycle observation adapters, not a new inference/cache engine.

Real-model MLX active/cached memory, allocator peaks, process-lifetime physical
high-water, cancellation and loaded-model cleanup: **NOT MEASURED**. Preflight
host values are not substitutes. No model was loaded or cancelled; no task-owned
inference worker, weights or model-state dump was retained. Later runs must record
MLX active/cache/peak and host physical footprint separately (not sum overlapping
metrics), plus swap/pressure, error outcome and post-close retention. Allocator-
retained storage alone is not a leak. Installed tests use small synthetic MLX
arrays; their process exits do not qualify model cleanup.

## 7. Validation and stopping boundary

Final focused command uses the existing environment, no added dependencies:

```sh
DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
 /Volumes/SDXC-512/ds41f-1017/bin/python -m pytest -q \
 tests/test_prefill_numerical_reference.py tests/test_prefill_fp8_mlx_p6.py \
 tests/test_production_mtp.py
```

- Focused tests: **58 passed, 12 subtests passed**, 41 existing dependency/API
  deprecation warnings, 38.86 s; no weights loaded.
- `git diff --check`: PASS (including staged review before commit).
- `PATH="$PWD/.venv/bin:$PATH" python tools/check_repository_self_containment.py`:
  PASS; no dependency installation.
- Diff review: only this report/compact evidence and test-only tools/tests;
  no production numerical or serving source changed. One focused local commit;
  its SHA is reported in the task response (a commit cannot embed its own SHA).

Remaining gates before even considering historical coverage changes: establish a
safe exclusive load window; execute A/B/C with independently recomputed input and
Target streams; cover meaningful state on all 40/3 at matched boundaries; resolve
packed-value/mask and P5/native-pending observations; establish defensible scoped
floating criteria and matched-position continuation; retain actual model memory,
stop/error and cleanup evidence. Shared DSpark math remains an independence
limitation even after these gates. Changed-history reconstruction requires its
own subsequent approval and cases; a baseline PASS would not complete C2.

No historical-image coverage guard, patch admission, production Scheduler/cache,
settlement, kernel, installed payload, official Harness/Recipe semantics or default
profile was changed. The 36,864 fresh-patch, 12 MiB projected-pair and 192 MiB
retained-root guards are untouched. CURRENT plan/progress remains unchanged.
Stop here; no push, deployment or follow-on implementation.
