# Phase 2 / Step 1 — ordinary numerical prefill reference

**Latest bounded result: [§12](#12-step-1c--compatible-restored-prefix-append-2026-10-25)
is OBSERVED for restored-prefix Case C at N-2 (40/40 Target layers, 3/3 rings),
with floating differences.** Cases A/B observed exact agreement in §§10–11.
Earlier BLOCKED reports below remain historical evidence. General numerical
acceptance, full Step 1, Phase 2 and C2 remain unqualified.

**Initial disposition (historical): BLOCKED for real-model qualification.** Source-level construction
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

## 8. Continuation — 2026-10-25 (HEAD ff0daa3)

**Disposition: BLOCKED; Step 1 remains incomplete.** The earlier LM Studio
observations above are historical and unchanged. Its termination resolves the
previously observed serving conflict, but this continuation does **not** complete
the required execution harness. No weights were loaded. This is a remaining
verification-tool prerequisite, not a demonstrated model defect or a claim that
the environment is still blocked by LM Studio.

### New preflight and identities

Started on required `Multimodal-Runtime`, exact expected HEAD
`ff0daa397c93664aa8dfd39638363fc7dea4971b`, clean working tree. Read-only process
inspection found no LM Studio application, connector, llmster, Python inference
worker, Ollama or other observed external inference loader. One coding agent,
`pi` PID 2432, remained. `lsof` found **no listener on port 1234**; the complete
listener list contained only rapportd, ControlCenter (5000/7000) and GitHub.
`launchctl list` had no matching inference service; the user/system LaunchAgents
and LaunchDaemons directories contained Google/Chrome Remote Desktop and Docker
network/socket helpers, not a discovered inference launcher. No Docker VM/model
worker was observed. No `lms` command was needed or invoked. No service or
unrelated workload was stopped, reconfigured or restarted. These inspections are
point-in-time observations, not an enforceable system-wide model-loading lock.

[Continuation preflight](phase2-reference-continuation-preflight.json) records a
new **isolated installed** `mtp_identity.inspect(load_runtime_config())` PASS,
without model loading. Interpreter and dependency versions, installed source
hashes, installed build provenance and identity digest all match §1. The 48-shard
checkpoint admission and metadata remain available; corn fixture was rehashed and
matches `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1`.
Checkpoint verification retains the installed unchanged-file cache caveat; no
claim of independently rehashing every shard. No installed asset was modified.

Checkout tests are **not** installed Runtime verification: pytest authenticates
the installed dependency payload in an isolated subprocess, selects its admitted
variant, then imports the checkout helper/Runtime. Installed ds41f remains 0.23.0,
source `aa8c5717b42aa0a216fef3134e72c6db7ca49287`; the test changes are not installed.
The archive ordinary layer-loop hash is
`9377537e16b5978ed3aeddc998f772d8e27546d2444db95454df760f74572c43`, different from
installed `702b8a1c...`: inspected diff contains compressed-growth append/reuse and
admitted pre-WO-A activation selection. The native DSpark archive/installed source
hashes agree. Archive equality is not substituted for installed identity.

Host sample: 512 GiB physical; 98% free from `memory_pressure`; about 446.6 GiB
free pages and 47.3 GiB inactive/speculative pages (reclaimability is not reserved
memory). Swap use, swapins and swapouts remain zero. Temporary/repository/model
storage available: 789 GiB / 190 GiB / 1.1 TiB. The conservative load condition
remains **359.226 + 96 = 455.226 GiB available**, not an RSS allowance. Free pages
alone are below that figure; an actual launch must freshly establish available
reserve, including defensible reclaimable accounting. No load-admission or
capacity guarantee is inferred from this idle sample. All §5 stop thresholds
remain unchanged; no supervised real-model execution window was activated.

### Collector improvements and model-free coverage

Extended the existing helper in `tools/qualify_prefill_fp8_mlx.py`; did not create
another reference implementation or change its historical CLI loader settings.
`ordinary_prefill_reference` still independently constructs fresh ordinary
40/3 state and now returns consumed token IDs for boundary identity checks.

- `paired_physical_metrics(..., config=...)` retains all physical diagnostics and
  adds meaningful selectors/numerical chunks for all 40 layers and seven slots.
  Selectors derive from admitted config and committed offsets, not discrepancy.
  Source ownership follows `make_cache`/ordinary Attention/Indexer. Slot 1 retains
  chronological `[max(0,C-window),C)`; source compressed/index rows use `C//ratio`;
  incomplete compressor/gate tails use `C%ratio`. Non-source slots have zero
  meaningful rows. Slot 6 compares complete exact integer Engram history,
  including sentinels, and applicable empty histories on other layers.
- Verified installed `quantization.py`, `language.py` and `cache.py`: slot 1 is
  8-bit/group-32; slot 2 is 4-bit/group-16/E4M3-scale; slot 3 is 4-bit/group-32.
  Existing `unpack_activation` decodes only selected slices of **at most 16 rows**,
  FP32 diagnostic output. No full decoded persistent-cache duplicate is created.
  Metrics are retained per bounded meaningful chunk, not a fabricated aggregate
  acceptance tolerance. Physical packed-byte differences remain separate.
- Explicit singleton offset, packing width/dtype, history layout and padding
  validation refuses unknown layouts. Physical shapes, metadata, lengths/padding,
  presence and integer values remain exact diagnostics, including padding
  differences; numerical meaningful-row metrics do not erase those differences.
- All three rings retain exact capacity/offset/mapping diagnostics, with committed
  tensor-layout checks and numerical keys metrics in original physical order.
  No ring is rotated or normalized to improve agreement.
- `paired_boundary_metrics` refuses unequal canonical consumed IDs/frontiers and
  unaligned layers/rings. For P5 it permits separately matched Target and ring
  frontiers, e.g. Target N versus independently recomputed Target N, rings N-1
  versus independently recomputed rings N-1. It never settles or advances state.
  It is a boundary **guard**, not a wired native lifecycle observation adapter.
- Added native-small-array tests for padding exclusion, last meaningful packed
  index row in layer 39, bounded decoder calls, source/non-source/tail masks,
  malformed layouts, ring-2 rotation/mapping/layout negatives, transient P5
  frontiers and wrong consumed identity. Tests verify array identity/offsets remain
  unchanged by observation. Existing freshness/independence tests remain intact.
  Nonfinite diagnostics now distinguish all nonfinite pairs from mismatched
  nonfinite values; nonfinite numerical metrics still remain undefined.

This is synthetic **40 x 7 / 3** collector traversal, not real-model execution.

### Native lifecycle and continuation limitations

Inspected `_start` and existing P6 matched-control collectors again. The existing
`checkpoint_capture` callback identifies N-2; P6 committed certificate and
`handoff_to_generation` identify N-1; `batch_generator_factory` and native
`OMLXMTPGenerationSession.start` provide the P5 construction/bootstrap seams.
These are reusable without a production API change. The new guard is **not yet
connected to those seams in an A/B/C runner**. In particular, no captured native
P5 Target/ring pair was compared here, and no independently recomputed image
embedding/canonical Recipe case adapter was executed. Reading lifecycle source
or testing synthetic boundary dictionaries is not completion of collector C.

`OMLXMTPGenerationSession.next_token` exposes `bound_observation` with lifetime,
revision, consumed prefix, token and original native **logprobs**, not an explicit
raw-logit/teacher-force transaction. Its sampler is also used by native proposal,
verification and bonus paths, including multirow calls; replacing it with a
call-count-based fixed-token iterator would not establish matched absolute
positions. Configured logits processors alter the distribution and are not a
validated passive raw-logit observation seam. No ready-made deterministic
teacher-forcing control with matched committed frontiers was identified in the
inspected lifecycle interfaces/tests. This is **not proof that it requires a
production scheduler change**: a position-bound diagnostic adapter may be possible,
but it has not been implemented/validated in this task. No such production change
was attempted. Teacher-forced continuation remains PARTIAL/unimplemented.

The missing wired case/lifecycle observations and external resource supervisor
are pre-load prerequisites from §5. Rather than load the 359-GiB-class model for
an incomplete or misleading comparison, execution was not started. This leaves
the requested task incomplete even though the previous service conflict is gone.
No automatic retry, historical-image reconstruction or follow-on work occurred.

### Frozen policy, actual results and remaining gates

The **pre-observation §4 policy is unchanged**: exact structural/metadata/mask and
same-operation snapshot equality; numerical measurement for different schedules;
independent calibration before accepting Target/ring/logit tolerances. No
Vision-only threshold was imported and no threshold selected from these tests.
Max-absolute/RMS-relative/cosine/nonfinite meaningful-chunk and per-ring metrics
are available; **no real numerical error envelope or matched-position logit
difference was measured**. A/B/C might remain PARTIAL even after execution if
independent calibration is insufficient. Shared checkpoint/operators, Vision,
quantization and native DSpark math retain all §2 independence limitations.

| Case | Actual model execution | Target coverage | DSpark coverage | P5 / continuation |
| --- | --- | --- | --- | --- |
| A cold text | NOT RUN | **0/40** | **0/3** | NOT RUN |
| B corn single image | NOT RUN; fixture hash verified | **0/40** | **0/3** | NOT RUN |
| C supported prefix append | NOT RUN; no restored frontier | **0/40** | **0/3** | NOT RUN |

No model allocator active/cache/peak, physical-footprint peak or loaded-model
cleanup measurements exist for this continuation. Model-free test processes
exited; no task-owned inference loader was created. No weight/tensor dumps,
duplicate environments or checkpoint/installed-payload changes were made.
Future launch must repeat exclusivity/memory inspection and wire external
supervision with the unchanged reserve/swap/pressure/430-GiB/budget stop rules.

Remaining Step 1 gates: wire existing lifecycle seams and Recipe/image adapters;
finish or precisely scope native matched-position continuation; validate the
supervisor; then execute bounded A/B/C in one admitted load, establish meaningful
40/3 numerical envelopes at equivalent boundaries, and measure cleanup. C2 is
not complete. Historical-image reconstruction and all later phases still need
separate authorization.

### Continuation validation and cleanup

Using `/Volumes/SDXC-512/ds41f-1017/bin/python`, with the explicit Flash-1017
`DS41F_CHECKPOINT`, ran `python -m pytest -q` on:

- `tests/test_prefill_numerical_reference.py`
- `tests/test_prefill_fp8_mlx_p6.py`
- `tests/test_production_mtp.py`
- `tests/test_m29_mtp_lifecycle.py`
- `tests/test_prefill_fp8_mlx_p5.py`

Result: **87 passed, 26 subtests passed**, 41 existing deprecation warnings,
44.18 s. [Compact test log](phase2-reference-continuation-tests.log).
`git diff --check`: PASS. `python tools/check_repository_self_containment.py`
using the same interpreter: PASS (35 current owners, 153 Markdown files, 584
live links). These checks are not real-model qualification.

Final read-only process/listener check found no task-owned Python/inference
worker, no LM Studio/llmster and no listener on 1234. Swap used remained zero;
Swapins/Swapouts remained zero. No model cleanup/cancellation claim is made for
an unexecuted load. Only verification tools/tests and compact evidence changed;
no production behavior, protected admission guard or numerical operator changed.
One focused local collector/evidence commit; SHA is in the final response.
No push, deployment or follow-on work.

## 9. Step 1A — first real-model Case A attempt (2026-10-25)

**BLOCKED: the meaningful-state collector rejected the candidate's NumPy int64
Engram history before completing the first layer's observation.** One real model
load, candidate prefill and fresh ordinary reference were executed; **no completed
40/3 numerical comparison was retained**. This is an observation-schema error,
not evidence of a Target/DSpark numerical invariant violation. No second load or
numerical/lifecycle retry was performed. Earlier BLOCKED results remain historical.

### Scope, identities and adapter

Started on clean `Multimodal-Runtime`, expected HEAD
`f3ebdd846844a30cedba7c3a30786cf7991a88e5`. LM Studio/llmster remained absent and
1234 had no listener. Other listeners remained rapportd, ControlCenter and GitHub;
no external inference worker/service or matching launchctl job was observed.
No unrelated process/service, installed package or checkpoint was changed.

Used `/Volumes/SDXC-512/ds41f-1017/bin/python` with `-I` for the task worker.
Installed admission ran **before** importing the checkout verification file by
path. Installed Runtime module binding was preserved; this was **installed
candidate execution**, not a claim to qualify all mutable checkout production code.
Identity digest remains `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`;
versions/build/checkpoint/numerics remain those in §1, including native GLM,
`flash-1017-pre-woa-fp8-v1` and the 48-shard Flash-1017 checkpoint on KIOXIA-PRO-2.
Installed `_start` and P6 append sources were byte-identical to checkout. Installed
paired capture/clone code was reused; checkout's later image-byte projection
helpers were not exercised. Exact source/checkpoint hashes and imported origins
are in [Case A evidence](phase2-reference-step1a.json).

Added only `observe_case_a_n2` in the existing verification tool. It calls the
existing `InternalMTPQualificationBackend._start` on an empty QualificationSession,
snapshots via the actual N-2 `checkpoint_capture` callback and existing
`PairedCheckpoint.capture`, then uses a private diagnostic exception to **end the
task at that boundary**. Native `_start`'s existing `finally` restores layer wrappers
and its resource stack releases the lease. This deliberately terminates the
qualification call, not a normal production request completion: no N-1/P5,
native generation, settlement or cache publication is claimed. Execution up to
N-2 is unchanged. The reference runs only after wrapper restoration, on fresh
ordinary factories, under the existing wired-limit lease/generation stream.
No candidate cache, hidden state, embeddings or restored pair feeds the reference.

Before weights, the adapter passed the existing small native-cache fixture test,
including capture/frontier alignment, unwind-before-reference, no input mutation,
all-40/3 collector consumption and rejection of images. The original fixture
used MLX histories and therefore missed the real NumPy-history representation.
No new framework, model loader, generic case runner or production API was added.

### Canonical input and actual execution

Existing Recipe `prepare_request('chat_completions', ..., ordinary=True)` prepared
one user message, `Explain immutable checkpoints in one short sentence.`,
`reasoning_effort=none`, temperature 0, max_tokens 1 (no output was generated).
Canonical **N=13**, observed **N-2=11**. Ordered IDs:

```text
[0,128803,65106,104124,4085,20819,295,834,3107,10175,16,128804,128822]
```

Full prompt little-endian int32 SHA256:
`8ce125c769d4a95f97bceac24a7c6aa2f8bbf36a1f0c4ff1d1c4a9ab82d79fb2`.
Candidate callback required the exact first eleven IDs. Ordinary reference used
that identical prefix, with fresh 40-layer Target and three native ring factories.
The existing boundary guard checked canonical IDs and aligned Target/ring
frontiers before entering the detailed collector. The console records the native
capture and the independent reference invocation in the
[short execution log](phase2-reference-step1a-run.log).

| Observation | Actual coverage/result |
| --- | --- |
| Model loads | **1**, `preserve_mtp=True`, `engram_ssd_offload=True`, native MTP via existing backend |
| Candidate N-2 | Captured real 40-layer/3-ring pair at 11; native pair validation succeeded |
| Ordinary reference | Executed fresh `_forward` and native DSpark append at 11; frontier checks succeeded |
| Completed meaningful Target numerical metrics | **0/40** retained; rejected while selecting layer 0 slot 6 |
| Completed DSpark numerical metrics | **0/3** retained; collector never reached rings |
| Numerical max-absolute/RMS-relative/cosine | **Unavailable**, no completed metrics serialized |
| Structural discrepancy | Collector's backend-specific dtype-object check rejected history; not a demonstrated primitive dtype/value mismatch |
| P5, logits/continuation, B/C, images | Explicitly not executed/required in Step 1A |

Source diagnosis: P6 `_make_segment_tensors` obtains NumPy history from the hasher
and `_set_cache_history` retains it; ordinary `_forward` explicitly publishes
`mx.array(history, mx.int64)`. The collector required `value.dtype == mx.int64`,
rejecting NumPy int64 despite the existing P5 validator's exact primitive dtype
contract. The run did not retain a tensor dump, so **no numerical history equality
or divergence is inferred from this diagnosis**.

After the failed observation and worker exit, corrected only the test collector
to reuse P5's existing `_dtype` spelling normalization. Both original dtype
representations remain disclosed in geometry; int32 versus int64 remains rejected,
and no state array is converted/mutated for agreement. Extended the same adapter
fixture with P6-style NumPy history and an int32 negative. **This correction has
model-free validation only; it was not rerun on weights.** The executed verification
source hash in the JSON intentionally identifies the pre-correction tool.

No cross-schedule floating tolerance was selected or enlarged. Shared checkpoint,
MLX/Metal, quantized operators, loaded layer modules, Engram data and native DSpark
projection/append operations retain §2's independence limitations. The independent
boundary is fresh ordinary dataflow/state versus candidate scheduling, not an
independent implementation of those numerical operators.

### Resources, safety and cleanup

Used a disposable external utility loop, not a persistent service or new repository
monitoring framework: existing psutil available-memory accounting, `vm_stat`,
`sysctl kern.memorystatus_vm_pressure_level`, process inspection and the existing
libproc rusage-v4 footprint offsets. Health checked about every two seconds;
compact samples retained about every ten seconds in
[resource evidence](phase2-reference-step1a-resources.json). Only the owned worker
could be terminated. Pre-load available reserve was **493.326 GiB**, above the
455.226 GiB start condition; this includes reclaimable pages, not free pages alone.
Memory pressure was normal (level 1), swap use and vm_stat swapouts zero throughout,
no external loader appeared. Minimum retained sampled reserve was **226.303 GiB**,
well above 64 GiB. Task lifetime maximum physical footprint **300.585 GiB**, below
the conservative 425-GiB stop used to stay before 430 GiB. No stop threshold fired
during the model run; elapsed worker time **124.52 s**, below all budgets.

Separate overlapping metrics (do not add): after load MLX active **287.920 GiB**,
cache **5.207 GiB**, peak **290.160 GiB**; physical footprint **293.743 GiB**.
After `model.close`, backend executor close, GC/synchronization/cache clear:
MLX active **2,052 bytes**, cache zero; process footprint still **178.208 GiB**.
That pre-exit footprint is not a leak verdict: worker then exited, host available
returned to **493.006 GiB**, and task worker/watcher were absent on final inspection.

One **pre-load** supervision false alarm is retained separately in resource JSON:
the initial disposable watcher treated a task-owned transient Python admission
child as an external loader and terminated only its own worker at installed
admission (no model load). Read-only inspection subsequently observed such a child
with the owned worker as parent. Ownership handling was corrected to exclude owned
descendants before the sole model load. This was not a numerical/lifecycle retry;
no service or unrelated process was terminated. The initial alert did not preserve
the triggering PID, so its ancestry attribution is source/process diagnosis, not
a retained PID-level proof. All temporary task scripts were removed after evidence.

### Validation and stopping point

Focused affected tests: `tests/test_prefill_numerical_reference.py`,
`tests/test_production_mtp.py`, `tests/test_prefill_fp8_mlx_p6.py`, using the admitted
interpreter and explicit checkpoint environment: **62 passed, 12 subtests passed**,
41 existing deprecation warnings, 43.63 s. `git diff --check`: PASS.
`python tools/check_repository_self_containment.py`: PASS (35 owners, 153 Markdown
files, 584 live links). Final process/port inspection found neither owned worker
nor watcher, no inference loader and no 1234 listener; pressure level 1 and zero
swapins/swapouts/use persisted. No unrelated verification stage was added. Only
the existing verification tool, its focused test and this report/compact evidence
changed. One focused local commit; SHA reported in the task response.

The **next smallest unresolved issue** is real-model validation of the corrected
NumPy/MLX exact-integer observation schema at the same N-2 boundary. That requires
separate authorization for another load; this task did not automatically retry.
Step 1A is BLOCKED, full Step 1 and C2 remain unqualified. No B/C, reconstruction,
P5/teacher forcing, tolerance campaign, push, deployment or follow-on work.

## 10. Step 1A — completed Case A comparison (2026-10-25)

**Disposition: OBSERVED.** The authorized repeat completed meaningful numerical
observation of **40/40 Target layers and 3/3 DSpark rings** at the identical
canonical **N-2 = 11** frontier. Every compared meaningful floating chunk and ring
was exactly equal in this execution: max-absolute **0**, RMS-relative **0**, cosine
**1**, nonfinite pairs **0**. This is a measured single-case result, **not an
independently approved accuracy tolerance**, comprehensive acceptance, Phase 2
completion or C2 qualification. The failed §9 attempt and its artifacts are intact.

### Review, scope and execution identity

Started on required `Multimodal-Runtime`, exact expected HEAD
`a07d559f1f9bf81ea55a80c41fc51833c5145ae1`, clean tree, one worktree and the same
single coding agent (PID 2432). No concurrent tracked changes or inference workers
were observed. Reviewed documentation policy, upstream-first sequencing, CURRENT
Phase 2/C2 criteria, Phase 1 §6, this report and verification guidance, then the
existing collector/tests, native `_start`, paired capture/clone, P6/P5 interfaces
and installed ordinary embedding/layer-loop/DSpark operations. Reused Phase 1;
no repeated upstream audit or new execution design was needed.

The missing capability was **real-model validation of the already corrected
history observation**, not a missing prefill implementation. Existing
`observe_case_a_n2`, `ordinary_prefill_reference`, `paired_boundary_metrics` and
`paired_physical_metrics` supplied the required execution and traversal unchanged.
**No repository code or test changes were necessary.** A disposable task-local
worker/supervisor used the prior approach. A passive wrapper serialized only the
three actual/reference Engram history integers before delegating to the unchanged
boundary collector; it did not substitute values or execute another comparison.
No new persistent helper, framework, fixture, loader or inference engine was added.

[Full numerical evidence](phase2-reference-step1a-comparison.json) records exact
import origins, package/build/checkpoint/source hashes, canonical input, every
layer/slot/chunk and ring metric. [Execution log](phase2-reference-step1a-comparison-run.log)
records capture, unwind, reference and cleanup. Execution used the existing
`/Volumes/SDXC-512/ds41f-1017/bin/python -I`; installed admission completed before
loading checkout verification code by path. Installed Runtime binding remained
in place. Identity SHA256 was
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`;
installed ds41f 0.23.0/build `aa8c5717b42aa0a216fef3134e72c6db7ca49287`,
oMLX 0.7.0, MLX 0.32.2, mlx-lm `0.31.4.dev132+g94cdcae13`, Recipe 0.2.0,
`glm_profile=native`, numerical selection `flash-1017-pre-woa-fp8-v1`.
Checkpoint remained the admitted 48-shard Flash-1017 on KIOXIA-PRO-2. Metadata and
installed native candidate/P6/paired source hashes matched §9; installed `_start`
and P6 append were also byte-identical to checkout. Installed weight admission
retains its unchanged-file verification-cache caveat, not a fresh independent
rehash of all weights. Nothing was installed, replaced or deployed.

Executed verification SHA256 was
`ec6865bd7c2ba5c3098449b0793e93f1029f0d487b7e1ecd0c8276f94225e1d0`,
the exact post-correction hash recorded in §9. This evidence binds that checkout
tool and the installed numerical payload, not all mutable checkout production
code or a later installed release. CURRENT contracts/progress were not changed.

### Canonical boundary and numerical observations

Recipe actually prepared the same ordinary Chat request/body as §9 before weights
loaded: `Explain immutable checkpoints in one short sentence.`, temperature 0,
max_tokens 1, reasoning none. Actual **N=13**, first-eleven frontier **11**, full
ordered little-endian int32 SHA256
`8ce125c769d4a95f97bceac24a7c6aa2f8bbf36a1f0c4ff1d1c4a9ab82d79fb2`.
Actual IDs were exactly §9's thirteen IDs; consumed IDs on both sides were
`[0,128803,65106,104124,4085,20819,295,834,3107,10175,16]`.

**Actual model load count: 1** in this task, no failed preliminary worker or retry.
The existing `InternalMTPQualificationBackend.load` retained `preserve_mtp=True`
and `engram_ssd_offload=True`. Native cold prefill reached the real N-2 callback;
existing `PairedCheckpoint.capture` validated/snapshotted the 40/3 pair. The
adapter unwound normally through its diagnostic exception, restoring layer wrappers
and releasing the candidate lease before reference execution. No N-1, P5, sampling,
settlement or publication was performed. Reference used fresh ordinary 40-layer
Target/three-ring factories over the same eleven IDs, independently recomputed
embeddings/hidden taps and native DSpark append from offset zero. It consumed no
candidate Target state, taps, embeddings or restored checkpoint.

| Observation | Actual result |
| --- | --- |
| Canonical identity / frontiers | Exact IDs; candidate/reference Target/rings all 11; boundary comparable |
| Target coverage | **40/40**; all **280** physical slots observed, all meaningful selectors equal |
| Integer/structural state | All slot shapes, primitive dtypes, presence, metadata/compression, offsets, lengths and padding agree; zero physical value mismatches |
| Meaningful window KV, slot 1 | All 40 layers, 11 rows each, absolute positions 0–10; decoded 8-bit/group-32 |
| Meaningful compressed KV/index, slots 2/3 | Layers 2/8/14: five rows each (ratio 2); layer 20: eleven rows (ratio 1); other layers have zero applicable rows |
| Meaningful compressor/gate tails, slots 4/5 | Layers 2/8/14: one row each; applicable empty state elsewhere |
| Exact Engram, slot 6 | Layer 0 `[[2638,8324,16]]` on both sides; NumPy `int64` versus MLX `int64` disclosed, exact mismatch count 0; other layers empty |
| Target floating observations | **54** bounded meaningful chunks, **244,992** decoded/tail scalar pairs; every chunk max-absolute 0, RMS-relative 0, cosine 1, nonfinite pairs/mismatches 0 |
| DSpark coverage | **3/3**; each keys shape `[1,1,11,512]`, BF16, 5,632 scalar pairs; each max-absolute 0, RMS-relative 0, cosine 1, nonfinite pairs/mismatches 0 |
| Ring mapping | Each capacity 128, offsets `[11,11]`, physical slot 0–10 maps exactly to absolute position 0–10; no rotation or alignment normalization |
| Structural/semantic mismatches | **None observed**; no threshold selected, changed or fitted |

Packed physical bytes also matched exactly, separately from decoded numerical
metrics. Existing `unpack_activation` decoded only meaningful selected slices of
at most 16 rows: slot 2 uses 4-bit/group-16/E4M3 scale; slot 3 uses 4-bit/group-32.
Empty/non-source state is recorded, not counted as populated numerical coverage.
No full decoded persistent-state copy or tensor dump was retained. Exact history
comparison tolerates only the verified backend representation difference; it does
not mask different primitive types or integer values.

Independence remains limited by shared checkpoint weights, Recipe/tokenizer,
loaded modules, MLX/Metal operators, quantization/diagnostic decoder, Engram data
and native DSpark projection/append/ring operations. Fresh ordinary layer-loop
state/dataflow versus candidate scheduling is the independent boundary. Exact
agreement here does not independently validate those shared operations, longer
chunk schedules, wrapped rings, Vision, P5 or continuation. Reference logits were
not compared against a candidate logit stream; none is claimed.

### Resource supervision and cleanup

Read-only preflight found no LM Studio/llmster, Ollama, Python inference worker,
on-demand model server, Docker VM or port-1234 listener. Listener inspection found
only rapportd, ControlCenter and GitHub. No inference launchctl job/LaunchAgent was
found; Docker's existing network/socket helpers are not a model worker. No unrelated
workload required intervention, and none was terminated, restarted or reconfigured.
These are point-in-time inspections plus ongoing competition checks, not a global
model-load lock. System/repository/model free storage was 789 GiB/190 GiB/1.1 TiB.

[Resource evidence](phase2-reference-step1a-comparison-resources.json) records the
same psutil available-memory, vm_stat swapout, pressure sysctl, process-ancestry and
libproc rusage-v4 footprint approach used previously. Owned admission descendants
were excluded from competitor checks. The supervisor withheld its load gate until
admission/canonical preparation completed and a fresh health sample passed. Health
checks ran about every two seconds, with compact samples about every ten seconds;
reserve minimum and lifetime footprint maximum were tracked at every health check.
All requested thresholds remained fixed: preload 455.226 GiB, reserve 64 GiB,
425-GiB footprint trigger before the 430-GiB ceiling, normal pressure, no increased
swap/use or swapouts, no competitor/1234 listener, 1,800-s total/600-s no-progress
budgets. Only the owned worker could be stopped. **No stop condition fired.**

- Pre-load gated available reserve: **493.421 GiB** (includes reclaimable memory,
  not free pages alone); minimum health-sampled reserve **203.108 GiB**.
- Lifetime physical-footprint high-water: **300.585 GiB**. Worker elapsed
  **108.19 s**, including admission and gate wait; supervisor exit sample 109.32 s.
- Pressure level **1 (normal)** throughout; swap use, swapins and swapouts **0**;
  no competing inference loader or 1234 listener appeared.
- Separate overlapping measures, not summed: after load MLX active **287.920 GiB**,
  cache **5.207 GiB**, peak **290.160 GiB**, footprint **293.746 GiB**. After
  comparison active **287.920 GiB**, cache **5.299 GiB**, footprint **293.899 GiB**;
  MLX peak remained **290.160 GiB**.
- After `model.close`, backend/executor shutdown, GC, stream synchronization and
  cache clear: MLX active **2,052 bytes**, cache **0**; pre-exit footprint
  **136.021 GiB**. This transient retained footprint is not a leak verdict. Worker
  PID 19318 then exited with code 0; host available returned to **493.669 GiB**.
- Final inspection after focused tests found no worker/watcher/inference process,
  no port-1234 listener, normal pressure and zero swapins/swapouts/use. Disposable
  worker/supervisor/gate files were removed; only compact evidence remains.

### Validation and remaining boundary

No preparatory test campaign or diagnostic correction was needed. Post-observation
focused collector/adapter validation using the admitted interpreter and explicit
`DS41F_CHECKPOINT`: `python -m pytest -q tests/test_prefill_numerical_reference.py`
— **11 passed in 41.21 s**, no weights loaded by tests.
[Focused test log](phase2-reference-step1a-comparison-tests.log).
`git diff --check`: PASS. `python tools/check_repository_self_containment.py`:
PASS (35 current owners, 153 Markdown files, 584 live links). Diff review found
only this report and four evidence files; no duplicated execution code, unused
helper, extra fixtures or permanent supervision infrastructure. Full per-layer
metrics are retained despite repetition so the 40/3 result is independently
inspectable rather than merely an aggregate claim. Changes committed locally;
commit SHA/working-tree status reported separately. No push or deployment.

**Smallest remaining numerical qualification gap:** independently justify the
accuracy/acceptance criterion beyond this one exact short-text N-2 observation.
This execution closes the §9 dtype-observation blocker, not the broader reference
qualification/continuation gaps. Cases B/C, P5, teacher-forced continuation,
images, prefix append, historical-image reconstruction and guard changes were
not executed or automatically selected. Phase 2/C2 remain incomplete. Stop after
Case A; any further model execution or work item requires separate authorization.

## 11. Step 1B — single-image cold prefill (2026-10-25)

**Disposition: OBSERVED for Case B at N-2 = 200.** Independently prepared image
patches and Vision embeddings, **40/40 Target layers** and **3/3 DSpark rings**
were compared. All measured values agreed exactly in this execution. This is
numerical observation, not a fitted tolerance, general qualification or C2 PASS.
Case A was not rerun; §§1–10 and their historical evidence are unchanged.

### Review, identities and smallest observation seam

Started on clean `Multimodal-Runtime`, exact expected HEAD
`971a0a6bd98d8a26e1bde39785e1f86a4501146f`, one worktree. Process/listener and
launch-service inspection found no competing inference loader, LM Studio endpoint
or concurrent tracked work. This is observation plus ongoing supervision, not a
system-wide exclusion lock. No unrelated service/process was changed.

Reviewed CURRENT documentation policy, upstream-first sequencing, Phase 2/C2,
Vision integration and verification/admission guidance; Phase 1 §6 and Case A §10;
then Recipe preparation, multimodal preparation/merge, native `_start`, P6,
production span encoding, installed `Model.get_input_embeddings`/ordinary loop,
Engram masking and existing Vision/MTP tests. The internal backend's default image
encoder requires OFF resource ownership and is **not** the correct MTP adapter.
The existing **ProductionMTPBackend** supplies the admitted pinned span encoder.
Its `_encode_new_images` source is identical between checkout and installation;
whole production modules are **not** identical. `_start`, P6 and CPU processing
are byte-identical. Installed multimodal code predates `validate_workload` and
`ImageSpan.source_pixels`; preparation and explicit fresh-budget checks were used,
not assumptions of checkout equality or modifications to the installation.

Added only `observe_case_b_n2` to the existing numerical tool and two parametrized
model-free cases in its existing test module. The helper follows Case A's native
capture/unwind pattern, preserving image identities in `PairedCheckpoint.capture`.
It calls **production `_start` directly**, not ProductionScheduler or HTTP. The
production image encoder is reused unchanged; passive wrappers observe its patches
and returned sparse rows. Native N-2 capture ends via diagnostic exception, allowing
existing finally/lease handling to restore wrappers before the ordinary reference.
No N-1, P5, generation, settlement, publication or Scheduler acceptance is claimed.
No candidate state was advanced for inspection. Production inference semantics,
protected guards, numerical operators and the historical tool CLI are unchanged.

[Machine-readable evidence](phase2-reference-step1b.json) binds actual imported
origins, installed admission/build/checkpoint identities, executed source hashes,
canonical IDs, patch/Vision metrics, every Target slot/chunk and every ring.
Execution used `/Volumes/SDXC-512/ds41f-1017/bin/python -I`; installed admission
completed before loading the checkout verification file by path. Runtime remained
bound to installed ds41f **0.23.0**, source
`aa8c5717b42aa0a216fef3134e72c6db7ca49287`; installed identity SHA256
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
Python 3.13.15, oMLX 0.7.0, MLX 0.32.2, mlx-lm
`0.31.4.dev132+g94cdcae13`, Recipe 0.2.0 and native GLM were admitted with
`flash-1017-pre-woa-fp8-v1`. Checkpoint was the required KIOXIA-PRO-2 Flash-1017,
48 admitted shards; unchanged-file verification-cache caveat remains, not a new
independent full-weight rehash. No assets/dependencies were installed or changed.
Executed tool SHA256:
`14ce943f882d8edee82ed6af92f79a0b34aadce2bf28dffd183c38894850af5e`.

### Canonical input, independent preparation and numerical results

Before weights, existing Recipe ordinary Chat preparation processed one user
content list: corn image, then `Identify this food in one short sentence.`
Temperature 0, max_tokens 1, reasoning none; no output token was generated.
Canonical Recipe ordering required no change. Actual input:

- `corn.jpeg`, **56,122 bytes**, **450×308** source pixels; verified SHA256
  `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1`.
- **N=202**, canonical frontier **200**. Expanded image occurrence 0 is
  **[2,191)**, length **189**; grid **(33,47)**, **1,551** ViT patches.
  Kinds: start 0 ×1, features 1 ×176, newline 2 ×11, end 3 ×1. All image
  positions precede N-2; terminal IDs **128804,128822** are non-image text/control
  positions under the existing convention. No span truncation or frontier split.
- Exact IDs are retained in JSON. Compact notation (not a handcrafted input):
  `[0,128803] + [129264]×189 +
  [271,71343,566,3344,295,834,3107,10175,16,128804,128822]`.
  Full little-endian int32 SHA256:
  `96c66aaf8db245ad3a5cd247bb74fa87c3892b376a33e2573d4449b8df986597`.
- Existing full-container/pixel validation and preparation enforced image byte,
  pixel, aspect, token and aggregate budgets. Explicit fresh checks confirmed
  1,551 <36,864 patches and 138,600 <16,777,216 source pixels. Encoded JSON body
  **75,115 bytes** <16 MiB. Expanded prompt plus one reserved output fits both
  checkpoint context and the 4,096-position experiment bound.

Reference pixels were constructed by a separate PIL decode and installed
`image_patch_array` invocation from the independently re-read, digest-verified
source bytes. Candidate preparation independently consumed its request-owned
source through the existing production span encoder. CPU patches and BF16
conversion were compared without substituting either input. The reference then
called **ordinary_prefill_reference** with exact canonical first-200 IDs, its own
pixels, grid, absolute span and kinds. Fresh 40-layer Target/three-ring factories,
ordinary `_forward`, recomputed hidden taps and native append from offset zero
received **no candidate embedding, Target state, tap or checkpoint**.

| Observation | Actual result |
| --- | --- |
| Prepared patches | Both `[1551,3,14,14]`; float32 and BF16, 911,988 scalar pairs each; exact layout/value equality |
| Vision embeddings | Candidate per-span versus independent full-prefix reference: `[189,5120]` BF16, 967,680 pairs; all rows equal, including start/newline/end groups at recorded absolute positions |
| Image masks | Passive native Engram-hasher observation saw exactly two `[1,200]` calls: candidate/reference IDs and Boolean masks equal; true exactly at **[2,191)**, including delimiters; native hasher masks these IDs to sentinel -1 |
| Boundary | Both Target/ring frontiers `[200,200]`; exact consumed canonical IDs |
| Target coverage | **40/40**, all **280 physical slots**, presence/shapes/primitive dtypes/metadata/compression/offsets/padding/lengths equal; zero physical value mismatches |
| Meaningful window KV | All 40 layers, 128 chronological rows, absolute **[72,200)**; existing 8-bit/group-32 decoder |
| Compressed KV / index | Layers 2/8/14: 100 rows each, ratio 2; layer 20: 200 rows, ratio 1; non-source rows inapplicable; existing 4-bit/group-16/E4M3 and 4-bit/group-32 decoders |
| Compressor/gate tails | Zero applicable rows at this even frontier; empty structures observed, not populated numerical coverage |
| Engram history | Layer 0 three int64 integers equal, zero mismatches; NumPy versus MLX representation disclosed; applicable empty state elsewhere |
| Target meaningful floating metrics | **388** chunks, **2,941,440** scalar pairs; every chunk max-absolute **0**, RMS-relative **0**, cosine **1**, nonfinite pairs/mismatches **0** |
| DSpark coverage | **3/3**, each `[1,1,128,512]` BF16, 65,536 pairs; max-absolute **0**, RMS-relative **0**, cosine **1**, nonfinite pairs/mismatches **0** |
| Wrapped ring mapping | All capacities 128/offsets 200; physical slots map to **[128,200) then [72,128)**, identically, without rotation normalization |
| First meaningful divergence | **None observed**, including packed physical bytes, masks, image metadata and meaningful decoded values |

Packed decoding remained bounded to at most 16 selected rows, with no persistent
full decoded-cache copy or tensor dump. Actual floating equality was not assumed
in advance and sets **no acceptance tolerance**. Shared checkpoint/modules,
Recipe/tokenizer, CPU recipe, Vision kernels, MLX/Metal, quantization decoder,
Engram and native DSpark math limit independence: this compares fresh ordinary
state/dataflow against P6 scheduling, not independent implementations of all math.
The 202-position case does not exercise the ≥8192 deferred bulk/tail branch.

### Supervision, cleanup, validation and stopping boundary

**One model load**, no retry or diagnostic failure after load. Existing backend
loader retained `preserve_mtp=True`, `engram_ssd_offload=True` and native profile.
The [short log](phase2-reference-step1b-run.log) records real capture/unwind,
independent reference and cleanup. The disposable external supervisor followed
Case A: psutil available reserve, vm_stat swapouts, pressure sysctl, process
ancestry/competitor checks and libproc rusage-v4 current/lifetime footprint.
Owned admission descendants were excluded; a fresh healthy load gate was required.
Health sampling was approximately two seconds, compact samples approximately ten
seconds, with minima/maxima tracked at each health check. No threshold fired.

[Resource evidence](phase2-reference-step1b-resources.json):

- Pre-load gated reserve **493.163 GiB**, above **455.226 GiB**; minimum reserve
  **197.763 GiB**, above **64 GiB**. Available includes reclaimable pages, not a
  memory reservation or free-pages-only claim.
- Lifetime physical-footprint high-water **300.576 GiB**, below the conservative
  **425 GiB** trigger / **430 GiB** ceiling. Worker/supervisor elapsed **110.55 s**,
  below 30-minute total and 10-minute no-progress limits.
- Pressure level **1 (normal)**, swap use/swapins/swapouts **0** throughout;
  no competing loader or 1234 serving listener appeared. Only owned worker could
  have been terminated; none required termination.
- Separate overlapping MLX metrics, not summed with footprint: after load active
  **287.920 GiB**, cache **5.207 GiB**, peak **290.160 GiB**; after comparison
  active **287.920 GiB**, cache **7.327 GiB**, peak **290.160 GiB**.
- After model close, backend/executor shutdown, GC, stream synchronization/cache
  clear: MLX active **2,058 bytes**, cache **0**. Post-close physical footprint was
  not separately sampled before exit; no claim of allocator-leak measurement.
  Worker **55993** exited code 0 and supervisor **55988** exited; host reserve
  returned to **493.671 GiB**. Final process/listener inspection confirmed absence,
  normal pressure and zero swap. Temporary worker/supervisor/gate files removed;
  their hashes remain in evidence. No duplicate model/image asset was retained.

Before loading, affected focused tests using the specified interpreter/checkpoint:
`tests/test_prefill_numerical_reference.py`, `tests/test_metadata_first_vision.py`,
`tests/test_prefill_fp8_mlx_p6.py`, `tests/test_production_mtp.py` — **106 passed,
12 subtests passed**, 41 existing warnings, 45.81 s. New tests exercise real tiny
native Vision through the unchanged production adapter, independent patch
construction, image/delimiter metrics, N-2 unwind-before-reference, source/frontier
negatives and deliberate candidate patch divergence without masking it. Existing
reference freshness and full-state collector negatives remain. No weights loaded
by tests. [Focused log](phase2-reference-step1b-tests.log).

`git diff --check` and `python tools/check_repository_self_containment.py`: PASS.
Policy/diff review: only verification tool/tests and appended compact evidence;
no competing execution owner, guard relaxation, production change, installed
payload modification or CURRENT qualification/progress update. One focused local
commit; SHA reported separately. No push or deployment.

Unobserved: production Scheduler/HTTP admission/lifecycle, N-1/P5, continuation and
matched logits, changed/history projections, cache recovery and longer schedules.
No independent floating acceptance envelope is established by this exact sample.
Case C, reconstruction, Harness lifecycle and general C2 remain unqualified and
were not started. **Stop after Case B; no automatic follow-on work.**

## 12. Step 1C — compatible restored-prefix append (2026-10-25)

**Disposition: OBSERVED, with meaningful floating differences.** A real seed pair
was saved/restored at **C=11**, then exactly **5** missing positions **[11,16)**
executed to current **N-2=16**. Fresh full-current-prefix ordinary prefill from
zero was compared across **40/40 Target layers and 3/3 DSpark rings**. Restore
values, snapshot immutability and cross-schedule exact structural invariants held.
Floating agreement did **not** hold; no general floating envelope exists under §4,
so this is neither numerical PASS nor an established tolerance violation. It does
not qualify reconstruction, Phase 2/C2, P5, Scheduler reuse or continuation.
Cases A/B were not rerun as qualification experiments.

### Review, identity and minimal integration

Started on clean `Multimodal-Runtime`, expected HEAD
`667158ca8802489049e7f5d9bc661d00ab52a018`, one worktree, sole observed coding
agent PID 2432; no competing inference loader or LM Studio/1234 endpoint. Reviewed
documentation policy, upstream-first sequencing, Phase 2/C2, ordinary MTP,
verification, Phase 1 §§4–6 and reference construction/acceptance/Cases A/B above.
Inspected native `_start`, paired capture/restore, production Scheduler and span
adapter, P6 append/committed APIs, installed ordinary `_forward`/Target and ring
factories/native DSpark append, and existing collector/paired/P6 tests. Reused
Phase 1 findings without repeating the upstream audit.

The missing capability was a **Case C observation adapter**, not a cache lookup,
prefill engine or model loader. Added `observe_case_c_n2` in the existing tool:
normal cold seed capture/unwind; diagnostic `PairedCheckpoint.restore` against the
current prompt; direct `_start` append/capture/unwind; then unchanged
`ordinary_prefill_reference` and existing physical/meaningful collectors. A passive
`execute_all` wrapper records actual P6 C/T without changing the plan/operations.
A bounded digest covers retained seed values/metadata before and after append;
BF16-to-FP32 digest conversion is exact. Mutable Target/cache/ring containers are
checked distinct; same-operation restore metrics cover the complete pair.
**No ProductionScheduler lookup, HTTP request execution, publication or settlement
occurred.** There is no new cache authority, numerical engine or production
selector. No P6 scheduling, quantization, native DSpark or P5 behavior changed.

[Full evidence](phase2-reference-step1c.json), [run log](phase2-reference-step1c-run.log)
and [resources](phase2-reference-step1c-resources.json) bind this single execution.
Interpreter: `/Volumes/SDXC-512/ds41f-1017/bin/python -I`. Installed admission
completed before loading checkout verification code by path; Runtime remained
installed ds41f **0.23.0**, build `aa8c5717b42aa0a216fef3134e72c6db7ca49287`.
Identity SHA256: `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
Python 3.13.15, oMLX 0.7.0 with the four admitted overlays, MLX 0.32.2,
mlx-lm `0.31.4.dev132+g94cdcae13`, Recipe 0.2.0/native revision
`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus semantic patch;
`glm_profile=native`, numerics `flash-1017-pre-woa-fp8-v1`.
Required Flash-1017 checkpoint on KIOXIA-PRO-2: 48 admitted shards, same identities
as §10; unchanged-file weight-verification cache caveat remains, not a fresh
independent full-weight rehash. Executed tool SHA256:
`b1a92a874648eda3d1d1fa210d3197e589fb46a8464c6c7e62c7a259e8e86bbf`.
Installed `_start` and P6 source matched checkout. Other whole modules are not
assumed identical. Nothing was installed, replaced or deployed.

### Canonical compatibility, real restore and execution

Before weights, installed Recipe ordinary Chat preparation produced both requests
with temperature 0, max_tokens 1, reasoning none. User text:

- Seed: `Explain immutable checkpoints in one short sentence.`
- Current: `Explain immutable checkpoints in one short sentence. Then state one benefit.`

The instruction was appended **inside the user message before Recipe framing**;
no hand-built token concatenation/delimiters or matching-rule change. Actual IDs:

```text
seed:    [0,128803,65106,104124,4085,20819,295,834,3107,10175,16,128804,128822]
current: [0,128803,65106,104124,4085,20819,295,834,3107,10175,16,5715,2501,834,8200,16,128804,128822]
```

Ordered little-endian int32 SHA256:
seed `8ce125c769d4a95f97bceac24a7c6aa2f8bbf36a1f0c4ff1d1c4a9ab82d79fb2`;
current `9059c79e53e40e13f60272e37226a6f5d99f2d5d6f53879d4dbaff1ff91c4c1b`.
Seed N=13/digest exactly matched Case A. Both first-eleven canonical IDs match;
current N=18, N-2=16>11. Existing ordinary admission resolved the one-output
reservation; even N+1+5 speculative positions is below 4096 and native context.
No image inputs/masks are applicable. No continuation suffix was selected.

**One model load**, via existing backend (`preserve_mtp=True`,
`engram_ssd_offload=True`). Cold seed executed **[0,11)** once and native capture
validated all 40 Target/seven-slot states and three rings at C=11. The captured
checkpoint bound those exact eleven IDs and the admitted execution digest above.
`restore` validated the current prompt and cloned/materialized the complete pair:
actual restored Target/ring frontiers **11/11**, all same-operation restore
structures/values equal, all mutable containers distinct from retained snapshots.
Actual P6 execution records are seed `{start:0,end:11,rows:11}`, append
`{start:11,end:16,rows:5}`. No append execution started at zero, no prefix replay
was observed, and neither N-1 nor P5 was reached. Diagnostic capture exceptions
unwound the native layer wrappers/lease before reference execution.

Retained seed digest before/after append was exactly
`4a41611c277fd2cea0f8b9b5a90b48956289d09889c97d4c9e8231e19ed82d3c`.
No snapshot mutation or unintended mutable state sharing was observed. The seed
snapshot, restored candidate and fresh reference coexisted under supervision;
small-case memory was measured, not inferred from A/B.

Reference consumed **all sixteen current canonical IDs [0,16)**, starting at zero,
using fresh ordinary Target/ring factories and independently recomputed embeddings
and hidden taps. It received no restored pair, candidate Target/rings, embeddings,
taps or outputs. Shared weights, kernels, MLX primitives, Engram/native DSpark math
and diagnostic unpacker limit independence as in §2: independent schedule/state,
not independently implemented math.

### Structural and numerical observation

| Observation | Result |
| --- | --- |
| Matched boundary | Exact consumed current IDs; candidate/reference Target/rings all **16** |
| Target structures | **40 layers × 7 slots**; presence, shapes, primitive dtypes, offsets, metadata/compression, lengths and padding equal |
| Exact Engram | Layer 0 three int64 entries, mismatch count 0; NumPy/MLX representations disclosed; other histories empty |
| Window KV | All 40 layers, 16 meaningful rows, absolute **[0,16)**; 8-bit/group-32 decoding |
| Compressed KV/index | Layers 2/8/14: 8 rows each (ratio 2); layer 20: 16 (ratio 1); other source rows inapplicable; 4-bit/group-16/E4M3 and 4-bit/group-32 decoding |
| Compressor/gate tails | No meaningful tail rows at even C=16; empty structures still compared |
| Target floating coverage | **48 chunks, 353,280 scalar pairs**; **230,446** differing values, nonfinite pairs **0** |
| Target extrema over meaningful chunks | Max-absolute **2.0** (layer 39 window); largest RMS-relative **0.2095074462**, smallest cosine **0.9780551779** (layer 37 window) |
| Rings | All 3 `[1,1,16,512]` BF16, 8,192 scalar pairs each; capacity 128, offsets 16, physical slots map exactly to absolute **0–15** |
| Ring 0 | Max-absolute **0.5**, RMS-relative **0.1298231648**, cosine **0.9915448918**; 6,055 mismatches |
| Ring 1 | Max-absolute **0.484375**, RMS-relative **0.1549562178**, cosine **0.9879770956**; 6,254 mismatches |
| Ring 2 | Max-absolute **0.53125**, RMS-relative **0.1707380970**, cosine **0.9854016948**; 6,463 mismatches |
| Ring nonfinites / mapping | Zero nonfinite pairs; exact mapping, no rotation normalization |

**First meaningful discrepancy:** layer **0**, slot **1** window KV, first decoded
flat index **11** = absolute row **0**, channel **11**: candidate **0.0234375**,
reference **0.025390625**. That chunk has max-absolute **0.125**, RMS-relative
**0.0081874600**, cosine **0.9999664847**, 319 differing decoded values. Thus even
retained-prefix floating rows differ from fresh full-current-prefill; append
execution records nevertheless show no prefix replay. Every Target layer's window
has differences. Packed-byte differences are recorded separately, not interpreted
as floating error. Quantized source/index differences also occur. No meaningful
rows were excluded and no tolerance was selected/fitted. The cause is **not
established by this observation**; exact A/B cold results cannot establish
cross-chunk append equality. No corrective reload or automatic follow-on diagnosis
was performed. Complete traversal supports OBSERVED, not an approved envelope.

### Safety, cleanup and validation

The disposable external supervisor reused psutil available reserve, vm_stat
swapout/use, pressure sysctl, process-ancestry/competitor and port-1234 checks,
and libproc rusage-v4 current/lifetime physical footprint (SDK offsets 72/240).
Health samples approximately every 2 seconds, retained samples approximately every
10; minima/maxima tracked at each health check. Owned admission descendants were
excluded from competitor detection. Fresh pre-load gate reserve **493.259 GiB**
exceeded **455.226 GiB**; minimum reserve **199.269 GiB** exceeded **64 GiB**.
Lifetime physical-footprint maximum **300.583 GiB**, below **425 GiB** trigger and
430 GiB ceiling. Worker **68526** elapsed **106.91 s**, supervisor **108.66 s**,
below 30-minute/10-minute-no-progress limits. Pressure **1 (normal)**, swap use,
swapins/swapouts **0** throughout; no competitor/endpoint or stop threshold.
Only task-owned workers could be terminated; no termination was necessary.

Overlapping measures, not additive: after load MLX active **287.920 GiB**, cache
**5.207 GiB**, peak **290.160 GiB**; after comparison active **287.920 GiB**, cache
**5.395 GiB**, same peak. Model close, backend/executor shutdown, GC, stream sync
and allocator clear left MLX active **3,256 bytes**, cache **0**. Post-close process
footprint was not separately sampled, so no allocator-leak verdict. Worker exited
code **0**, exit verified; supervisor exited. Host reserve returned to **493.509
GiB**. Final process/listener inspection after tests found no worker, competing
inference loader or 1234 listener, normal pressure and zero swap/use. Temporary
scripts/gates were removed; only hashes/compact metrics/logs retained, no tensors
or duplicate assets.

Focused validation ([log](phase2-reference-step1c-tests.log)):
`tests/test_prefill_numerical_reference.py tests/test_production_mtp.py` — **49
passed**, 41 existing warnings, 43.80 s (before load);
`tests/test_prefill_fp8_mlx_p6.py` — **20 passed, 12 subtests passed**, 35.94 s
(after worker exit). Five new adapter cases cover successful restoration/full
reference, incompatible prefix, missing suffix, incorrect restored frontier and
snapshot mutation; existing paired tests cover identity and wrapped-ring snapshot
negatives. No weights loaded by tests; unchanged image/A/B campaigns not repeated.
`git diff --check` and `python tools/check_repository_self_containment.py`: PASS.
Review found only the small adapter/tests, evidence and factual stale progress
corrections; no duplicate authority/framework or unnecessary production change.
Committed locally; SHA reported separately. No push/deployment.

**Unqualified:** general floating accuracy envelope; N-1/P5, teacher forcing and
matched-logit continuation; ProductionScheduler cache-hit lifecycle; changed-image
or history reconstruction; Harness lifecycle; Phase 2/C2 acceptance. The observed
floating differences remain unresolved evidence, not hidden by a tolerance.
**Stop after Case C. No follow-on work is authorized or started.**

## 13. Step 1D — DeepInfer-first numerical attribution (2026-10-25)

**Disposition: UNRESOLVED for Case C; source-level attribution narrowed.** No
specific violation of an established DeepInfer invariant, local numerical defect,
correction or acceptance envelope is established. Historical fixed-input effects
remain **SHAPE_DEPENDENCE_OBSERVED** within their original fixtures, not a cause
certificate for this case. **Zero model loads and zero new numerical replays.**
Stage A establishes the supported execution behavior and excludes the prior L0
leaves as direct producers of the observed KV difference; that is sufficient to
stop without an unnecessary full-model repetition. Stages B/C were not invoked.
No production code, guard, tolerance, dependency or execution selector changed.
Cases A/B/C and their receipts remain immutable; Phase 2/C2 remain unqualified.

### Authority, checkout and installed identity

Started on clean `Multimodal-Runtime` at the exact expected
`4d8bc6fefcb1d088805592deb0522c7b4dd9cb4d`; one worktree. Process inspection
found no competing coding work/model loader; no LM Studio/port-1234 listener.
These are point-in-time checks, not a system-wide lock. Reviewed CURRENT
`documentation-policy.md`, `implementation-plan.md` (upstream-first and sequencing),
`multimodal-runtime-plan.md` (Phase 2/C2), `mtp-production-serving.md` and
`verification.md`, then §§10–12 and the full structured Step 1C receipt.

Cloned **https://github.com/deepseek-ai/DeepInfer** read-only for review. Actual
fetched HEAD and inspected revision are both
**`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`**, identical to the requested
reference; no newer local revision was mixed in. Primary inspected paths:

- `README.md`, `deepinfer/environ.py`, `deepinfer/worker/runtime_config.py`;
- `deepinfer/model_executor/models/v4_modules/attn.py`;
- `deepinfer/model_executor/layers/attention_backend.py` and `linear.py`;
- `tests/models/test_batch_invariant.py`, `doc/inference_deployment_qa.md`.

Followed golden-token plumbing in `engine/llm_engine.py`, `sampling_params.py`,
`csrc/engine/{pybind_sequence.cpp,sequence.hpp,prepare_sample.cpp}` and searched
model tests for explicit chunk/cache-equivalence coverage. Vendor submodules were
not initialized or executed: the tree pins DeepGEMM
`1fc9c66975ee322d628b3288534ca0e9ce12a64d` and FlashMLA
`2e5429fc5653bab6e081f09477126f731882a6a9`. Findings below describe the official
configuration/call contracts, not independently verified kernel internals or a
new run of upstream tests.

[Compact source evidence](phase2-reference-step1d.json) records hashes, package
origins/build provenance, historical Git blob identities and planner assertions.
Isolated `/Volumes/SDXC-512/ds41f-1017/bin/python -I` installed payload inspection
passed without checkpoint/weight admission or loading: Python 3.13.15, macOS
26.5.2, ds41f 0.23.0, oMLX 0.7.0, MLX 0.32.2, mlx-lm
`0.31.4.dev132+g94cdcae13`, Recipe 0.2.0, native GLM,
`flash-1017-pre-woa-fp8-v1`. Installed build remains
`aa8c5717b42aa0a216fef3134e72c6db7ca49287`, with compressed-growth,
bit-exact-kernels, template-reuse and context-copy overlays. The oMLX direct-URL
build directory no longer exists; its pathname/version alone is not source proof.
Actual installed files were inspected and hashed instead. Executed Step 1C
`internal_mtp`, paired-checkpoint, P6, oMLX language and MTP hashes still match;
all overlapping installed numerical hashes checked against the earlier preflight
also match. The old complete execution identity/checkpoint digest in §12 is
inherited evidence, **not freshly re-admitted 48-shard evidence here**.

Checkout P6, block runner, `_start` and first-party quantization match their
installed ds41f copies. Whole paired/production modules differ; the exercised
paired copy/clone/restore/validation semantics agree (AST comparison), while newer
byte-budget/lookup-admission additions are not assumed installed. Actual Target
execution uses **oMLX `patches/deepseek_v41/language.py`**, not first-party
`model_execution/language.py` or generic mlx-lm `deepseek_v41.py`.

### Official invariant coverage versus local properties

Classifications describe the inspected upstream evidence, not unsupported-feature
defects. All upstream references in this table use the revision above. Local
references use the installed identity above and the checkout hashes in the receipt.

| Numerical property | Official implementation / evidence class | Local equivalent and Case C relevance |
| --- | --- | --- |
| Request-batch composition invariance | **IMPLEMENTED_AND_TESTED**, bounded by the published generation fixture. `runtime_config.configure_worker_kernels` enables deterministic DeepGEMM/TileKernels; `attn.py:89` changes selected fusion eligibility; decode wrapper forwards FlashMLA `enable_batch_invariant`. README explicitly NVIDIA-only for this mode. | No equivalent composition-invariant mode established in this admitted MLX path. Ordinary production does not combine independent requests into one native batch; Case C compares singleton row geometries, not different request batches. Missing global guarantee is not a demonstrated Case C defect. |
| Prompt-chunk-boundary bitwise invariance | **NOT_ESTABLISHED**. Budgeted chunking is described in Q&A; neither that scheduler architecture nor the batch test explicitly compares fixed chunk partitions. | P6 and ordinary implement contiguous causal execution, not a documented chunk-invariant arithmetic contract. No correction follows from comparing 11+5 against 16. |
| Prefill versus decode numerical equivalence | **NOT_ESTABLISHED** as an exact invariant. `attn.py` retains distinct prompt/decode paths and cache representations. | Local packed/native attention and projection dispatch also depend on row count. Case C ends before P5/decode, so decode equivalence would not settle it. |
| Cached-prefix continuation equals full-prefix recomputation | **NOT_ESTABLISHED** as bitwise distributions or cache-state equality. Cache reuse/recompute architecture is supported, but no matching 11+5/16 test was found. | Compatible exact restore is measured in §12; fresh full-current execution equality is not. Structural/frontier identity is insufficient. |
| Reuse of existing KV / immutable retained state | Reuse without recomputing/overwriting block KV in bounded replay is **DESIGN_OR_SOURCE_ONLY** here (`inference_deployment_qa.md`; attention load/store mappings). An externally retained immutable snapshot/clone contract is **NOT_ESTABLISHED** by the public batch test; full KV storage service is not published. | **UPSTREAM_ALIGNED**, narrowly: existing paired authority restores values without numerical repack, independent mutable containers and immutable retained seed were observed in §12. Live restored-prefix preservation during append is a distinct, not yet measured assertion. No replacement cache/checkpoint owner is needed. |
| Quantized KV / sparse-attention numerical consistency | **IMPLEMENTED_WITH_LIMITED_EVIDENCE**: `_prepare_swa_kv` applies norm/RoPE/E4M3 with UE8M0 scales; prompt casts back, decode stores packed; compressed KV/index formats are documented. Batch test indirectly exercises numerical outputs, not all internal state or every layout relation. Exact chunk/prefill/decode equivalence remains **NOT_ESTABLISHED**. | Existing pack/unpack, group-32 FP8 SWA, group-16 E4M3-scale FP4 compressed KV, group-32 FP4 index, absolute positions and causal sparse indices already supply corresponding format/history properties. Format correspondence is not CUDA bitwise conformance. No new quantizer or CUDA kernel translation is justified. |
| Deterministic output distributions / committed state | Returned token logprobs and top-5 IDs/logprobs on the **fixed reference trajectory** are **IMPLEMENTED_AND_TESTED** across the tested batch compositions. Full vocabulary distributions and committed Target/DSpark arrays are **NOT_ESTABLISHED** by this test. README's fixed-`seq_seed` deterministic sampling recipe additionally disables variable-length MTP; it is not this test's autonomous-rollout proof. | Existing sampler/RNG and paired committed-state ownership do not establish equality of distributions across Case C schedules. No logits or generation were compared in §12; do not infer them from cache metrics. |

**Actual batch-invariant test:** one model case, EP=8, maximum nextn=5,
64 random non-special-token prompts per DP rank (lengths around 2500±1000),
max batched/model tokens 16384. It generates 50 tokens at temperature 1/top-p .7,
ignore-EOS, then runs each request singly with `golden_token_ids` copied from its
batched result. Python equality compares token IDs, selected token logprobs,
top-token IDs and top logprobs (top-5). The singly executed continuation is thus
conditioned on the same reference-generated history; token equality alone is not
independent seeded-sampling equivalence. No prompt-logprob comparison, forced
11/5 partition, external restore, all-layer cache comparison or full-distribution
array comparison is present. Tests being published is not a claim they were run
on this host or that every possible batch has been verified.

Upstream deliberately bounded replay is **not exact recovery**: Q&A rebuilds
SWA using keys `max(s,i−127)..i` for a replay starting at s, explicitly yielding
approximate SWA. Encoder replay reuses block KV unchanged; decoder replay's
approximate SWA is used for decode and never enters prefix cache. Stored SWA
alignment (128), first-20-layer storage, backtracking reserve and missing external
cache service differ from the local complete 40-layer/three-ring checkpoint.
These are not a license to accept Case C error or implement local replay. Case C
uses no bounded-replay/CED approximation branch.

### Historical evidence, not a present acceptance contract

Used `git show e36779ab5a6765a982cbfa618c6b705e454db71c:<path>` for
`docs/flash-1017-cross-route-investigation.md`, its JSON, and
`docs/flash-1017-l0-leaf-attribution.md`, its JSON. Their five-token MTP-OFF
fixture executed four rows versus four one-row segments. Fixed-input leaf replay
isolated `wq_b` max-absolute **1.52587890625e-5** (one value) and `wo_b`
**0.000244140625** (three values), even with the activation round trip removed
from the comparison. Fixed-input RoPE, grouped `wo_a` and input Q/DQ controls
were exact there. Independently, L2 compact versus padded sparse slots changed
online-softmax/BF16-PV results by **0.0078125 after inverse RoPE**, with identical
Q/KV/sink/ordered valid rows. Compact single-query versus compact batched was
exact. Historical L0 packed KV was exact, unlike present Case C.

That proves **SHAPE_DEPENDENCE_OBSERVED** for those operations/inputs and multiple
historical mechanisms, not **MULTIPLE_MECHANISMS** causing present Case C or its
full Target/DSpark discrepancy. Neither later amplification nor full-logit causal
sufficiency was demonstrated. Numerical profile, MTP ownership, row geometry and
some installed projection dispatch now differ.

Also reviewed `df234dd37df550c063456231137d36a70e68f2f3` CUDA sparse comparison
report/receipt. Its isolated Python lacked Torch/TileLang and its frozen raw L2
fixture was absent: no official CUDA compilation/execution or cross-backend
numerical comparison completed. Driver/hardware visibility was not conformance.
That historical **BLOCKED** does not make current MLX execution blocked, and its
old MTP prohibitions are not the current serving contract. No CUDA work was begun.

### Actual A/B/C/D shapes and Layer 0 dependency graph

Inspected P6 planner, command runner, installed ordinary `_forward`, production
`_prefill_tap_start`, native `_start`, DSpark append and paired clone path. Ran
**planner-only** assertions using the installed P6 implementation: all 40 layers
have one `ENCODE_ROWS` command of the indicated size in each plan. This checks
below the logical segment boundary; no weights or model instance were needed.

| Path | Target execution through observed frontier | Shapes relevant to L0 |
| --- | --- | --- |
| A seed (historically executed) | `[0,11)`, one `ORDINARY_COMPLETE_RANGE` | One 11-row block/projection batch; attention input `[1,11,5120]`, new window bytes `[1,11,528]` |
| B restored append (historically executed) | Restore 11; `[11,16)`, one `ORDINARY_COMPLETE_RANGE` | One 5-row block/projection batch; new packed rows `[1,5,528]` concatenated to 11 old rows |
| C cold P6 control (**not numerically executed**) | Planner `[0,16)`, one `ORDINARY_COMPLETE_RANGE` | One 16-row block/projection batch, no restored cache |
| D fresh ordinary (historically executed) | `_forward` on `[1,16]` IDs; one singleton row, each layer receives all 16 positions | One 16-row block/projection batch, separate fresh Target/rings |

These short P6 plans do **not** use the `remaining<=3` one-row rule from the
historical four-versus-one fixture, the 8192 encoder tile split, source-only
publication, decoder cone, CED suffix reconstruction or bulk/ring-tail bridge.
Ordinary D does not enable `_ced_prefill` and its >=256-row backpressure is off.
P6's command materialization/publication and ordinary's extract/merge/lazy loop
still differ; equal 16-row command geometry would not prove identical arithmetic.
A/D 11/16-row attention satisfy the installed native packed-attention `length>8`
branch (admitted symbol present), while B's five-row attention uses the other
packed sparse path. This can affect appended/later-layer results, **not already
produced L0 prefix KV**.

The actual L0 KV-producing graph is:

```text
IDs -> embedding -> repeat HC streams
    -> Layer-0 Engram (hash history/table -> Engram.wkv -> gating/add)
    -> hc_pre_norm(h, initial pre, attn_norm) -> attention input x
    -> Attention._input_projections(x): activation Q/DQ -> wkv projection
    -> kv_norm -> RoPE at absolute positions -> pack_activation (FP8/group32)
    -> concatenate retained packed window rows -> retain last 128 rows
```

The exact Engram integer history in §12 does not prove equal Engram projection,
gate/hidden values, `hc_pre_norm` output or attention projection input. The HC
mixes also produce attention residual/next-pre coefficients; they do not feed the
current L0 KV through `wq_b`/`wo_b`. L0 `wq_b` is a sibling query branch, while
attention, inverse RoPE, pre-`wo_a`, `wo_b`, FFN and all later layers are downstream
of this KV store. Thus historical `wq_b`, `wo_b` and L2 padding cannot directly
cause the present L0 position-0 KV value. Shape sensitivity in **earlier** Engram
projection, HC/pre-norm, activation conversion or `wkv` remains plausible but
unmeasured; different numerical inputs must be excluded before calling any leaf
comparison a fixed-input shape effect.

Important installed distinction: oMLX `QuantizedProjection.project_quantized`
tries `gemv.mxfp8_gemv` before `mx.quantized_matmul`, unlike the first-party
projection read above. Its guarded specialized path accepts eligible 1/4/5 rows;
11/16 fall back. The kernel claims matching MLX arithmetic for each corresponding
shape, not invariance between shapes. This is an existing capability, not a
proposed alternative path. It cannot by itself change seed-prefix rows on append;
it is a possible separate appended-row boundary requiring fixed inputs before
attribution. Pack/unpack function ASTs match the first-party implementation;
projection dispatch does not. No kernel was modified or replayed.

`pack_activation` computes amax/scales within each row's 32-channel group, not
across tokens. It packs **new** KV only. In this case installed attention reads
`old_len=min(11,128,11)`, concatenates `old[:,:11]` and new rows, then slices the
last 128 of 16: no old-row decode/repack or truncation is specified. P6 publication
advances offsets/history and preserves persistent handles; no L0 window rewrite
is introduced by these ordinary-complete commands. Snapshot restore/capture only
clones/materializes arrays. This establishes **source behavior**, not a measured
live-prefix equality certificate or proof against a backend/store bug.

DSpark independently projects reduced Target hidden taps via its own `wkv`,
normalization/RoPE/Q-DQ and absolute-offset ring append. Its shared-prefix hidden
inputs, append values and row geometries were not separately isolated. Target
error magnitude does not explain DSpark error magnitude; no common cause for all
40 layers/three rings is established.

### What is proven, what remains open, and minimum action

The earliest **proven differing observed boundary** remains §12's decoded stored
**L0 window KV, absolute position 0/channel 11**: **0.0234375 versus
0.025390625**. This is after packing/state update, not the earliest arithmetic
operation. L0 window max-absolute **0.125**, RMS-relative **0.0081874600**;
overall Target **2.0 / 0.2095074462**, DSpark **0.53125 / 0.1707380970**
(max-absolute / largest RMS-relative) are inherited measurements, not new ones.

| Immediate distinction | Present evidence |
| --- | --- |
| A seed prefix preserved in live B after append? | Source says yes for this short non-truncating window; **not directly compared** in Step 1C. Retained snapshot digest equality proves a different property. |
| A cold P6 11 versus C cold P6 16, shared first 11 rows? | **Unmeasured.** Different leading projection geometry is established, numerical effect is not. |
| C cold P6 16 versus D ordinary 16? | **Unmeasured.** Same row widths, distinct scheduling/state plumbing. |
| Differences only in new five rows? | **No** for B versus D: observed mismatch is at position 0. Appended-only metrics and contributors are not isolated. |
| Inputs already differ before quantization/suspected matmul? | **Unknown.** No L0 Engram/attention-input, pre-pack or projected-KV tensors were retained. The packed difference cannot distinguish pre-quantization propagation, quantization, or unexpected live-state update. |

**Minimum upstream-aligned action now: no numerical correction.** Keep the
existing cache/segment/quantizer authorities and fail-closed qualification status.
A deterministic-MLX mode cannot be justified as a fix by copying a CUDA flag, nor
can compacting indices, forcing one-row execution, changing scales or fitting a
tolerance be justified from these maxima. Upstream does not supply an established
11+5/16 exact-state invariant to import. This is not evidence that the difference
is acceptable, or that chunk invariance is impossible.

If further attribution is separately selected, the smallest useful observation
is **L0 prefix preservation and the A/C/D pre-pack boundary**, not another 40-layer
campaign: reuse `observe_case_c_n2` and existing collectors; compare seed L0 window
rows `[0,11)` against live B at 16 in both packed bytes and meaningful decoded
values; compare A/C shared rows and C/D before attributing an append mutation.
Restrict passive captures to L0 Engram output/attention input, `wkv` input Q/DQ and
output, and norm/RoPE output immediately before pack. Use original arrays/shapes,
no substituted values or extra inference calls inside observers. Check observer
restoration, no changed execution plan/cache authority, canonical IDs, absolute
positions, dtypes, model/profile/source identities and numerical input equality
first. Replay a **fixed already captured input** at 11/16 shapes through only the
first implicated operation; otherwise it is propagation, not isolated kernel
shape sensitivity. Quantizer-only leaf replay requires pre-pack values, which
the existing compact receipts do not retain. Assess selected-tensor leaf recovery
before authorizing any full-model load. No diagnostic infrastructure or tensor
fixture was added speculatively here, and none of these proposed observations
was performed.

Any eventual correction needs a demonstrated semantic gap, fixed-input and
state-ownership regression, affected real Case C comparisons and prospectively
justified numerical/continuation evidence; CUDA output equivalence, whole Phase
2/C2 acceptance and production adoption remain separate. **LOCAL_NUMERICAL_GAP**,
Case C **MULTIPLE_MECHANISMS**, numerical PASS and full-discrepancy causal claims
are not supported by this audit.

### Resources, validation and stop

No checkpoint tensors read, model instance, inference worker, CUDA process or
supervised load was launched. Installed payload inspection and planner imports
are not a model load. Consequently the 455.226-GiB preload gate, 64-GiB reserve,
425/430-GiB footprint limits and 30/10-minute execution supervision were not
exercised; no future allocation upper bound is inferred from §12. Point-in-time
pressure was **1 (normal)**, swap use/swapins/swapouts **0**, no port-1234 listener
or competing loader. No unrelated process or service was changed. Review scratch
sources/scripts were task-owned, outside the repository, and removed after use;
no weights, tensors, dependency upgrades or persistent alternate engine retained.

Validation: installed payload inspection **PASS**; planner assertions verify one
11/5/16-row command per layer across all 40 layers; paired-operation and quantizer
source assertions recorded in the receipt. No code changed, so no new inference
or pytest campaign was required. `git diff --check` and
`python tools/check_repository_self_containment.py`: **PASS**. Diff review contains
only this appended section and compact source evidence, with existing report
prefix and all previous receipts unchanged. No CURRENT contract/qualification
fact changed. Local commit SHA reported separately; no push or deployment.

**Smallest unresolved issue:** whether seed L0 prefix bytes survive in live B
exactly as source specifies, and, if so, which pre-pack operation first separates
cold 11-row from cold 16-row prefix production. Cause remains unresolved, not a
new acceptance condition. **Stop after Step 1D; no follow-on task started.**

## 14. Step 1E — Layer 0 prefix preservation attempt (2026-10-25)

**Disposition: BLOCKED / PARTIAL / UNRESOLVED.** The single real-model loading
attempt was stopped by the task supervisor **before `backend.load()` returned**.
No Case C forward, capture/restore, append, cold control, pre-pack observation or
leaf replay ran. There is **no new numerical result**, and no numerical PASS,
PREFIX_PRESERVED, PREFIX_MODIFIED, COLD_SHAPE_DIFFERENCE,
P6_ORDINARY_DIFFERENCE or LEAF_ATTRIBUTED claim. The unanswered numerical
questions from §13 remain unanswered. No automatic model reload was performed.

### Review, identities and canonical preparation

Started at the expected clean `fee0ef98f52168feb819d75f6b47e5d0d9b1da73`,
`Multimodal-Runtime`, one worktree. Point-in-time process inspection found the
existing coding agent but no other coding/model worker or port-1234 endpoint.
Read documentation policy, upstream-first/sequencing, Phase 2/C2, ordinary MTP,
verification, §§10–13 and Step 1C/1D structured evidence. Reused Step 1D's official
DeepInfer audit at **`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`**; no new source
question required repeating that audit. Its lack of an established bitwise
11+5/16 invariant neither accepts arbitrary divergence nor establishes a defect.

Before designing hooks, inspected checkout `_start`, paired clone/capture/restore,
P6 append/planning/publication and block-runner execution, and actual installed
oMLX `language.py`, `cache.py`, `quantization.py`, `engram.py`, activation Q/DQ,
projection dispatch, HC/pre-normalization, norm/RoPE and ordinary `_forward`.
The installed Target path remains **oMLX**, not first-party model execution.
Installed `_start`, P6 and block runner equal checkout; paired module differences
remain budget/lookup additions outside the capture/restore operations reviewed
in §13. No installation or production numerical code was modified.

[Identity/preparation receipt](phase2-reference-step1e.json) records actual origins,
installed numerical source hashes, checkout hashes, package direct-URL identities,
build provenance and full installed checkpoint admission. Interpreter was
`/Volumes/SDXC-512/ds41f-1017/bin/python -I`; checkpoint was the required
KIOXIA-PRO-2 `DeepSeek-V4.1-Flash-1017`. Admission **PASS**, identity SHA256
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`,
profile **`flash-1017-pre-woa-fp8-v1`**, native GLM. Installed ds41f 0.23.0/build
`aa8c5717b42aa0a216fef3134e72c6db7ca49287`, oMLX 0.7.0 with the four admitted
overlays, MLX 0.32.2, mlx-lm `0.31.4.dev132+g94cdcae13`, Recipe 0.2.0/native
revision `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus semantic patch.
The unchanged-file checkpoint-verification-cache caveat still applies; this is
not a new independent rehash of all 48 weight shards.

Installed Recipe prepared the **exact Step 1C bodies** before the load gate:

| Request | Actual N / N-2 | Actual ordered little-endian int32 SHA256 |
| --- | --- | --- |
| `Explain immutable checkpoints in one short sentence.` | 13 / 11 | `8ce125c769d4a95f97bceac24a7c6aa2f8bbf36a1f0c4ff1d1c4a9ab82d79fb2` |
| `Explain immutable checkpoints in one short sentence. Then state one benefit.` | 18 / 16 | `9059c79e53e40e13f60272e37226a6f5d99f2d5d6f53879d4dbaff1ff91c4c1b` |

Temperature 0, max_tokens 1, reasoning none; exact IDs matched Step 1C, common
prefix length 11 and planned append [11,16). No handcrafted IDs or altered text.
An earlier **zero-load preflight** used the wrong scratch-driver protocol key
`chat`; installed `prepare_request` rejected it. The failure and cleanup are
retained in [preflight evidence](phase2-reference-step1e-preflight-error.json).
Correcting it to the inspected `chat_completions` key preceded successful
canonical verification and the sole weight-loading attempt.

### Focused diagnostic preparation, not real-model qualification

Extended **the existing `observe_case_c_n2`**, opt-in only, instead of adding a
second execution owner. The missing interface was a live L0 observation before
`PairedCheckpoint.capture`, and B0 before `_start`; the former snapshot digest
cannot supply these. Prepared mode copies packed bytes and meaningful FP32-decoded
values from live cache boundaries A/B0/B1, records primitive layout, absolute row
mapping, metadata/offsets and MLX memory, and compares all three prefix relations.
A restore difference stops before append; a live append difference stops before
cold arithmetic attribution. With exact preservation it reuses the same `_start`
and ordinary-reference interfaces for C/D, with genuine capture/restore and
separate P6/ordinary schedules. Existing default Case C behavior remains intact.

Narrow temporary installed-oMLX hooks retain original L0 Engram input/output,
Engram projection input/output, attention input, projection post-input-Q/DQ/output,
KV norm output and immediate pre-pack input. They perform **no tensor operations,
slicing, host reads, eval or synchronization inside numerical calls**, substitute
no values, and restore methods through `ExitStack`, including exceptions.
They extend graph lifetime; compiled interiors and the internal Engram gate are
not separately traced. Additional gated-contribution instrumentation would only
be justified if these initial boundaries left that specific question unresolved.
There is no claim that their full-model scheduling neutrality was experimentally
validated: **the hooks were never installed in the real worker**. Duplicate L0
calls fail rather than silently merge shapes. Post-run review removed an
unnecessary planned all-cache eval at the ordinary measurement boundary; executed
and final diagnostic hashes are distinguished in the receipt.

These helpers are diagnostic-only, bounded to short singleton non-wrapped L0
windows; they are removable after investigation. They create no inference engine,
cache/session authority, quantizer, arithmetic correction or acceptance threshold.
Retained handles increase lifetime/resource costs; the attempted run did **not**
reach checkpoint/candidate/cold/reference coexistence, so those costs remain
unmeasured. Model-free tiny controls are measurement-integrity evidence only.

### Interruption, resources and cleanup

[Run log](phase2-reference-step1e-run.log) and
[supervision evidence](phase2-reference-step1e-resources.json) show one worker
**18316**, one invocation of the existing model loader, **zero completed loads**
and **zero numerical executions**. Supervision sampled about every two seconds,
retaining compact samples about every ten; libproc rusage-v4 current/lifetime
footprint used SDK offsets 72/240. Thresholds stayed at preload 455.226 GiB,
reserve 64 GiB, stop footprint 425 GiB before ceiling 430 GiB, normal pressure,
no swap-use increase/swapouts, no competing inference/1234 endpoint, and
1800-second total/600-second no-progress limits.

**The stop was a supervisor false positive caused by this task**, not evidence of
a competing inference loader: while loading, the assistant launched a short
read-only Python process **24444** to print resource JSON. The conservative
process-name detector treated every non-descendant Python process as competition;
it did not recognize this task-owned reader. At the next sample it sent SIGTERM
to **only the owned model worker**. The worker exited **-15**, verified absent;
no reload followed. No unrelated service or process was terminated or changed.
A future supervisor must recognize explicitly owned telemetry processes (or avoid
launching them), without weakening detection of actual competing model activity.
No supervisor framework or admission change was added to the repository.

- Pre-load gated available memory **493.071 GiB**; health-sampled minimum reserve
  **241.216 GiB**. Available includes reclaimable pages, not only free pages.
- Sampled lifetime physical-footprint high-water **252.313 GiB** during incomplete
  loading, below 425 GiB. This is not the unsampled final allocation peak or a
  completed-model memory bound. No post-load MLX active/cache/peak sample exists.
- Attempt elapsed **92.44 s**, including admission/gate wait; preceding zero-load
  preflight **35.95 s**. No time, pressure, reserve or swap threshold fired.
- Pressure **1 (normal)**; swap use, vm_stat swapins/swapouts **0** throughout.
  No LM Studio/1234 endpoint or actual competing model activity was observed.
- SIGTERM bypassed Python `finally`, so model-close/allocator-clear metrics do
  **not** exist for the interrupted worker. OS process exit reclaimed its resources;
  final host available memory **493.337 GiB**. Hooks had not been entered and no
  reusable pair, native generation owner or published cache existed to retire.
  Supervisor/worker/reader exit and absence of port 1234 were rechecked after tests.
  Task-owned scratch scripts/gates were removed after retaining hashes; no tensors
  or extra model assets were retained.

### Validation and numerical stopping boundary

Before loading: affected collector/adapter/P6 tests **41 passed, 12 subtests
passed**. Final [focused validation](phase2-reference-step1e-tests.log) additionally
covers the optional live-boundary lane and restore-before-append failure guard.
Controls detect an intentionally changed retained packed byte and its decoded
value, a changed pre-pack value, wrong absolute-position mapping, modified restore,
and modified live append; verify genuine clone/restore isolation, C/D sequencing,
no cold controls after prefix modification, and exact method restoration on
success/exception. These tests load **no model weights**. Final command:
`tests/test_prefill_numerical_reference.py tests/test_prefill_fp8_mlx_p6.py` —
**44 passed, 12 subtests passed in 41.97 s**. `git diff --check`: PASS.
`python tools/check_repository_self_containment.py`: PASS (35 current owners,
153 Markdown files, 585 live links). Final diff review found only the existing
diagnostic tool/tests and appended evidence: no production changes, duplicated
execution authority, historical evidence edits or unsupported acceptance claims.
Validated changes/evidence committed locally; SHA reported separately. No push
or deployment.

A/B0/B1 packed/decoded prefix comparisons: **not executed**. A/C and C/D:
**not executed**. Actual real-model operation shapes, first differing values,
positions/dtypes/metrics and fixed-input leaf attribution: **not observed**.
Only §13's source/planner shape evidence (11/5/16 rows) remains; it is not a new
numerical measurement. Cases A/B/C and Step 1D remain unchanged historical
receipts. No production defect, tolerance, correction or numerical acceptance is
established. C2, P5, continuation, production reuse and Harness remain unqualified.

**Smallest remaining numerical question:** do the original eleven L0 packed KV
rows survive genuine live restore and append exactly? The operational prerequisite
for any separately authorized attempt is avoiding the telemetry false positive;
this task does not start another attempt. **Stop after Step 1E.**
