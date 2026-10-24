# Harness image capacity alignment — PARTIAL

Scope: `Visual-Feedback-Loop`, base `c85d490`. Source implementation and model-free
qualification only. No deployment update or new real-model qualification.

## Reference and effective configuration

Installed official packages under
`/Users/kioju/.local/share/deepseek-harness/node_modules/@deepseek-ai` report
Harness **0.2.1-rc.2**. Executing the installed official `Config` volatile getters
confirms **600 occurrences, 134217728 raw Files bytes, 20971520 represented inline
image bytes, count offload quantum 20**. The qualified Q2
[composed configuration](../visual-feedback-loop/phase-4-q2-evidence/composed-config.yml)
contains no image-budget overrides in its `llm-deepseek` route and mounts both
`compaction-basic` and `compaction-image-offload`. This verifies the effective
qualified reference composition, not an arbitrary live client's settings.
Harness budgets are configurable; Runtime's count ceiling is 600, not a promise
to support raised upstream budgets or unconditional model execution.

[Executable official checks](../../tools/check_harness_image_envelope.mjs) use the
installed code, not a copied offload algorithm. [Receipt](official-envelope.json)
includes source hashes. Checks: 600 fits; 601 requests offload of 20 oldest
occurrences; raw-byte equality fits; one-byte excess offloads; base64 equality and
excess are checked; a 16 MiB raw image fits Files budget but exceeds the 20 MiB
represented inline-image fallback budget. Repeated IDs count as occurrences.

## Retained evidence and safety decision

Reviewed [Q1/Q2](../../docs/flash-1017-visual-endurance-envelope.md),
[F0](../harness-files/gate-f0-report.md), the
[cached-image admission decision](../visual-feedback-loop/cached-image-admission-decision.md)
and its upstream audit. These investigations were not repeated. Q1 retains six
observed images and seven model-free at measured screenshot geometry. Q2 retains
four task screenshots, partial grounding and paired continuity. F0 retains two
Files images and one historical reuse; none qualifies 600-image model residency.

The prior decision identifies two separate obstacles: pre-restoration quadratic
key allocation, and enlarged queue/restore/capture coexistence. The first can be
improved compatibly at 600. The second is **not established by synthetic tests**:
32 waiting requests plus one active request can coexist, preparation serialization
is not a queue memory reservation, 18 retained pair slots are not a byte budget,
and restore/append/unpublished prompt/final capture peaks cannot assume zero-copy
sharing. Existing context finiteness alone is not physical-memory qualification.

Therefore **retain the production all-history pixel/patch guards**. Do not move
those checks to fresh-only workload admission on the basis of model-free tests.
This is the concrete stopping boundary, not a new speculative architecture or an
arbitrary image-count target. No 5,698-image workload was run or optimized.

## Implemented improvements

- Ordinary image paths use **600 occurrences** as the Harness request-count
  compatibility reference. Files occurrences and the independent **128 distinct
  storage slots** are no longer conflated. Physical guards can reject much earlier.
- Every historical occurrence still receives complete container/pixel validation,
  digest/span/grid accounting, and per-image, encoded-byte, pixel/patch and context
  checks. No cached image receives a security or physical-admission exemption.
- Explicit `validate_workload(actual_frontier)` independently checks actual fresh
  pixel/patch work **after validated paired restoration and before native start**;
  encoder entry rechecks it. It never authorizes historical replay or weakens the
  earlier all-history guard. Capability metadata explicitly identifies the scope
  and advertises `historical_discount_qualified: false`.
- `image_ranges` shares immutable image-key leaves across required exact prefix
  tuples instead of rebuilding their grid/key tuples at each prefix. Published
  key values and exact prefix-selection semantics are unchanged. At **600** the
  measured construction peak falls from **24,586,916 to 1,469,596 Python bytes**.
  Diagnostic single-run times: 88 ms versus 0.52 ms; not a throughput guarantee.
  [Measurement](image-keys.json). Quadratic tuple references remain because the
  upstream interface consumes exact prefixes; no hash/schema/cache rewrite.
- No second cache, session, storage-history or compaction owner. Context/output,
  capture extendability, publication only after canonical settlement, queue
  admission and eviction rules are unchanged. No body/storage/workload ceiling
  was raised. No production package was replaced.

## End-to-end compatibility assessment

| Layer | Result |
| --- | --- |
| Harness request count and byte semantics | Official model-free boundary checks PASS. Local 600 count reference is independent of physical admission. |
| Runtime transport | **PARTIAL**: existing 16 MiB entire JSON body cannot cover Harness's 20 MiB represented inline-image budget plus history/JSON overhead. |
| Files | **PARTIAL**: 16 MiB/file, 128 retained files, 64 MiB storage and 64 MiB encoded occurrence residency do not implement a general 128 MiB request store. Base64/Recipe/parser copies still coexist. |
| Historical validation | PASS within existing physical envelope: every occurrence fully decoded and identity-accounted, including repeated bytes. |
| Fresh encoder work | Already skips fully covered historical spans; explicit post-restore work checks added. A cold miss must fit physical admission; uncovered historical images reject rather than replay. |
| Cache/context/queue/capture/memory | Existing guards retained. 600-image physical admission/execution **not qualified**. At minimum tiny-image geometry, 24 occurrences fit 36,504 patches, 25 reject. At Q1 screenshot geometry, seven fresh/all-history images fit and eight reject. |
| Official offload / compaction | Official projection works upstream; general paired continuity after canonical change remains unsupported. |
| Real-model qualification | Unchanged Q1/Q2/F0 scope. No new model run, numerical promotion, Windows or maximum-capacity claim. |

### Error recovery and canonical changes

Installed `dsh-llm-deepseek/lib/index.js` calls `assertImagesFit` before provider
execution (raw budget), then again before inline fallback (base64 budget). It
raises `IMAGE_OFFLOAD_REQUIRED` with `offloadImages`. Installed
`dsh-compaction-image-offload` handles exactly that code: records durable
`image/offload` occurrence selections and asks the existing agent/summary seam to
retry. The executable check mounts the actual plugin, logs its decision, executes
its actual projection, and confirms 20 oldest images become text without mutating
original history. `INVALID_REQUEST`/HTTP 400 does **not** invoke that recovery.
Provider classification maps 400/413 to `INVALID_REQUEST`; File-ID rejection can
invalidate/reupload a mapping, but does not imply canonical image offload.
Storage quota errors likewise do not promise image-history recovery.

Offload replaces image tokens with text and shifts later absolute identities.
Official summary compaction similarly replaces a canonical prefix through its
existing summary/surface mechanism. No old pair may be relabeled with the new
canonical tokens. Exact-pair selection may reuse only a genuinely matching
prefix; if it cannot cover surviving historical images, admission rejects before
native start. Logical sessions additionally require their exact retained canonical
lease. If no historical images survive, existing fresh/text miss behavior applies.
An explicit fresh conversation using the client's existing history mechanisms can
operate within current limits; automatic paired reconstruction is not implemented.
No automatic generation resend, silent deletion, historical re-encoding or
unsafe tool-effect retry was added.

## Validation and isolation

Fresh Python **3.13.15** venv: `/private/tmp/ds41f-visual-capacity-test`. It received
an independent copy of the installed Flash-1017 dependency/native/tokenizer closure
(`lib` and `share`). Original deployed Runtime and model assets were not changed
or restarted. Initial wheel-only inspection correctly failed on missing installed
dependencies; supplying the complete closure fixed packaging rather than bypassing
checks. Conflicting bootstrap pytest dependency metadata was removed only in the
disposable venv. Isolated `python -I` `inspect_payload()` passes
([receipt](installed-admission.json)); pytest's normal installed-payload fixture
also passes. This authenticates the dependency installation for source tests,
not a release/seal of this checkout.

[Focused tests](model-free-tests.log): **206 passed**, 289 existing deprecation
warnings. Coverage includes native Chat/Messages preparation, Files quota/body/
expiry/concurrent snapshots, geometry against real Pillow, small random MLX Vision
adapter equality, paired hits/misses/target-only hints, changed identities, partial
spans, cancellation/cleanup, historical-origin policy and canonical settlement.
New larger **valid** histories use 23 retained + one fresh tiny image, and reject
600 at the unchanged physical guard. Actual 600-key compatibility is tested
against every previous prefix value, including shared-leaf identity. A lowered
test-only fresh budget exercises pre-native rejection and source release without
raising any production history guard. Whole test-process peak footprint was
2.26 GB; this is not Runtime/model memory qualification.

[Repeated valid-history preparation](history-residency.json): six iterations of
24 occurrences, releasing sources and collecting request geometry. Post-cleanup
Python memory is stable (see raw samples), with no unexplained growth. CPU
preparation only, not restored/captured MLX KV residency. No real-model loading
was attempted: retained realistic visual-agent evidence already exists, and the
unqualified residency boundary is intentionally not relaxed.

Reproduction in a correctly installed isolated dependency environment:

```sh
python -m pytest tests/test_harness_image_capacity.py tests/test_metadata_first_vision.py \
  tests/test_image_files.py tests/test_expanded_vision_capacity.py \
  tests/test_production_mtp.py tests/test_visual_feedback_loop.py \
  tests/test_multimodal_contract.py tests/test_multimodal_restore_ownership.py \
  tests/test_multimodal_cancellation.py tests/test_runtime_capacity.py \
  tests/test_documentation_authority.py -q --disable-warnings
node tools/check_harness_image_envelope.mjs /path/to/official/node_modules
```

**PARTIAL:** compatible allocation/accounting improvements implemented; default
Harness request envelope is **not** an end-to-end Runtime execution guarantee.
Stop at the demonstrated all-history physical-admission, transport/storage and
canonical-prefix recovery boundaries. No replacement architecture, push or
automatic follow-on campaign.
