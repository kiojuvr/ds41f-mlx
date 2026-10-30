# Official Files quota recovery — scoped PASS

Source-only, finite model-free protocol qualification. Production unchanged.
No real-model, Windows, full CLI conversation, endurance or promotion claim.
Earlier evidence in the parent directory is preserved.

## Verified identities and preflight

- Clean `master`, confirmed HEAD `30b0befa4ef0d7daae84c527bd8bda8b45500bbb`.
  No branch/history operations, package installations or unrelated changes.
- Installed official `/Users/kioju/.local/share/deepseek-harness/node_modules`:
  CLI, DeepSeek provider and compaction packages report **0.2.1-rc.2**.
  Inspected the actual provider `lib/index.js`, including Files transport,
  parsing, classification, cancellation and `DeepSeekFileStore` recovery.
  Provider SHA-256:
  `eba35025ed1eedda89c51f2da46a3a968f7be1d59719f2ceeadfc4518dff37d6`.
- Independently fetched and read both complete official source files from
  `packages/llm/llm-deepseek/src/{files-api,file-store}.ts` at release revision
  `687f282e8625c6b8d11ff58211a90524be2e4317` from
  <https://github.com/deepseek-ai/deepseek-harness>.
  Source SHA-256 respectively:
  `0f38622c30c85266bf17db38e0171a130bbee5ce0404327a1de949b1393eca26`,
  `3a361dcc47ab661a89f8a44e65e607f8d4e56a073cfce9cde8e4121e43bd109e`.
  These contract/recovery functions agree with the installed bundle; a package
  version alone does not establish its build Git SHA. An older retained source
  directory lacked these files and was not used as authority.
- Production `/Volumes/SDXC-512/ds41f-1017`: ds41f-mlx 0.23.0,
  Recipe 0.2.0, multipart 0.0.32. Read-only source hashes distinguish installed
  Production from current checkout: Production still has the earlier Files
  implementation (including the former count policy), and earlier capacity/
  multimodal files. `server.py`, `mtp_public.py`, `ordinary_admission.py` matched
  the checkout baseline. No Runtime/Harness process appeared in initial process
  inspection. No service restart, model loading or checkpoint scan.
- Tests use the existing disposable `/private/tmp/ds41f-step4ar1-tests`, Recipe
  0.2.0, multipart 0.0.32, pytest 9.1.1, uvicorn 0.54.0; not checkout's older
  Recipe environment and not Production. Installed native payload authentication
  remains the existing test fixture's responsibility.
- [Identities](identities.json), [initial read-only hashes](identities-before.txt)
  and [final unchanged check](unchanged-check.json) retain exact evidence.

## Exact official contract / original incompatibility

1. Non-2xx responses become `DeepSeekFilesError`; `code`, `type`, `message`
   strings form `detail`. Quota classification requires this error class and
   `quota|storage|stored files|file count|too many files` (case insensitive).
   Runtime's existing 429 `file_storage_quota` / `file storage quota exceeded`
   already matches. No error semantics correction was needed.
2. List sends `GET /v1/files?limit=1000`, then
   `GET /v1/files?after_id=<previous last_id>&limit=1000` through the configured
   API root, with official attribution/Anthropic headers and redirects refused.
   The parser requires `data` array and boolean `has_more`; null/missing
   `first_id`/`last_id` are omitted internally, otherwise they must be strings.
3. Each metadata object requires nonempty string `id`, `type="file"`, string
   `mime_type`, nonnegative safe integer `size_bytes`, nonempty `filename`, and
   **string** `created_at` parsable to a nonnegative whole-second epoch. Runtime
   reuses upload's UTC RFC3339 whole-second string, not an integer timestamp.
   No expiry metadata is required by listing. Upload also checks returned byte
   length, and computes its reuse deadline from creation plus requested expiry.
4. Ownership is exactly case-sensitive `filename.startsWith("dsh-")`.
   Harness scans **all pages**, stable-sorts candidates by ascending creation
   time after each page and retains at most its cleanup batch (default 100,
   configured 1..1000). Equal-second timestamps preserve traversal order.
5. Harness deletes selected IDs using `DELETE /v1/files/{id}`. It requires the
   same `id` and `type="file_deleted"`. Existing Runtime deletion already fits.
6. An upload quota failure invokes this cleanup once. Zero deletions rethrows
   the original error without retry; positive deletions permit **one** upload
   retry. Retry failure propagates without another cleanup. List/delete/abort
   failures propagate too; no partial success is represented as an image.

Original blocker: ordinary Runtime rejected collection GET with 404, and its
boundary rejected all query strings. Thus existing upload/delete were insufficient.
Individual retrieve/content endpoints are not used by this quota workflow.

## Minimal implementation / bounded ownership

- `image_files.py`: share the existing upload metadata formatter; add strict
  pagination parsing and one listing method over the **same mapping and RLock**.
  No index, filesystem, second store, tenant state or persistence.
- `server.py`: register only ordinary MTP's collection GET; perform listing in
  the existing threadpool so bounded scans do not run on the event loop.
- `mtp_public.py`: admit exactly this GET and exempt only it from generic query
  rejection. The listing parser rejects unknown/duplicate/malformed parameters.
  All other routes retain query rejection and ingress policies.

Ordering is upload-publication/insertion order, not clock order; existing atomic
mapping publication preserves it. `after_id` is exclusive; `limit` is 1..1000
(default 1000). Wire parameters are the exact bounded ASCII grammar emitted by
Harness; percent-encoded aliases and unsupported sort/filter options are rejected.
At most 1000 metadata dictionaries are built, no image-byte copies or whole-store
metadata sorting. Filenames remain bounded to 512 characters, so response size is
bounded independently of file count. The existing sweep's temporary key list is
still bounded by retained store allocations. Listing is O(store entries), with
page-sized metadata plus the pre-existing O(entries) sweep allocation; not an
RSS reservation or zero-allocation claim.

Expiry and selection occur under the existing lock. Empty/terminal pages have
null cursors and `has_more=false`; a full page only has `has_more=true` if an
additional live entry exists. Stable stores traverse without skips/duplicates,
including more than 1000 files. There is no cross-request snapshot: deletions
before a surviving cursor and appends after it are safe; a deleted/expired/unknown
cursor fails explicitly with 400 `invalid_file_cursor`. A listed file concurrently
deleted/expired before DELETE gets the established 404. Harness may fail that
recovery; Runtime does not hide the race or retry independently.

Preserved: no default 128-file count cap, optional lowered count policy, **64 MiB
retained-allocation budget including metadata/mapping overhead**, atomic upload
publication, immutable request-held snapshots, process-local lifetime and expiry.
Only Harness chooses quota victims and whether to retry. Listing has the same
trusted single-operator deployment boundary as upload/delete; it adds no auth claim.

## Executed acceptance

[Official protocol log](official-tests.log): **22 passed**, no skips.
The new suite runs the actual installed, unmodified `DeepSeekFilesClient` and
`DeepSeekFileStore` against the modified Runtime app via real uvicorn loopback
HTTP and native Node fetch/FormData. Only the upload index is injected as a
process-local test double, avoiding writes to the operator's Harness state.
This qualifies official recovery transport/policy, not the complete CLI agent.

- Default budget: five valid uncompressed 2048x2048 PNGs (~12 MiB each) plus
  unrelated tiny file; the next upload exhausts retained storage. Official error
  detail matches quota classification. Actual sequence is POST 429 → GET 200 →
  DELETE 200 → POST 200, no limit increase, unrelated file preserved.
- Multi-page recovery: 1001 unrelated files plus two owned files. Oldest owned
  creation time is deliberately on page two, later in insertion order. Actual
  limit=1000 and after_id traversal selects/deletes that older ID only; retry
  succeeds under a lowered test-only retained budget. `DSH-` and embedded `dsh-`
  near-matches are not selected.
- None owned: POST 429 → GET 200; no DELETE and no second upload.
- Insufficient reclaimable bytes: one tiny owned file deleted, larger valid
  upload retry remains 429; exactly two upload attempts and one list/delete pass.
- Cancellation before work sends nothing; cancellation after recovery listing
  sends no deletion/retry. A deterministic competing DELETE after listing yields
  official DELETE 404 and aborts recovery without retry.
- HTTP pagination traverses 1005 stable entries with exact metadata and boundaries,
  one-entry and 1000-entry limits, empty/terminal pages, invalid queries/cursors,
  additions/deletions between pages and expiry. Concurrent listing/deletion is
  synchronized and returns only consistent metadata pages.
- Reclaimed IDs cannot be resolved by new snapshots or new Messages requests
  (404); already acquired immutable snapshots retain their original bytes.
  No inference request is prepared for missing IDs.

[Regression log](regressions.log): **158 passed**, no skips. Existing Files,
Messages/Chat Recipe boundaries, cancellation, restore ownership, ordinary
context/body and documentation authority tests remain active. These exercise
byte/hash identity, message/tool-result ordering, historical image reuse/fresh
admission and missing-ID fences without weights or numerical execution.

Storage metrics are printed per official scenario in the protocol log. Default
recovery ends with 6 entries charged 62,948,565 bytes under the unchanged
67,108,864-byte budget. The 1003-entry multi-page case ends at 612,717 bytes
under its test-only 612,826-byte budget. Successful recovery remains within its
unchanged configured allocation budget. Deletion
removes the stored record immediately, but snapshots intentionally retain bytes
until their request owner releases them. Existing weakref/expiry/clear tests pass;
fixture shutdown clears each process-local store and closes the isolated server.
No persistent Files artifacts, global request-memory reservation, RSS plateau or
leak-free endurance claim is made.

## Protected invariants and exclusions

No changes to message resolution, multimodal preparation, source hash/identity,
Target/DSpark cache ownership or continuity, model execution/numerics, canonical
settlement, Tool-effect delivery, compaction or image offload. Body/image-byte,
pixel/patch/context limits are unchanged. Reclaimed IDs fail closed; listing cannot
trigger historical replay or substitute another file. No native request preparation
changes requiring a real-model qualification were introduced.

Still unsupported: individual metadata retrieval, file content/download,
OpenAI-purpose variants, arbitrary documents/URLs, general filesystem management,
accounts/tenancy, persistent IDs and cross-process stores. General Harness envelope
compatibility remains PARTIAL because independent Runtime limits still apply.

**PASS for the installed official Harness-managed quota-recovery sequence against
the existing bounded ordinary Files store.** Production deployment and real-model
qualification are not part of this result. Stop here; no other soft-cap work.

Reproduce:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python -m pytest \
  tests/test_files_quota_recovery.py -q -s --disable-warnings
PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python -m pytest \
  tests/test_image_files.py tests/test_recipe02_messages_serving.py \
  tests/test_recipe02_chat_boundary.py tests/test_multimodal_cancellation.py \
  tests/test_multimodal_restore_ownership.py tests/test_production_context.py \
  tests/test_documentation_authority.py -q --disable-warnings
```

Set `HARNESS_NODE_MODULES` to another installed official node_modules directory
when needed; missing official client is an explicit test skip, not qualification.
