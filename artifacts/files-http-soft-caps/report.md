# Files / ordinary HTTP soft caps — PARTIAL overall

**PASS: scoped source correction and model-free acceptance. PARTIAL: full official
Harness envelope compatibility.** No Production installation, model execution,
Windows test, endurance claim or follow-on soft-cap campaign.

## Verified authorities

- Clean `master`, HEAD `ca1f738c4f02586ac9cbd8c68cc08e93a1a4308c` before edits;
  remote `origin/master` and remote HEAD advertised the same commit. No branch or
  history changes. No pre-existing dirty files.
- Production `/Volumes/SDXC-512/ds41f-1017`: ds41f-mlx 0.23.0, Recipe 0.2.0,
  python-multipart 0.0.32. Installed provenance reports source `69689d514fc5acb2cbe229f8c863da1e0a855e4a`
  plus subsequent in-place source promotions. Read-only hashes confirm installed
  `image_files`, `mtp_public`, `capacity`, `ordinary_admission` and `multimodal`
  exactly match this task's baseline and were unchanged on final recheck.
  Provenance alone is not its current source identity.
  Recipe source authority: `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus the
  existing semantic-boundary patch. Package/native/manifest hashes:
  [identities](identities.json). No checkpoint scan or native/model allocation.
- Installed official Harness CLI/provider/compaction packages: **0.2.1-rc.2**.
  Provider bundle SHA-256 `eba35025ed1eedda89c51f2da46a3a968f7be1d59719f2ceeadfc4518dff37d6`
  matches the retained W1 identity. Official release tag independently resolves
  to `687f282e8625c6b8d11ff58211a90524be2e4317`; tagged `file-store.ts` was fetched
  and compared with installed behavior. Installed manifests do not prove a build Git SHA.
  Remote HEAD at investigation: `ce7073d51254c26cd63b22a60835b8f948deb447`, not
  substituted for the installed workflow.
- DeepInfer Production architecture reference: `8b16149741a4935e1dec8cea27853f273b374c5b`.
  Read official `deepinfer/engine/llm_engine.py` and
  `deepinfer/ve/processor/image_processor.py` at that revision (hashes retained).
  Engine explicitly has no online-serving wrapper: cache profiling, block/state
  managers and Scheduler own execution resources, not a Files store or HTTP body.
  Its lazy image tensor loader likewise does not reserve our Python HTTP/storage
  allocations. Remote HEAD and clean local reference checkout:
  `553741557a57df44a46fb3ab72b6aafeaf945337`. These are references, not an installed
  DeepInfer distribution; do not import its backend memory assumptions into MLX.
- Checkout `.venv` has Recipe 0.1.1: unsuitable for this test. Used existing
  disposable `/private/tmp/ds41f-step4ar1-tests`, Recipe 0.2.0; normal test fixture
  authenticates its installed payload in an isolated subprocess. No installs or
  dependency changes. An older disposable environment lacked runnable pytest;
  it was not repaired. No running Runtime was restarted or modified.

## Limits and exact rejection ownership

| Candidate / decision | Rejection path and reason |
| --- | --- |
| **128 retained entries — REMOVE default; REPLACE with retained allocation admission** | Former `ImageFiles.upload`: `len(_files) >= limits.count` -> 429 `file_storage_quota`, even with only ~14 KiB of 129 tiny PNG payloads. Independent of request occurrences and official policy. Optional explicit positive `FileLimits.count` remains an operator restriction, not a deployment ceiling. |
| **64 MiB persistent store — REPLACE payload-only accounting; RETAIN finite budget** | `ImageFiles.upload` -> same 429, now before atomic publication if `retained_bytes(candidate)` exceeds `FileLimits.storage_bytes`. Count removal must not permit arbitrarily many metadata objects under a payload-only budget. The separately named `FILE_STORAGE_BYTES` is no longer borrowed from a request-image constant. |
| **16 MiB ordinary JSON — RETAIN in existing buffered implementation** | `LocalBoundary.__call__`: excessive Content-Length -> 413 `body_limit` before reading; `bounded_receive`: excessive chunked/actual bytes -> same error before app buffering continues. `ordinary_admission.admit_structure` separately rejects oversized direct callers (`body limit exceeded`). `prepare_request`/Recipe create JSON/string/native copies before Scheduler admission. Exact 16 MiB is local, not an official policy or token estimate; removing it without replacement would remove the only raw-body bound. |
| **16 MiB encoded file/image — RETAIN** | Multipart boundary uses `UPLOAD_BODY_BYTES = MAX_IMAGE_BYTES + 64 KiB`; `upload`/`parse_upload.part_data` return 413. Shared `validate_encoded_image` independently rejects excessive bytes, invalid containers, MIME mismatch, frames and excessive pixels. Uploads are serialized, fully buffered and decoded; the byte guard is not made redundant by pixel limits (encoded metadata/container overhead). |
| **64 MiB encoded occurrences/request — RETAIN** | `resolve_messages.visit` rejects `aggregate encoded image byte budget exceeded` **before** each File-ID base64 copy; `prepare_multimodal` independently enforces aggregate bytes for all inline/resolved occurrences before queuing encoded sources. Repeated IDs count repeatedly, even if immutable store bytes are shared: base64, expanded JSON and Recipe parser copies are not shared. This is transport/queued-source residency, not persistent storage or fresh Vision work. |
| **Full Harness-sized ingress/store and quota-list parity — DEFER** | No existing Scheduler/KV/resource-admission lease owns pre-tokenization Python bytes or persistent Files allocations. A broad host-memory reservation or storage redesign needs separate qualification. Do not replace these guards with larger guessed constants or implement list/reclamation here. |

All four local numerical sizes are Runtime policies, not official configurable
Harness budgets. Installed `dsh-llm-deepseek` has configurable request defaults:
`maxImagesPerRequest=600`, `maxRequestFilesBytes=128 MiB` (raw per occurrence),
`maxInlineRequestImageBytes=20 MiB` (represented base64). Its Files client has
128 MiB upload / 10,000 stored files / 25 GiB provider constants, but its actual
image-store admission is **32 MiB/image**. None describes our persistent store.
Expiry (3600..2592000 seconds), refresh margin and quota cleanup batch are separate
Harness lifecycle policies. [Executable official checks](official-envelope.json)
exercise equality/excess, repeated occurrences and actual offload projection.

## Recovery and interaction

Installed provider `prepareImages` / `assertImagesFit` checks raw occurrences
before Files resolution and represented bytes before request-wide inline fallback.
`DeepSeekFileStore.ensureUploadedOnce` reuses a durable variant mapping, or uploads;
quota-classified upload errors invoke `reclaimOldestOwned`, which calls Files
**list**, deletes oldest Harness-owned entries and retries once. Runtime GET
`/v1/files` returns 404: this recovery cannot reclaim our quota. The existing
request-wide Files failure can fall back to inline, which must independently fit
both Harness's inline budget and Runtime's smaller whole-JSON bound. For example,
a 16 MiB raw image fits Harness's raw policy but exceeds its 20 MiB represented
inline policy. Unrelated text/schema/history bytes also count toward Runtime ingress.

Missing/deleted/expired IDs return 404 `file_not_found` with the exact ID;
`providerRejectedFileId` / resolver invalidates the matching mapping for bounded
reupload. That mechanism is not quota, body-limit or image-residency recovery.
Runtime ordinary 400/413 is classified `INVALID_REQUEST`; installed offload plugin
handles only `IMAGE_OFFLOAD_REQUIRED` plus selections. Thus an otherwise valid
upstream request can receive a rejection that does not trigger image offload.
No error-code forgery, inference retry or hidden reconstruction was introduced.

Offload/compaction removes canonical image occurrences upstream but does **not**
call Runtime deletion in the installed request/offload path; successful Files
uploads remain until explicit DELETE, expiry/sweep or app shutdown. Explicit
upstream file-store release uses DELETE; quota cleanup additionally requires list.
Repeated requests/IDs do not create new Runtime files unless uploaded again.
Snapshots acquired before delete/expiry remain valid. Paired-KV continuity and
current-projection Messages reconstruction remain independently owned and unchanged.

## Exact correction and safeguards

Only `ds41f_mlx/serving/image_files.py` changes Runtime behavior:

- Default count is `None`; positive explicit operator count remains supported.
- Retained admission sums `sys.getsizeof` for the mapping, ID, stored record,
  record dictionary and its field values, charging shared values conservatively
  per entry. Encoded payload is included. This bounds entry metadata as well as
  bytes, without a competing store, allocator or lifecycle owner.
- Under the existing lock, build one bounded shallow candidate mapping (no image
  byte copies), account its actual table/resize allocation, then publish atomically.
  Rejection leaves the old mapping untouched. The temporary table adds bounded
  preparation overhead; this is **not** an RSS/global request-memory reservation.
- Preserve validation, IDs, immutable data, expiry, deletion, preparation lease,
  request cleanup and all native execution boundaries. No Processor, numerical,
  Scheduler, KV, canonical settlement or Tool-effect changes.

Object accounting excludes allocator arenas and already acquired request snapshots;
those remain under existing finite request/queue ownership. Removing the byte
budget as well would be unsafe. CURRENT capabilities/Files/operations docs describe
this source-only behavior; historical evidence is unchanged. Production still has
the baseline count/payload-only policy until separately authorized promotion.

## Focused acceptance

[Focused log](focused-tests.log): **158 passed**, 409 existing deprecation warnings,
no skips/xfails, 45.94 seconds. Tests cover Files actual ASGI ingress, native
Chat/Messages conversion, image identity/continuity, cancellation, restore ownership,
ordinary context/body and documentation authority. No weights or real model.

- Previously rejected **129 distinct valid HTTP uploads** succeed well below the
  retained memory budget; repeated first/last IDs resolve in order with unchanged
  image identity. Explicit count policy still rejects safely.
- A **16 KiB test-only retained-allocation budget** rejects tiny images on actual
  metadata usage while raw bytes alone still fit: 429, no candidate publication;
  deletion/expiry allows admission again. Malformed/oversized images still fail.
- Declared oversized body and unending chunked Files/Messages streams reject at
  the finite ingress bound. Tests verify no further draining, no publication or
  prepared inference, and a subsequent upload succeeds (lease released).
- Expiry, deletion and clear preserve acquired immutable snapshots, then weakrefs
  confirm records are collectible when the last snapshot is released. Existing
  disconnect, request failure, cancellation and source-release tests remain active.
- [Finite storage observation](storage-residency.json): six cycles of 256 real
  multipart/Pillow uploads; 28,160 payload bytes vs **156,080 charged bytes** per
  populated store, Python traced peak at most **245,816 bytes**. Store empty after
  each clear. Traced post-clear memory rises from 44,724 to 62,529 bytes; this is
  not a flat-process-memory or leak-free endurance claim. A separate
  [allocation diff](storage-allocation-diff.json) records the small remaining
  timestamp/Pillow/tracemalloc allocation growth and zero retained entries.

Reproduce targeted acceptance:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python -m pytest \
  tests/test_image_files.py tests/test_recipe02_messages_serving.py \
  tests/test_recipe02_chat_boundary.py tests/test_multimodal_cancellation.py \
  tests/test_multimodal_restore_ownership.py tests/test_production_context.py \
  tests/test_documentation_authority.py -q --disable-warnings
node tools/check_harness_image_envelope.mjs /Users/kioju/.local/share/deepseek-harness/node_modules
```

**PASS** for removal of the unnecessary default count blocker with finite retained
allocation admission. **PARTIAL** for general official Harness compatibility:
body/image/storage byte guards, unavailable quota listing, fresh pixel/patch work,
context/output and paired integrity can still reject. Their broad replacement is
**DEFERRED**, not silently qualified. Stop here; no next investigation started.
