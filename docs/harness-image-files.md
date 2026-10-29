# Bounded native Messages image Files API (Gate W2)

Implemented for ordinary `mtp-serving-v1` only; **model-free verification, not
Windows or official Harness execution qualification**. The wire authority is
Harness 0.2.1-rc.2 source `687f282e8625c6b8d11ff58211a90524be2e4317`,
correlated in the [Gate W1 audit](flash-1017-harness-files-compatibility.md).
No provider modification, second Messages translator, Vision processor or KV
owner is introduced. Existing Phase 0–3 evidence remains unchanged.

## Gate F0 observed qualification

**PASS — bounded macOS official Harness File-ID acceptance**, tested checkout
`6766b652fd3293f7523533decc102a9697168af1`. One unmodified official Harness
0.2.1-rc.2 native Messages conversation used only official `read_image`: two
successful Files uploads, A then A+B nested file-reference histories, meaningful
visual recognition, two fresh Vision encodes and one historical reuse. All three
native completions and complete paired-cache traces were retained, including
final synchronous flush; Target/DSpark continuity, settlement and zero replay/
repack passed. Fresh disposable installation, original Flash-1017, Recipe 0.2.0,
native MTP/GLM, Thinking OFF, read-only sandbox and session log disabled.
See [report and evidence](../artifacts/harness-files/gate-f0-report.md).
This adds only two-image macOS tool-path qualification: not Windows/GUI,
persistent or multi-user Files, lifecycle/quota recovery, eviction/restart,
maximum-capacity, endurance or production promotion. The W2 model-free scope
and historical Phase 0–3 receipts remain unchanged.

## Bounded offload/reacquisition follow-up

[Phase 3 / Step 3A](../artifacts/multimodal-runtime/phase3-step3a-offload-reacquisition.md)
adds one undeployed official macOS session: count budget 3 / quantum 1 produces
two durable offloads; ordinary changed Messages restore C=711 and freshly encode
surviving historical images. A new official `read_image` of accessible A preserves
its old omitted occurrence and consumes the new occurrence through Vision,
settlement and publication. C3/C5 positive cases pass for this session only;
overall acceptance is PARTIAL because a post-shutdown diagnostic failure prevented
cleanup weakref/MLX/executor measurements. No Files durability or arbitrary Runtime
400 recovery is established. Historical W2 refusal descriptions below retain their
original scope; current independent Messages reconstruction is governed by
[ordinary paired admission](mtp-production-serving.md#paired-capture-and-restore).

## Supported operations

### `POST /v1/files`

Uncompressed `multipart/form-data` with exactly these three fields (any order):

- `file`: binary PNG/JPEG/WebP, declared `Content-Type` and nonempty UTF-8 filename
  (at most 512 characters, no control characters).
- `expires_after[anchor]`: `created_at`.
- `expires_after[seconds]`: decimal integer **3600..2592000**, inclusive.

No `purpose` field is required or accepted. Multipart header/part counts and
sizes are bounded; duplicate/unknown fields, incomplete bodies, transfer-encoded
parts, wrong MIME/format and unsupported expiry fail. The shared Runtime
`validate_encoded_image` verifies actual format, complete decoding, single frame,
pixels. Aspect ratios use the existing checkpoint resize planner, subject to
expanded-token and patch budgets, not a fixed 2:1 guard. Admitted encoded bytes are retained **exactly**, without
recompression, normalization or filesystem interpretation of filenames.

Success: HTTP 200 with exactly this schema (values are examples):

```json
{"id":"file_opaque_server_id","type":"file","mime_type":"image/png","size_bytes":24147,"created_at":"2026-10-24T00:00:00Z","filename":"original.png"}
```

IDs are random server-issued UUID-based opaque tokens, not hashes, paths or
filenames. Creation timestamps are actual UTC RFC3339 whole-second timestamps.
Expiry is `created_at + requested seconds`; no `expires_at` response is required.

### `DELETE /v1/files/{id}`

No body or Content-Type is required. Success: HTTP 200 with exactly:

```json
{"id":"file_opaque_server_id","type":"file_deleted"}
```

Unknown, deleted, expired or other-instance IDs return HTTP 404, including repeated
deletes. Deletion makes new resolutions unavailable. Acquired immutable request
snapshots remain valid through concurrent deletion/expiry. Neither deletion nor
cleanup accesses Scheduler, model state or paired KV.

### Native `POST /v1/messages` references

```json
{"type":"image","source":{"type":"file","file_id":"file_opaque_server_id"}}
```

Supported only in user-message image content, or image content inside a user
`tool_result`. Not assistant/system images, recursively nested tool results,
documents, paths or URLs. Image blocks accept exactly `type,source`; file sources
accept exactly `type,file_id`. All other native field/capability rejection remains
in force. Arbitrary tool-input JSON and ignored inventory metadata are not scanned
for images or interpreted as file references.

Strict duplicate-key/depth/raw-body admission and native Recipe deserialization
validate the original structure. The request-local resolver replaces only admitted
file sources with existing base64 sources, retaining block order, wire message
positions, result nesting and exact tool IDs. The **same resolved tree** feeds
conversion, policy, origin classification and existing image admission. Recipe
still owns semantic source ordering (including result reordering to call order).
No expanded internal JSON is submitted to the raw network-body limit again.

Missing references fail before inference with the ordinary error envelope:

```json
{"error":{"type":"invalid_request_error","code":"file_not_found","message":"file not found or expired: file_opaque_server_id","param":null}}
```

This matches rc.2 `providerRejectedFileId` text classification and names the exact
ID for bounded mapping invalidation/reupload. Other errors are non-2xx as well:
invalid form/schema/image/expiry 400, excessive upload 413, retained-storage quota
429 (`file_storage_quota`), busy upload 503 (`upload_busy`). No inference retry,
blank image substitution, history removal or filename reconstruction is performed.

## Ownership, memory and lifecycle

Ownership is **one app/process's existing trusted deployment scope**. Runtime has
no authentication/TLS or per-user isolation: loopback or explicitly trusted private
LAN, single operator. Auth-looking client headers do not create authenticated
accounts or separate namespaces. Everyone admitted to that deployment can use or
delete its IDs; a different app instance cannot. Do not expose this as a secure
multi-tenant Files service. Multiple worker processes do not share IDs: use the
existing single-process deployment.

Storage is a bounded in-memory mapping of immutable bytes; no upload directories,
user paths, weights, caches or temporary disk files are opened. Publish occurs
atomically only after the complete multipart body and image have passed admission.
Malformed, oversized, disconnected and otherwise incomplete uploads publish
nothing; their request buffers are released on unwinding. A complete admitted
upload whose response is lost can remain until deletion/expiry (an uncertain
receipt is not a partial image). There is no persistent file-ID availability.
Graceful shutdown clears the mapping; process exit/restart loses all IDs. Passive
expired entries are swept every 30 seconds, and every upload/resolve/delete also
checks expiry immediately. Request-held snapshots may outlive that sweep.

| Bound | Enforced default/ceiling |
| --- | --- |
| Encoded bytes per file | 16 MiB |
| Retained files per deployment | 128 |
| Aggregate retained encoded bytes | 64 MiB |
| Concurrent uploads | 1; reuse the existing HTTP preparation lease, reject busy uploads rather than queue their bodies |
| Multipart wire bytes | 16 MiB + 64 KiB (includes form overhead); enforced before buffering, including chunked bodies |
| Multipart shape | 3 parts, 2 headers/part, 2048 bytes/header, 32 bytes/expiry field, 70-byte boundary |
| HTTP body timeout | Existing 30 seconds; no new unbounded upload wait |
| Actual Messages JSON wire bytes | Unchanged 16 MiB |
| Files-mode image occurrences | 600 including inline images and repeated IDs; independent of the 128 distinct storage slots |
| Resolved encoded image bytes | 64 MiB aggregate; counted per occurrence before base64 expansion |
| Fresh source pixels | 16,777,216 after verified paired coverage; no cumulative covered-history pixel charge during resolution (bounded Production acceptance linked below) |

`create_app(files_limits=FileLimits(file_bytes=..., count=..., storage_bytes=...))`
can **lower only** the first three storage ceilings; no new CLI/environment or
request-level limit override exists. Multipart ingress remains under its fixed
ceiling even with lower per-file configuration. Parsing uses the already pinned
`python-multipart` dependency with bounded headers/parts and 64 KiB parser chunks.
Buffering is bounded, not zero-copy: raw body, parser bytes and decoding have
finite transient overhead in the one shared preparation slot. Resolution checks
occurrence/encoded budgets before allocating base64 copies; at most 64 MiB encoded
bytes become approximately 86 MiB base64 plus bounded JSON/native-parser copies.
This is not a zero-cost memory path or expanded model capacity.

After resolution, the shared Vision path still enforces **16 MiB/image,
4,194,304 pixels/image, processor-planned aspect ratios and expanded positions**
(1024/image for the Flash-1017 Processor in the ordinary checkout), encoded
residency, terminal-text, context expansion and output reservation. The separate
[per-image token correction](../artifacts/per-image-expanded-token-admission/report.md)
has isolated native 1016/1024-position evidence and subsequent
[69689d5 installed Production acceptance](../artifacts/production-multimodal-69689d5/report.md)
through HTTP Files/Messages. The installed ordinary per-image ceiling now comes
from Processor configuration, not the historical 1014 qualification.
Historical occurrences still undergo complete sequential decoding validation and
identity accounting, but decoded buffers are not queued. **16,777,216 source
pixels / 36,864 ViT patches** bound actual fresh work after validated paired
coverage; historical wire flags only defer early charges, not execution admission.
Conditional aggregate-patch discount still requires complete compatible historical
coverage and supported pair layouts. Existing independent Messages reconstruction
rules are unchanged; no new replay/recovery path is added. Files does not allocate
model KV on upload or resolution. The [20-occurrence source correction](../artifacts/historical-image-aggregate-admission/report.md)
has isolated native evidence; the subsequent
[b2f7654 Production installation](../artifacts/production-multimodal-b2f7654/report.md)
passes the affected HTTP Files/Messages boundary. The earlier source-task report
remains unchanged.

Harness rc.2's default 600 occurrences / 128 MiB raw / 20 MiB base64-image budgets
are request policies, not these physical storage/transport limits. Full-envelope
compatibility remains **PARTIAL**. Runtime 400 is `INVALID_REQUEST`, not
`IMAGE_OFFLOAD_REQUIRED`: it does not cause the official image-offload plugin to
log a projection or retry. Offload/summary compaction can invalidate the canonical
prefix. Existing independent Messages C=0 / whole-image-safe C>0 reconstruction
is described by [current capabilities](capabilities.md#bounded-inline-images);
this pixel fix does not extend its rules. See the historical
[assessment](../artifacts/harness-image-capacity/report.md).

## Identity, cache and compatibility exclusions

The existing encoded-byte SHA-256/span/grid identity and paired cache remain
authoritative. Same bytes at the same occurrence produce the same Recipe tokens
and image identities for inline input, a file reference and a replacement ID.
A live file **does not** restore missing historical Target/DSpark paired state.
Independent Messages reconstruction, where already supported, must charge every
uncovered image fresh and fit both fresh budgets; exact logical Chat lease loss
still fails closed. No fabricated hit, partial-image restore or new replay path
is introduced. File lifetime and paired-KV lifetime are independent.

**Unsupported:** list/pagination, metadata retrieval, content/download endpoints,
OpenAI `purpose`/Files variants, arbitrary documents, remote URLs, persistent IDs,
account emulation and general file management. In particular, upstream quota
recovery that lists/reclaims oldest files is **not implemented**. Upload quota
failure may cause Harness's existing bounded Files failure/inline fallback; this
is not complete upstream quota handling. Operators can explicitly delete known
IDs, wait for expiry, or restart (which also loses paired state).

Supported Harness configuration requires native Messages/image modality, Runtime
image bounds, stable/addition-only tools, and explicitly:

```text
session-log-deepseek.enabled=false
```

`dsh_session_log` remains rejected; no incremental ingestion/watermarks/session
service is added. `dsh_plugin_packages` handling is unchanged (bounded admission,
then ignored). Tool removal and other unsupported provider extensions remain off.

## Verification and next gate

`tests/test_image_files.py` exercises actual HTTP parsing/resolution and native
Recipe/Vision preparation with a controlled backend response, plus the real paired
admission rejection fence with synthetic owners. No weights/model loading or
real-model inference is used. The focused suite covers binary formats, multipart
contract/limits/cancellation, deletion, stale/scope/expiry errors, nested tool IDs,
order/identity equivalence, repeated historical references and resource fences.

Windows normal-operation qualification remains separately authorized: retain exact
installed rc.2/provider/config identities and sanitized wire traces; test ordinary
GUI upload **and** official tool-image continuation, stale-ID reupload, paired
continuation and explicit cache-miss rejection in the installed Runtime. Also
qualify the finite memory envelope under the selected bounded workload. No Windows
or macOS Harness run, Phase 4, package promotion or broad compatibility claim is
implied by these model-free checks.
