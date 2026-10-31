# HTTP / encoded-image soft caps — PARTIAL compatibility; review complete

**Decisions: 16 MiB HTTP body — RETAIN. 64 MiB encoded occurrences — RETAIN.**
Removing either protection requires work outside the permitted existing-path
corrections. No Runtime behavior, CURRENT documentation, dependency installation,
Production or Harness files changed. No threshold was raised or overridden in the
new tests. Stop condition reached; no next soft-cap investigation begun.

## Authorities verified before changes

- Clean `master`, requested HEAD **2ccd97fdf84cc6e8d3ac1736713567ea916bed9c**.
  `git ls-remote origin refs/heads/master` independently advertised that same HEAD.
  No branch/history/staging operations. Only new tests and this evidence directory
  were written (the repository ignores `artifacts/`; evidence remains on disk).
- Production: `/Volumes/SDXC-512/ds41f-1017`, **ds41f-mlx 0.23.0**,
  **Recipe 0.2.0**, MLX 0.32.2, oMLX 0.7.0, uvicorn 0.54.0,
  FastAPI 0.142.2, Starlette 1.7.0, AnyIO 4.15.1, python-multipart 0.0.32.
  Manifest provenance is source `69689d514fc5acb2cbe229f8c863da1e0a855e4a`;
  this is **not** a claim that the installed Python tree equals that commit or
  current checkout. Current Production differs from checkout in Files, HTTP,
  server, capacity and multimodal code; ordinary_admission matches. In particular,
  Production still has older Files count/storage/listing and aggregate pixel
  behavior. Both byte limits under review have the same numerical values.
  [Exact installed package/source/native/manifest identities](identities-before.json),
  [selected source differences](production-vs-checkout.diff),
  [unchanged final Production hash check](identities-after.json).
  Production Python was not executed; hashes and distribution metadata were read
  externally. No Runtime process restart, model load or checkpoint scan.
- Isolated existing `/private/tmp/ds41f-step4ar1-tests`: the same relevant dependency
  versions, including Recipe 0.2.0 and pytest 9.1.1. Tests authenticate the installed
  execution payload in an isolated subprocess, then exercise checkout source.
  Checkout `.venv` has Recipe **0.1.1**, so was not used. No installs performed.
- Installed official Harness CLI/provider/LLM/image-offload/basic-compaction:
  **0.2.1-rc.2**. Provider bundle SHA-256
  `eba35025ed1eedda89c51f2da46a3a968f7be1d59719f2ceeadfc4518dff37d6`.
  The installed manifests do not establish a build Git SHA. Independently resolved
  official `dsh-v0.2.1-rc.2` tag to
  **687f282e8625c6b8d11ff58211a90524be2e4317** and fetched/read release
  `images.ts`, `request-pricing.ts`, `request-files.ts`, `serialize.ts`
  ([retained source](upstream/), hashes in identities-after). Installed bundle
  behavior and executable checks, not a different upstream HEAD, decide findings.
- Recipe source: existing `third_party/mtp/recipe-02-source.tar.gz`, pinned by
  `third_party/mtp/normal-local.json` to
  **93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170**, plus the repository's existing
  native-semantic-boundary patch. Extracted only to `/private/tmp/http-encoded-audit`.
  Read Python/Rust request deserialization, image conversion, Python image getters,
  and v4/v41 encoding collection paths. The semantic patch changes response/stream
  ownership, not these ingress/image allocation paths. Archive/patch hashes retained.

References, not repeated broad audits:
[previous Files/HTTP review](../files-http-soft-caps/report.md),
[Files recovery evidence](../files-http-soft-caps/),
[historical-image admission](../historical-image-aggregate-admission/),
[cached-image admission](../cached-image-admission/),
[fresh Vision review](../fresh-vision-soft-caps/).
The previous Files report's unavailable-listing statement describes its older
baseline, not current HEAD. Current Files quota recovery is separate from these
request guards and is not reopened here.

## Actual allocation / lifetime path

`HTTP socket → LocalBoundary → Starlette body → Python JSON → Recipe native schema
→ Recipe conversion → render/encode → prepare_multimodal → capacity → release
preparation lease → ProductionMTPBackend._enqueue → existing Scheduler → paired
admission / encode_new → settlement / retirement`

| Stage / owner | Allocations and copies actually made | End of lifetime / rejection |
| --- | --- | --- |
| uvicorn h11 / socket | `cycle.body += event.data`; read flow pauses above 65,536 bytes, subject to delivered-event overshoot. Socket/header/connection limits are separate. This is backpressure, **not** a whole-body bound. | `receive()` transfers/resets cycle body. No socket-level total request budget replaces LocalBoundary. |
| `LocalBoundary.__call__` | Validates headers before receipt. Ordinary preparation lock serializes receipt/conversion, not all in-flight request memory. `bounded_receive` counts actual bytes, checks one deadline; it does not aggregate. | Content-Length >16 MiB: **413 body_limit before receiving**. Actual/chunked cumulative bytes >16 MiB: **413 before returning offending event to Starlette**; does not drain unending input. Offending ASGI event already exists. Finally releases lease on errors/cancellation; timeout 408. |
| `Request.body()` | Starlette retains chunk references in a list, then `b''.join(chunks)` caches `_body`. Multi-chunk join temporarily coexists with chunks. Single-chunk join may share; do not universally charge or credit a full copy. | Chunk list dies at body return; cached raw bytes stay with Request, including while response/delivery holds Request. Releasing preparation does not free `_body`. |
| `prepare_request` / strict admission | Initially `json.loads(body)` creates `policy_payload`. Ordinary converter then independently calls `admit_structure` → `strict_json`: UTF-8 decode + Python objects, duplicate-key/nonfinite/top-level-object validation. Messages with a store performs another strict parse before resolver, and converter parses raw again before selecting resolved payload. Defaults are shallow dict copies; strings still share within those copies. | Direct `admit_structure`/converters reject len(raw)>16 MiB before their parse. **Direct `prepare_request` does its initial non-strict parse before this guard**; it ultimately rejects, but is not safe untrusted unbounded ingress by itself. HTTP protects it first. Temporary trees die on function/exception unwind, not Scheduler admission. |
| Messages File-ID resolution | Only admitted native image locations, including supported tool results, are traversed. Original file grammar is first validated with `MessagesRequest(json.dumps(original).encode())`, without conversion. `snapshot` returns immutable StoredImage; data bytes share persistent store storage at this step. Each occurrence then independently makes base64 bytes and ASCII str; only str stays in mutated request-local tree. | `resolve_messages.visit`: count and logical encoded total checked **before each File-ID b64encode**. Excess → FileError 400, before expanded Recipe conversion/Processor/Scheduler. Earlier occurrences may already be expanded privately; none published. Snapshot survives deletion for acquired occurrence; a later lookup can fail if deleted. No replacement under the same immutable ID. |
| Mixed sources / expanded JSON | File and inline occurrences count together, not unique IDs. Inline estimate uses represented length/padding. Resolver mutates only this tree's image source, never semantic history or arbitrary tool inputs. `json.dumps(recipe_payload).encode()` creates expanded string and bytes, **allowed to exceed raw 16 MiB**. | Resolver failure leaves partially expanded tree only in the unwinding request/traceback; no store mutation. Temporary serialization released after converter return. No whole-expanded-body 16 MiB check is incorrectly applied. |
| Recipe 0.2 deserialization / conversion | Rust `deserialize`: serde JSON `Value`, then typed request; dict input uses `python_to_json`, so simply passing a dict is not a zero-copy fix. Consuming `convert()` transfers request ownership. Messages `base64_image_source` decodes for validity, discards decoded bytes, constructs data URL String. Chat data URLs also validate decoding. | Native schema/conversion rejects invalid structure/base64; consumed request cannot be reused. Converted conversation retains source data URLs. Native allocations are not Python-tracemalloc allocations. |
| Recipe render / encode / getters | Encoding clones message/image sources; rendering collects cloned sources. Each `rendered.image_sources` getter creates Python wrapper objects with cloned Rust String/Vec backing (`image_source_to_py`). Server accesses it repeatedly (tests for presence, origins, preparation, prepared result). `RecipePreparedRequest.conversation_request` and `.converted` are aliases of **one** converted object, not two copies, but `.image_sources` is separately cloned. | Temporary render/encode/getter lists drop after preparation; converted conversation and prepared image wrappers remain queued/response-owned. Python URL getter additionally materializes a Python str; partition/base64 decoding makes further transient representations. |
| `prepare_multimodal` | Sequential full container/pixel decode and geometry validation; only encoded bytes + spans/tokens survive. `image_bytes` independently decodes every occurrence into new Python bytes for data URLs. `_sources` keyed by expanded span start, not File ID/digest. Identical images have independent occurrence positions and distinct decoded byte allocations. | Aggregate 64 MiB check is **after image_bytes for current occurrence**, before that occurrence's Pillow validation. Thus at most one additional per-image decode can precede this check; it is not a universal pre-base64-allocation guard. Per-image 16 MiB bounds it. Local partial dict/spans disappear on failure/unwind; no partial MultimodalInput is returned. |
| Prepared request / Scheduler | Retains tokens, converted conversation, image source wrappers, multimodal `_sources`. Capacity checks actual expanded IDs/output, not HTTP bytes. oMLX queue admission bounds request count (`max(max_num_seqs*4,32)` waiting), **after** preparation. Execution/KV/paired managers own different allocations. HTTP Request/body may also remain alive through delivery. | Source bytes clear on encode_new finally, Scheduler admission/start failure/retire/abort/close. Historical sources clear/pop even for paired hits. Clearing `_sources` does **not** free base64 in converted/prepared image sources. Those die when prepared/response/request ownership ends. |
| Failure / cancellation / settlement | HTTP finally releases transport lease. AnyIO threadpool conversion is not forcibly interrupted mid-native conversion; intermediates live until worker exits. Existing protected native worker drains before cleanup; Scheduler remains sole executable owner. | Error handlers unwind local trees/tracebacks; no promise of immediate RSS return. Explicit source release is idempotent. Paired identity/frontier checks, retirement and canonical settlement remain untouched. |

Logical encoded bytes are consequently neither unique persistent-store bytes nor
actual retained bytes. Likewise raw JSON length is not its parsed/native heap size.
A paired hit avoids fresh Vision work, not transport validation or source materialization.

## Official Harness contract and concrete mismatches

Defaults: **600 image occurrences**, **128 MiB raw request-image bytes** for Files,
**20 MiB represented base64 image bytes** for inline fallback. Counts/bytes are per
occurrence across retained history, not per unique attachment. These are provider
configurations, not Runtime resource reservations or whole-JSON budgets.

Official request preparation deduplicates reading an attachment version in a Map;
Files mapping can reuse one immutable provider ID. Serializer nevertheless emits
an image block and handle text for **every occurrence**, including historical user
images and tool results (preserving tool IDs/order). History is not replaced by a
Runtime KV reference on the wire. Raw admission precedes Files resolution. A Files
resolution failure triggers request-wide inline fallback, independently checked at
represented base64 size; exceeding it requests durable offload, not blind retry.

Harness already has two representation-reduction mechanisms:

1. Deterministic request-image variants: route dimensions/pixel budget, default
   **2 MiB target** and per-side projection. The target is not an absolute guarantee
   (quality-ladder fallback); it is not a total request-memory lease.
2. Durable oldest-occurrence image offload/compaction. `IMAGE_OFFLOAD_REQUIRED` plus
   selections records offload, retries projected history with text placeholders;
   original durable images are not silently mutated. Count quantum defaults to 20.
   Runtime generic 400/413 maps to `INVALID_REQUEST`, **not** that offload signal.
   Forging the signal or changing canonical history here would change protocol/
   lifecycle ownership. Upstream configurable lower budgets can mitigate a route;
   they do not expand Runtime capability or bound arbitrary non-Harness clients.

[Executable installed envelope/offload checks](official-envelope.json).
[Concrete provider serialization](official-requests.json), generated by
[check_official_requests.mjs](check_official_requests.mjs): no network, injected
request-local Files/attachment services, no Harness state writes. It calls the
**actual installed request-image transform** and actual provider preparation/
serialization, capturing fetch before transport. The images are valid 32×32 PNGs
with opaque ancillary chunks, not pixel bombs. Request transform preserves their
bytes at the default 2 MiB target. This avoids assuming large source bytes survive
normalization. No model-capacity acceptance is claimed.

| Minimal compact construction | Official result | Runtime obstacle |
| --- | --- | --- |
| One 1 MiB immutable PNG repeated 13 times in one user message, controlled Files-resolution failure | Raw 13 MiB; represented base64 **18,175,352 bytes** <20 MiB; actual fallback JSON **18,179,185 bytes** | 16 MiB whole-body **413**, before Recipe/Processor. `test_inline_harness_sized_body_mismatch` reproduces equivalent wire shape. Non-image text/schema/history also consume this bound. |
| One 2 MiB PNG repeated 33 times via same File ID | Reads variant once; resolves 33 occurrences; raw **69,206,016 bytes (66 MiB)** <128 MiB; actual JSON only **10,787 bytes** | 64 MiB resolver **400** at occurrence 33, before its base64 allocation. Upload/store is just one 2 MiB image; Files quota recovery cannot solve this. |

These are genuine provider-envelope mismatches, not proof every such request would
also pass independent model context/fresh-patch/paired-history restrictions. The
second example also works as retained historical occurrences at provider level;
Runtime still requires exact paired spans/digests/frontier for historical reuse.
Unique images similarly contribute all bytes; inline-only ordinary requests normally
hit the raw body guard well before reaching 64 MiB decoded image occurrences.

## Independent decisions / alternatives evaluated

| Limit | Classification | Precise blocker to removal/replacement |
| --- | --- | --- |
| `ORDINARY_BODY_BYTES` =16 MiB | **RETAIN** | Only total raw ingress/JSON allocation bound. Backpressure bounds unread socket staging, not consumed/retained chunks. Preparation lock bounds concurrency, not bytes. Recipe fully buffers/deserializes; actual-token and Scheduler/KV checks happen later. Neither official Harness budgets (image-only, client-side) nor source pixels bound arbitrary JSON structure/text. No qualified existing byte lease replaces it. |
| `MAX_REQUEST_IMAGE_BYTES` =64 MiB | **RETAIN** | Bounds occurrence expansion and queued encoded transport. Store admission owns unique persistent bytes/metadata, not repeated base64/Recipe/queued bytes. 600 occurrences × per-image bound is a much looser finite worst-case, not equivalent resource protection. Pixel/patch/context/paired admission owns fresh work or execution, not encoded metadata or repeated transport representations. Current Recipe source clones and span-keyed decoded sources defeat zero-cost duplicate assumptions. |

**DEFER the architectural replacements**, not the two current guards. Both values
are local conservative policies; this audit does not prove their exact numeric
optimum or provide a global RSS budget. Existing finite receipt/queue/worker limits
remain relevant but are not dynamic admission for all retained Python/native bytes.

Small copy improvements exist in principle: avoid repeated raw parsing/getter calls,
share one request-local base64 string across repeated immutable IDs, use Recipe's
dict deserializer, or stop retaining redundant transport wrappers. None bounds
unlimited raw ingress, changes the Rust per-occurrence String clones, or eliminates
the decoded span-owned queue bytes. Dict deserialization still builds serde Value;
clearing `_sources` still leaves converted conversation source Strings. Sharing
store bytes through the entire pipeline would require changing the established
Recipe/File-ID resolution and source-lifetime boundary, not just deleting a guard.
No isolated copy micro-optimization is presented as permission to admit larger
requests, and no incidental Runtime rewrite was made. A streaming JSON parser,
second converter, new cache/offload layer or general memory lease would violate
this task's scope. Larger memory snapshots/multipliers are not substitutes.

## Focused verification and memory

- [Focused tests](focused-tests.log): **172 passed**, 513 existing warnings, no
  skips/xfails. New `tests/test_http_encoded_soft_caps.py` tests both protocols at
  the actual 16 MiB equality boundary and rejection one byte above; strict malformed,
  duplicate-key, top-level-array and nonfinite JSON; actual official-sized fallback;
  repeated 2 MiB IDs and mixed inline/file occurrences rejecting before the next
  File-ID base64 allocation; no preparation publication on refusal, subsequent
  request succeeds; timeout, disconnect and cancellation release HTTP lease;
  encoder error/CancelledError clears sources and prepared objects are collectible.
- Existing focused tests cover Content-Length and unending chunked receipt (no
  further drain), unique/mixed/repeated sources, upload/delete/expiry snapshots,
  exact identity/order/tool-result origins, changed/partial/missing pair refusal,
  historical fresh-work accounting, Processor/container validation, cancellation,
  restore ownership and capacity. Older tests with lowered constants are unit
  accounting checks only, not claimed expanded capability; new boundary tests use
  unchanged deployed numerical byte limits.
- [Additional lifecycle run](lifecycle-targeted.log): **72 passed, 4 deselected**.
  Initial unfiltered run: **72 passed, 4 failed** ([log](lifecycle-tests.log)); all
  four unchanged `test_worker_failure_diagnostics` parameterizations fail again
  [in isolation](diagnostics-isolated.log). The test keeps `numerical_profile`
  monkeypatched to raise its sentinel while requesting `/health`; current health's
  `vision_capacity` calls it and returns 400, not the asserted 200/503.
  [Confirmed response/cause](diagnostics-cause.log). Scheduler error/retirement
  assertions preceding health passed. No production/test-fixture behavior was
  changed to hide this baseline failure. Therefore no all-regressions-green claim.
- [Finite memory observation](memory.json), reproducible with
  [observe_memory.py](observe_memory.py): 569-byte Files request, one stored 1 MiB
  PNG repeated four times. Resolver retains **four distinct** base64 strings,
  5,592,416 represented bytes; traced peak **7,046,771 bytes**. After resolved tree
  release, traced current returns to 42,552 bytes. Full preparation retains **four
  distinct** decoded source bytes objects, 4,194,304 total, traced peak 16,831,465.
  Clearing sources drops traced current from 4,643,822 to 449,519 bytes, but four
  Recipe source wrappers still remain. Dropping prepared makes both prepared and
  multimodal weakrefs collectible; native allocator RSS need not fall.
- Production-size 33×2 MiB resolver refusal: Python-traced peak **94,817,648 bytes**,
  traced current 2,521,179 before vs 2,521,771 after unwind; after request/store
  release 424,095. RSS rose from ~228 MB initial to ~383 MB process high-water and
  stayed resident. **tracemalloc does not see Rust/Pillow/MLX native allocations**;
  sampled RSS includes native heaps, allocator retention, imports and arenas, not
  a per-source allocation attribution or macOS physical-footprint measurement.
  No claim of leak-free endurance or a validated higher resource ceiling.

No model-facing preparation or ownership behavior changed, so bounded model-free
protocol/lifetime tests suffice for this review. No real-model run, maximum-capacity
campaign, Windows qualification or threshold-override expansion claim.

Reproduction (from repository root, existing isolated environment):

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python -m pytest \
  tests/test_http_encoded_soft_caps.py tests/test_image_files.py \
  tests/test_recipe02_messages_serving.py tests/test_recipe02_chat_boundary.py \
  tests/test_multimodal_cancellation.py tests/test_multimodal_restore_ownership.py \
  tests/test_production_context.py tests/test_historical_image_aggregate.py \
  tests/test_cached_image_admission.py -q --disable-warnings
PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python \
  artifacts/http-encoded-image-soft-caps/observe_memory.py
node tools/check_harness_image_envelope.mjs /Users/kioju/.local/share/deepseek-harness/node_modules
# check_official_requests.mjs takes node_modules and a disposable directory holding
# 1048576.png / 2097152.png, generated with test_http_encoded_soft_caps.sized_png.
```

**PASS — scoped audit, independent classifications and focused unchanged-boundary
verification. PARTIAL — official Harness envelope compatibility and broader test
suite (four baseline diagnostic-fixture failures).** Both caps remain real resource
restrictions; replacement is blocked on pipeline/resource ownership redesign, not
on permission to increase constants. All excluded limits and paired settlement
semantics remain unchanged; Production hashes were rechecked unchanged.
