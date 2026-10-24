# Gate F0 — official Harness File-ID vision acceptance

**PASS — one bounded macOS conversation; A–E independently pass.**
Tested branch `Visual-Feedback-Loop`, starting/tested HEAD
`6766b652fd3293f7523533decc102a9697168af1`. No production source, official
provider, serialization, tool results or synthetic conversation history changed.
No replacement conversation, external generation driver, renderer or browser.

| Acceptance | Result | Direct evidence |
| --- | --- | --- |
| A. Files upload | **PASS, 2/2** | Official multipart uploads of A and B, each HTTP 200 with valid metadata and a distinct server-issued ID. |
| B. Messages references | **PASS** | Native `image.source.type=file`, original nested tool-result IDs, histories A then A+B; no inline fallback. |
| C. Vision | **PASS** | Normal admission resolves exact uploaded bytes into Recipe spans; meaningful independent visual recognition of both PNGs. |
| D. Paired Cache | **PASS** | Exact prefix restoration, Target/DSpark agreement, fresh-only encoding, historical reuse, canonical settlement/publication and zero replay/repack for all three generations. |
| E. Evidence retention | **PASS, 3/3** | Three native completions match canonical Runtime responses; three complete raw traces, final memory count = archived count before backend destruction. |

Evidence: [gate-f0-evidence/](gate-f0-evidence/), especially
[audit](gate-f0-evidence/audit-results.json), [wire](gate-f0-evidence/wire.jsonl),
[sequence](gate-f0-evidence/sequence.json), [cache](gate-f0-evidence/cache-summary.json),
[preparation](gate-f0-evidence/prepared-images.jsonl),
[retention](gate-f0-evidence/trace-retention.json), and
[independent PNG assessment](gate-f0-evidence/independent-review.json).
HTTP success and the model's completion claim are not the acceptance basis.

## Identities and isolated installation

- Official installed Harness **0.2.1-rc.2**, Node **v24.5.0**. All 17 qualified
  package versions and entrypoint SHA-256 values match Phase 3
  ([identities](gate-f0-evidence/identities.json)). Official rc.2 source authority
  remains `687f282e8625c6b8d11ff58211a90524be2e4317`; installed manifests do not
  establish an exact build Git SHA. No official package was patched or installed.
- Newly built, non-editable Runtime **0.23.0**, installed provenance **the tested
  `6766b65…` revision**, not the old Phase 3 disposable package. Used the existing
  `tools/build_execution_package.py --checkpoint-variant flash-1017 --glm-profile native`
  and isolated `uv` wheel installation. **All 114 Runtime Python sources match**
  the checkout, independently and again in the execution process before backend
  imports. This includes Files, Messages admission, server, Vision and paired-cache
  owners ([source comparison](gate-f0-evidence/source-package-match.json),
  [in-process admission](gate-f0-evidence/runtime-admission.json)).
- Fresh `/tmp/ds41f-f0-runtime`: Python **3.13.15**, Recipe **0.2.0**, repaired
  wheel SHA-256 `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`,
  MLX **0.32.2**, MLX-LM **0.31.4.dev132+g94cdcae13**, oMLX **0.7.0**.
  Offline `--no-deps` installation preserves every existing dependency version;
  before/after freezes and wheel hash are retained. Existing native GLM provisioner
  and tokenizer staging were reused, without dependency upgrades.
- Original **DeepSeek-V4.1-Flash-1017**, checkpoint revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`, 48 official shards,
  `flash-1017-pre-woa-fp8-v1`, original external checkpoint path retained in
  [identity](gate-f0-evidence/runtime-identity.json). Full-byte admission was
  performed for the fresh installation; no identity cache was copied to bypass it.
  Execution identity `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`
  remains the qualified native numerical/dependency identity; this is not used
  as a substitute for the separate current-W2 source comparison.
- **Native Target/DSpark MTP, native GLM, Thinking OFF**, `mtp-serving-v1`,
  loopback-only `127.0.0.1:18083`, one worker. Normal admission precedes backend
  construction/executor-owned eager loading. Ready health is retained.

Two setup-only bootstrap rejections are disclosed in
[installation record](gate-f0-evidence/installation.md): the initial manually
staged venv lacked the GLM native bundle; the next lacked the external tokenizer.
Both failed payload admission **before model loading, zero conversations/requests**.
Their logs/exit receipts remain in separate startup directories. Completing the
existing provisioner's two resource-staging steps resolved admission; no protocol
or production correction was needed. A fresh Harness root was used each time.

## Actual Harness configuration and budget

Retained [patch](gate-f0-evidence/profile.patch.yml) and
[composed tree](gate-f0-evidence/composed-config.yml) are the executed configuration:

- Native official DeepSeek API-key Messages provider; `session-log-deepseek.enabled=false`.
  `dsh_plugin_packages` remains enabled and reaches every Messages request unchanged.
- **Only official `read_image` is model-facing**. Other filesystem operations,
  shell, web, subagents, plan mode and automatic generation retry are disabled.
  Existing official **read-only** sandbox, approval `never`, single tool concurrency.
- Fresh isolated home/profile/agents/session/workspace and attachment/upload-index
  state. Immutable workspace image copies; no operator instructions or historical
  attachment mappings copied. A minimal official monotonic `tools.guard()` policy
  only limits two image effects to the two supplied paths; it dispatches nothing,
  does not schedule observation order or change arguments/results.
- One headless CLI conversation, **36.16 s**, **3 generation requests**, **2 tool
  effects**, **699 output tokens total**. Bounds: 5 requests, 1200 output tokens
  per request, 180 s per generation, 900 s conversation, existing 1200 s startup,
  16 MiB Messages body and unchanged Runtime image limits. Largest actual JSON
  body **9,171 bytes**. No generation, uncertain upload or tool-effect retry.
- [Task](gate-f0-evidence/task.json) specifies filenames and A→B observation order,
  description/comparison and finish; it supplies **no expected visual answers**.
  Harness selected and executed the tools normally. Observer passes original
  fetch inputs/responses unchanged and makes no generation requests.

## Actual Files transport and immutable images

Both PNGs are 1280×720, copied without supervisor transformation and chmod 0444.
Original = workspace copy = uploaded bytes = Recipe source SHA-256, and final
workspace hashes remain unchanged ([images](gate-f0-evidence/images.json)).

| Image/source | Bytes | SHA-256 |
| --- | ---: | --- |
| A: Phase 0 `probe.png` | 24,147 | `7f594ec2e888cf9777a8a9bddc1baacc03df39459e30a43f37eaa9d7dd41f08a` |
| B: Phase 2 continuation `render-00.png` | 35,329 | `fc6dd2f551cb196b84eb64d1bd0c91b01633ce881403d4bc4f0a1d602c1176d9` |

Official `FormData` uploads have exactly `expires_after[anchor]=created_at`,
`expires_after[seconds]=604800`, and binary `file` with MIME `image/png`.
Fetch owns multipart encoding/boundary; the passive observer records original
FormData fields and binary bytes before unmodified native fetch, not a reconstructed
upload. Actual server success and Recipe digest agreement independently confirm
admission. No credential headers, full base64 or binary bodies are retained.

| Wire request | HTTP | Returned ID | Original official Tool Call ID |
| --- | --- | --- | --- |
| 2: upload A | 200 | `file_3ea9a69d78d84b82a252139268408137` | `call_00_lY1KRqjXnZvOkpzzqdyS7034` |
| 4: upload B | 200 | `file_6ae25595f086470aa1e38a04cc7e57bc` | `call_00_sUkeTFq7ihy6TZ9OneYz0304` |

Each actual returned metadata object has exactly
`id,type,mime_type,size_bytes,created_at,filename`, `type=file`, correct MIME/size,
UTC creation time and original official upload filename. Full sanitized receipts
are retained in `wire.jsonl`.

Messages wire 3 retains A at `messages.2.content.0.tool_result.2`; wire 5 retains
that same A and B at `messages.4.content.0.tool_result.2`. Both results retain
`[text,text,image]`, `is_error=false`, original call IDs and order. Each recorded
native image block is exactly:

```json
{"type":"image","source":{"type":"file","file_id":"actual-returned-id"}}
```

The evidence stores the **actual IDs**, not this report's placeholder. All three
image occurrences refer to successful earlier uploads in this same disposable
Runtime; A's ID stays unchanged. **Zero base64 fallback, upload failures or
stale-ID recovery.** Native Messages requests 1/3/5 each complete with captured
`message_stop`, and their assistant blocks equal the full Runtime canonical
responses. One cloned reader emits `AbortError` only **after** its stop was
persisted; other streams reach observer EOF. No completion is missing.

## Vision and independent interpretation

Normal File-ID resolution and native Recipe preparation produce:

| Generation | Recipe spans (start, length, grid) | Historical flags | Fresh encodes |
| --- | --- | --- | ---: |
| 1 | none | `[]` | 0 |
| 2 | A `(951,578,[52,92])` | `[false]` | 1 |
| 3 | unchanged A; B `(2014,578,[52,92])` | `[true,false]` | 1 |

Prepared source SHA-256 values equal the actual uploaded binary hashes. The
passive evaluation wrapper calls original `prepare_request` once and returns the
same prepared object unchanged; no production admission or Vision path is bypassed.
Wire occurrence positions, tool-result nesting, image identity and Recipe span
ordering agree. The final restored frontier 1721 fully covers A's end 1529.
A is reused without encoding; B alone is newly encoded. A live ID is not counted
as paired-KV availability: restoration and historical-span coverage are checked
separately. No eviction/cache-miss recovery was induced.

Independent direct PNG viewing confirms the model recognizes A's red circle,
blue square, green upward triangle, heading and **K7M4** marker. It recognizes B's
**Field Notes**, subtitle, stacked yellow panels with **12 / 84 / 24**, their
labels/captions, burnt-orange header and large empty right side, then meaningfully
compares the designs. Minor imprecision: A's card is not vertically centered;
B's layout is not literally edge-to-edge because of narrow outer margins. These
are retained, not accepted as precise geometry. Visual acceptance is meaningful
recognition, not perfect prose or the model's own completion declaration.

## Complete paired-cache continuity and retention

| Generation | Restored → canonical frontier | Fresh | Historical reuse |
| --- | --- | ---: | ---: |
| 1 | 0 → 655 | 0 | 0 |
| 2 | 655 → 1721 | 1 | 0 |
| 3 | 1721 → 3072 | 1 | 1 |

All 40 Target and three DSpark offsets agree at each canonical frontier. Authority
is settled, consumed/emitted/quiescence frontiers agree, queues/predictions retire,
no cancellation, paired cache publishes after canonical completion. Prompt replay,
full-cache repack and quiescence history replay/repack counters are **zero** for
all retained generations. No image replay, full-cache rebuild or fallback owner.

Reused `tools/phase3_trace_retention.py`: poller stops/joins after serving settles;
final synchronous flush precedes `shutdown_serving()` (which itself closes the
backend) and destruction. Lossless raw trace archives are read back and checked.
**Final memory = archived = complete = 3**, including `trace-02.raw.json.gz`.
Native SSE completions are also **3/3**. No final-trace inference from HTTP success.
Offline `audit.py` passes; Python AST and Node syntax checks pass. No broad suite
or additional real-model run was executed.

## Cleanup and exact qualification limits

Harness PID **2754**, Runtime **90026**, supervisor **90024** have exited; Runtime
and Harness exits are 0, port **18083 is free**. Both rejected startup roots and
the final home/profile/session/workspace/KV root are removed; image hashes were
verified before removal. The newly owned disposable venv, wheel directory and
temporary dependency list are removed after identity/evidence retention. No
renderer/browser was started. Persistent operator installations, old disposable
Phase 1 environment, checkpoint and all Phase 0–3/W1/W2/Q1 evidence are unchanged.
See [cleanup](gate-f0-evidence/cleanup.json).

This qualifies **only this two-image, single-operator loopback macOS official
`read_image` → Files → native Messages → Recipe Vision → paired-MTP conversation**
on M3 Ultra 512 GB, original Flash-1017, Thinking OFF and the recorded configuration.
Not Windows or GUI upload, persistent/multi-user/session-scoped Files, list/quota
recovery, delete/expiry/stale recovery, cancellation, eviction/restart, maximum
memory/context/image capacity, endurance/Q2 or production promotion. No concrete
protocol incompatibility was discovered; no architecture correction is proposed.
One focused local commit only; no push or further execution authorized here.
