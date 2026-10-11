# Tool Capability Completion — source-pinned milestone accepted

Baseline `c8e1ee5` was not reset. The earlier working-tree transport/MIME/batch
fixes were audited and retained. The source-specific intermediate receipt
`artifacts/tool-capability-audit/cpu-attempt-1/receipt.json` is unchanged and
historical, not final acceptance. The separate final milestone receipt is
`artifacts/tool-capability-completion/final-attempt-1/receipt.json`.
It accepts the implemented Tool Capability Completion slice on its pinned
macOS/Apple M3 Ultra standard-OFF source/dependencies, with 31 explicit cases.
It does not authorize R1, release, runtime promotion or new maximum-envelope claims.

## Application ownership and effect semantics

`web_artifacts.py` holds immutable acquired original bytes plus transfer identity
(source/final URL, MIME, redirects, truncation, SHA-256). Random 256-bit receipt
capabilities are same-origin application references, not filesystem paths or
model/cache authority. Every URL call is an explicit acquisition; there is no URL
cache, implicit retry, general-purpose download manager, or document database.
`artifact_id` requests are local interpretations and never call transport.

SQLite publishes a transfer's bytes and metadata atomically. Before GET, a write
transaction reserves the maximum transfer plus 256 KiB worst-case JSON metadata
space, serializing staging writers across processes. Connections close explicitly
on settlement. Storage refusal occurs before download. A failed publication
following completed download is explicitly `uncertain_external_effect_outcome`,
not retry permission. A process crash remains uncertain, with browser/effect
reservation fail-closed behavior; this is not exactly-once network execution.

Default staging path: `~/.local/share/ds41f/web-artifacts.sqlite3`; override with
`DS41F_WEB_ARTIFACT_DB`. Directory/file permissions are 0700/0600 when created.
Logical storage is 256 MiB for original bytes, metadata and interpretation JSON,
with 512 transfers and seven-day non-sliding retention. Expiration rejects reads;
physical deletion is lazy on the next acquisition/write or explicit
`ArtifactStore.cleanup()`. `delete(id)` explicitly removes a transfer and its
interpretations. SQLite secure deletion is enabled; freed database pages may
remain allocated/reused. These are logical byte limits, not a guaranteed physical
file-size or secure-storage/encryption promise. No live receipt is evicted to fit
another acquisition. No runtime history is deleted by staging cleanup.

PDF interpretation results are immutable, page-scoped and keyed by selected
mode, raster policy, worker source hash and installed dependency source/native
hashes. The exact generated PNG is retained. Additional pages reuse the original
PDF, not its URL. Parser temporary files are private, per-invocation and removed
on settlement. A page interpretation is not a second external download effect.

## Supported tool surface (source-specific qualified boundary)

| Tool | Surface | Model material |
|---|---|---|
| `web_search` | Existing Exa/Parallel discovery and attributed fallback | Bounded title/snippet; exact unsliced source URL |
| `fetch_url` | Exactly one `url`/`artifact_id`, optional character `offset`/`length` | Retained readable text, actual runtime-fitted excerpt and next offset |
| `fetch_image` | Exactly one `url`/`artifact_id` | Original PNG/JPEG/WebP as an actual inline image content part |
| `fetch_pdf` | Exactly one `url`/`artifact_id`, `pages` (default [1]), `mode` (`auto`, `text`, `visual`), optional text `offset` | Metadata/page count, selected text/links and/or actual PNG page image parts |

Image MIME and magic must agree; complete single-frame decoding, pixels and
aspect are validated by a CPU helper shared with manual image preparation. Web
images are not recompressed, resized or transcoded. Base64 only lives in ordinary
image transport parts, never model-facing tool text. The existing recipe accepts
images inside tool results and produces canonical image spans. Runtime multimodal
preparation remains responsible for expanded positions, checkpoint binding and
complete-request admission. Historical encoding still uses the existing frontier
skip and exact-original-byte identity checks; no alternative Vision encoder was
introduced. Observed image parts are saved in ordinary IndexedDB conversation
history/snapshots just like manual attachments, independent of staging expiry.

PDF `auto` uses readable selected-page text. Missing/short text, raster images or
vector paths request Vision; `visual` explicitly inspects figures/tables/layout.
Auto is a conservative heuristic, not a proven semantic classifier. Text and
visual material may both be returned. No whole-document text injection or OCR
dependency. Text offsets support retained continuation after budget fitting.

### PDF dependency identities and security

The `web-binary` optional dependency group pins:

- `pypdf==6.10.0` (BSD-3-Clause): mature document graph, metadata and link inspection.
- `pypdfium2==5.3.0` (Apache-2.0/BSD-3-Clause wrapper; bundled PDFium BSD-style and
  third-party licenses): mature native text extraction and page rasterization.
  The installed build reports PDFium `145.0.7616.0`.

Versions and executable Python/native source hashes are included in interpretation
identity. This avoids implementing a PDF parser or using OCR as the main reader.
Install with `uv pip install --python .venv/bin/python '.[web-binary]'` in the
existing runtime environment (this extra does not provision the runtime itself).
Default tool registration checks PDF dependency availability/identity before
advertising the capability or starting an acquisition.

Parsing is a one-shot subprocess. It never initializes PDFium form environments,
JavaScript or action-processing APIs, nor installs external-resource callbacks.
Reachable active/embedded capabilities are refused (JavaScript, additional
actions, forms/XFA, launch/submit/import, remote GoTo, embedded files and rich
media). Static initial page destinations are ignored, not executed. Discovered
HTTP(S) links are inert text; credential/active-scheme links are omitted. No
attachment extraction, recursive document expansion, shell execution or parser
network access is implemented. This is process/resource isolation, **not an OS
network/filesystem sandbox** or a claim of immunity to native parser defects.

## Numerical classification and replacement audit

Q = qualified model/protocol envelope; S = security boundary; R = resource
circuit breaker; P = provider discovery/noise policy; L = legacy replaced below.
Resource limits never grant model admission.

| Limit | Class and current decision |
|---|---|
| Acquisition 8 MiB + overflow probe | R retained; bounds transfer/staging, probe excluded from hash. Binary prefixes are retained but never parsed as complete files. Text prefixes remain usable. |
| Acquisition 45 s / 8 redirects | R/S retained across transfer and redirects, public DNS and actual-peer checks before every GET, no proxy trust or compression. OS DNS itself remains non-cancellable. |
| MIME header 4096 characters | R bounds retained header metadata before body read; source/final URL escaping is covered by the 256 KiB pre-transfer metadata reservation. |
| URL 8192 characters | S/R; reject, never slice resource identity. |
| MCP 1,000,000 bytes | R retained exactly; explicit overflow error, never truncated JSON success. |
| MCP 20 s | P/R existing inactivity timeout; not a proven absolute trickle deadline. |
| Search default 10 / maximum 20; title 200 / snippet 900 | P retained for discovery noise; old URL 500-character slicing removed. |
| Old tool batch 8 calls | L removed; nonempty protocol validity plus 1 MiB aggregate call serialization and 256-character call IDs replace fixed count. Whole-batch structural and built-in semantic preflight precedes effects. |
| Tool batch 180 elapsed seconds | R checked between operations; defers untouched calls with explicit NOT-executed outcomes. It does not cancel an in-flight operation or make OS DNS/MCP a hard absolute deadline. |
| Tool batch result 32 MiB | R aggregate serialized messages/displays, with worst-case classified receipt envelopes reserved for every remaining call before effects. Oversized acquired results return retained identity and explicit local-next-action; later calls can be deferred without execution. |
| Old fixed 1 MiB fetch excerpt | L removed; a 4096-character discovery default (P, not a hard ceiling) avoids gratuitous early context consumption; explicit `length` can select the full bounded retained text, followed by actual runtime fitting. No char/token estimator. |
| Ledger 32 namespaces / 256 MiB | R; reserve twice (32+1) MiB per batch before effects for source/fitted representations. After settlement charge actual serialized source/fitted bytes. These are serialization circuit breakers, not exact Python heap measurements. Uncertain entries are not evicted. |
| Web request 100 MiB | R retained for inline images/serialization. Direct OFF runtime ingress still needs a separate body-bound review; Web does not protect direct callers. |
| PDF selected pages default 1 / maximum 4 | Default is capability policy; maximum is local raster/output R (not four new images per conversation). Additional pages are explicit local requests. |
| PDF 10,000 pages / 100,000 graph visits | R bounded inspection; no automatic context injection. |
| PDF metadata nine standard Info fields / 1000 characters each | P bounded discovery, explicit metadata-truncation flag; arbitrary Info keys cannot inject document-wide text. |
| PDF page text 1 MiB | R extraction allocation/UTF-8 bound per page, explicit refusal, not context accounting. |
| PDF page dimensions 14,400 points | R inspection; raster must ALSO fit the existing Vision pixel/aspect envelope. |
| PDF raster scale 1.5 (108 dpi) | Interpretation policy; fixed, not adjusted to fit model capacity. Original PDF unchanged; generated PNG retained. |
| PDF worker 15 CPU s / 20 wall s / 32 MiB output | R independent parser circuit breakers. Linux additionally has 1 GiB virtual-memory limit. macOS rejects useful RLIMIT_AS; parent samples RSS every ~50 ms and kills above 512 MiB. Sampling can overshoot and is not an instantaneous hard memory guarantee. |
| Vision 4 historical images / 8192 context / 1014 expanded positions per image | Q unchanged; fetched images and PDF pages consume the SAME full-history envelope as manual images. |
| Vision 16 MiB encoded image / 4,194,304 pixels / aspect 1:2–2:1 / single frame | Q/R unchanged; acquisition's separate 8 MiB R is tighter for Web original bytes. |
| Text 1,048,576 intersect checkpoint limit | Q unchanged, runtime request preparation/admission sole authority. |
| Legacy run_turn 4 rounds / output 512 | L removed; Output Auto and explicit 128-round runaway R, matching browser policy. The normal browser still uses canonical streaming/reconciliation, not this compatibility loop. |
| Live sessions 4, UI rounds 128, existing observation/transport timeouts | R unchanged, not tool/context quotas. |
| IndexedDB history/saves | Browser quota with existing persistence failure behavior; no new aggregate browser-storage guarantee is claimed. |
| Separate MTP 1 MiB and restricted tool profile | Q/S/R unchanged, not an OFF tool/binary envelope. |

Mixed fitting previews the actual complete next request, including history,
structural recipe text, tool text, new and historical images and output reservation.
Only current unobserved text can be shortened; images are never removed,
transformed or recompressed to fit. A failed preview retains completed acquisition
and canonical result parts, exposes the admission reason and pauses consumption;
Continue rechecks admission, not download. Committed text/image history is never
edited. Compact disclosure stays `Used N tools`, with safe sources, filename,
selected PDF page and short failure/truncation state, not parser JSON/base64.

## Final evidence boundary

The final receipt records **142 CPU tests plus 8 subtests**, **15 renderer/platform
tests**, and source-stable real Chrome/runtime/GPU probes from development
attempt 8. CPU evidence is explicitly distinguished from actual model consumption.
Safety/parser/resource tests include malformed and animated/oversize images,
DNS/private-peer/redirect refusal, active/malformed PDFs and an actual native
raster output-ceiling refusal with original PDF bytes retained.

Real probes cover PNG/JPEG/WebP Vision, HTML retained offsets, image/PDF/text
reload BEFORE consumption, canonical effect dedupe, Stop after Vision consumption,
completed acquisitions retained at admission exhaustion, explicit Continue without
redownload, and native save/fresh restore with matching original-byte browser
history. Runtime diagnostics demonstrate **zero historical image encoding,
prompt replay and full-cache repack** on ordinary/restored continuation.

The representative research task autonomously searches primary sources, reads
Harvard HTML, views its architecture PNG, acquires the 15-page arXiv paper once,
reads page 1, views retained page 3, reads retained page 4 and finishes a cited
answer. Ordinary Auto-reservation continuation answers `5`, then reload preserves
the history. Vision must be planned within the existing full-history envelope;
reading too much text before a later image can legitimately exhaust admission.
The failed oversized-reservation/order probes are retained, not relabelled PASS.

All development/failed attempts remain under
`artifacts/tool-capability-audit/development-attempt-*`; the earlier intermediate
receipt is unchanged. Source changes invalidate this source-specific acceptance.
No new 1M maximum-context GPU, Linux, OCR-accuracy, OS-sandbox or exactly-once
network guarantee is claimed. Native artifacts still require matching browser
ordinary history. Direct OFF ingress and absolute OS-DNS/MCP cancellation remain
separate reviews. No R1/release/promotion work was run.
