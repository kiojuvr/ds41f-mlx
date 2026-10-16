# PDF document application acceptance

This is **retrospective application evidence** for its recorded source/runtime
inputs. Counts, ports and service statements are acceptance-time observations,
not present process readiness or later-HEAD qualification. Current behavior and
setup belong to [PDF workflow](pdf-document-workflow.md) and [operations](operations.md).
The [original report edition](archive/editorial-baseline/pdf-document-acceptance.md.txt)
is preserved; acceptance JSON and original receipts/results are unchanged.

Real official **DeepSeek-V4.1-Flash**, actual Chat Server/runtime, not a mock-model
acceptance. Implementation and acceptance are application-layer work, not runtime
requalification, R1, release promotion or a new production Vision capability.
Architecture/run instructions: [retained PDF workflow](pdf-document-workflow.md).
Compact computed receipts and local evidence hashes: [acceptance JSON](pdf-document-acceptance.json).

## Workloads and observed results

| Workload | Real-model result |
|---|---|
| Ordinary 3-page text PDF | Read pages 1/3; correct inventories 17/51 and citations. |
| 100-page operations PDF | Search ranges 1–32, 33–64, 65–96, 97–100; found Sector99 / 1683 on p.99. |
| Noncontinuous references | Same original artifact, p.2/50/99; correct 34/850/1683. |
| More than four pages | Actual read tools for all p.1–12; all twelve inventory values correct. |
| Real public paper | *Attention Is All You Need*, arXiv 1706.03762, 15 pages. Parser search/read located Table 2 on p.8. |
| Figure/table Vision | Isolated p.3/p.8 requests: encoder/decoder layout and Transformer(big) BLEU 28.4/41.8. Explicit derived-observation caveat and page citations. |
| Scanned/image-heavy PDF | 12-page raster-only inventory/table/figure document. Twelve separate real Vision requests; Depot1–12 and all bolts quantities 11–132 correct. No parser text layer. |
| Observation reuse | Same scan artifact p.2/7/12: 22/77/132, blue-bar description; no fresh page inference or acquisition. |
| Transient raster pipeline | Additional fresh 6-page scan: six actual stateless Vision requests, all quantities correct, then p.1/3/6 reuse. No permanent raster rows and no main-history images. |
| Public acquisition/reuse | Model used `document_pdf acquire` for W3C dummy PDF, then retained `read`; exactly one acquire call/receipt, no conversation images. |
| Native Stop/reconcile | Actual active text generation cancellation, canonical GET settlement and subsequent continuation, in MTP and standard-OFF. |
| Stop during page tool | Paper p.4 observation durably reserved; Stop paused the outer workflow. Single page result completed, explicit Continue consumed it, unchanged observation response ID, no duplicate tool message/inference. |
| Real Chrome UX | File-input upload, page count, raster preview without attachment, model reads/clickable tool source, explicit ordinary-Vision attachment preview, reload and additional-page reuse. Catalogue announcement appears only once; committed image history remains zero. |

Known-answer PDFs are deliberately generated test documents, but their reading is
performed by the real model. The external paper and W3C PDF provide ordinary public
sources as well. Unit tests alone were not used as acceptance.

The scan conversation settled **22 native requests**, including twelve actual
page observations, reuse, real-paper figures/tables, cancellation and subsequent
continuation. Main history contained **0 images**, including after twelve scanned
pages. The final transient-raster conversation settled **11 native requests** for
six further real page inspections and reuse. Across this acceptance, **21 isolated
document Vision requests** executed: 18 scan pages and 3 real-paper pages. Reuse
added none. Normal document observation/preview paths retain raster identity and
provenance, not a large PNG per page; final Chrome acceptance also verifies this. Each scan observation
finished normally, without output truncation. Stateless target offsets matched
prompt plus generated frontier on all 40 layers; replay/repack were zero and no
active generation session remained after each stateless request.

The public paper's table/body EN-FR discrepancy (41.8 versus 41.0) was flagged by
the model, not silently resolved. Its table column alignment and generated chart
explanations remain interpretation, not source authority.

## Cancellation and invariants

MTP text workflow and separate active cancellation/reuse checked all 40 target and
3 DSpark offsets against the canonical frontier. Example cancelled MTP frontier:
**1374**; subsequent continued frontier **1460**. Both replay/repack counters were
zero. Standard-OFF cancellation likewise recovered the canonical committed suffix
and continued without re-reading sources. Main application/native identities and
counts are fenced by the unchanged canonical settlement implementation.

An initial Stop collector reached an unsettled/queued request rather than active
generation: application stayed blocked with **no settled outcome / no automatic
retry**, and that session was explicitly retired. This is recorded as a guarded
uncertain outcome, not a successful active-cancellation test. A subsequent fresh
workflow waited for actual streamed content and verified real cancellation. No
frozen uncertain request/history was replayed.

Stop during isolated page inference preserved one tool result and its page-scoped
response identity. No source acquisition or inference was automatically repeated.
Reserved/process-death and cross-policy uncertain-observation non-reexecution, source integrity/expiry,
30K-character offset continuation, parser-only discovery, typed bad arguments,
9 MiB local static PDF upload and error handling also have focused regressions.

## OpenCode / application dogfooding

Actual Mac OpenCode, ordinary Plan review, provider alias `deepseek-v4.1-flash`,
official production runtime. Session **`ses_ed5810d48ffeAqZWglLMQybWB3`**. It read
actual working files/diff and performed real Chat Server status, upload, parser,
page and retained-tool probes. This was read-only review/dogfooding; implementation
was not represented as autonomous OpenCode-authored code.

Concrete frictions addressed:

* OpenCode discovered :8080 was an older installed application while :8090 served
  the working code. At acceptance, the canonical :8080 Chat Server was switched
  to the accepted source; this is not a statement of present service identity.
* Discovery/uncertain observation payloads could falsely mark pages as read.
  Metadata discovery no longer extracts page 1; UI distinguishes referenced pages
  and extracted/observation state.
* Failed visual raster could disappear silently. Page preview now shows raster
  error separately and preserves parser text.
* Production is genuinely text-only. Fresh observation availability/configuration
  is explicit; `--runtime-url` is honored unless an isolated endpoint is configured.
* Search coverage is explicitly parser/retained-observation coverage, never full
  visual completeness. Derived hits retain their authority label.
* Browser journal reload made a pre-lock document object stale, risking repeated
  catalogue announcements. Actual reload/reuse acceptance verifies the fix.
* Local 8 MiB file limits were inadequate for ordinary long scans. Local upload /
  isolated parser input now allow 64 MiB, retaining existing parser CPU/RSS/object/
  raster/output and aggregate-store safeguards. Public network acquisition remains
  8 MiB; no qualified Vision/context limit was enlarged.

OpenCode's alleged double-slicing corruption was not reproduced: each read starts
from immutable cached original parser material. A real 30K-character page regression
reconstructs the exact text across successive offsets. Crash-padding/key-prefix
speculation was not treated as a proven defect. The Chrome collector's initial
DOM-node/return-by-value wait was fixed in the collector, not by changing the app
or injecting browser behavior; the final real-browser run passed.

## Historical validation / service at original acceptance

Focused Python tests: **138 passed**. Renderer/platform/policy Node tests:
**19 passed**. No npm dependencies/build system or new document database was added.
The existing artifact SQLite store owns derived material; runtime owns sessions.

At acceptance, canonical Chat Server: **http://127.0.0.1:8080**, with the existing qualified
**standard-OFF** runtime at :8000 for fresh Vision. The original application
routing treated production MTP as text-only and required OFF page observations.
That routing is historical: the current [PDF workflow](pdf-document-workflow.md)
uses isolated stateless ordinary MTP Vision by default when advertised. These
original receipts are not evidence for that later integration.

Full local receipts: `artifacts/pdf-documents/{text,mtp-cancel,scan,transient,tool-stop,acquire,browser}.json`,
`off-execution.jsonl`, `opencode-review.jsonl`. Runtime/source bytes, policy, canonical
settlement, qualified four-image full-history envelope, replay=0/repack=0, and
applicable target/DSpark alignment were not relaxed to satisfy PDF frontend needs.
