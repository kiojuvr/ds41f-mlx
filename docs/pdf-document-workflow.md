# Retained PDF document chat

PDFs are document resources, not a sequence of conversation images. This is an
application change; no runtime/session/cache/generation policy was changed.

## Architecture and authority

* Existing `ArtifactStore` retains immutable original PDF bytes, acquisition
  receipt, capability ID, SHA256 and expiry. `(original SHA256, 1-based page)` is
  the source identity. Uploads and public acquisition use the same original-byte
  authority. No automatic reacquisition occurs on expiry or interpretation error.
* Existing isolated safe PDF worker supplies **parser-derived** metadata, page
  count, page text and PNG raster. Discovery inspects metadata without extracting
  page 1. Parser metadata/text are cached with worker/dependency policy identity.
  Document rasters are transient: source previews and stateless observations do
  not permanently store a PNG for every inspected page. This avoids exhausting
  staging storage on image-heavy books; legacy `fetch_pdf` raster caching stays
  unchanged for explicit legacy users.
* `document_pdf` exposes `acquire`, `info`, `read`, `search`, `observe` and
  `observation`. Acquisition is text-only; read has 12,000-character page windows
  with offsets/`has_more`; search scans 32-page ranges and returns `next_page`.
  Read can select noncontiguous pages. Search explicitly distinguishes parser
  matches and model-observation matches; it never claims complete visual search.
  An empty text layer is not evidence that a scanned page lacks the query.
* `observe` renders exactly one selected page and sends one **stateless**, bounded
  Vision request to the existing runtime `/v1/chat/completions`: negotiated
  `generation-policy-v1` sampling defaults, reasoning none, 2048 output tokens.
  Main chat receives only text/observations, not that request's image bytes.
  The temporary raster is released after the request; its identity/provenance,
  not its large PNG payload, is retained with the observation.
* Page observations are **model-derived, non-authoritative**, never saved in the
  parser `text` field. They retain source SHA/page, raster identity, model alias,
  prompt policy, parser policy, generation settings, advertised runtime dependency
  identity, response ID, finish reason and truncation. Truncated/unreadable material
  is not promoted to original source. Check the original page for consequential
  values; parser-flattened tables also require care with inferred column alignment.
* A durable 64 KiB observation reservation precedes inference. Concurrent owners
  cannot run the same page observation twice. Completed observations are reused;
  reserved/uncertain outcomes block another inference for that identity/policy,
  including after process death or a prompt-policy version change. No automatic retry, alternative endpoint fallback
  or URL reacquisition is used. Storage publication failure leaves the reservation
  blocked. Source bytes remain separate and unchanged.

Public URL acquisition remains bounded at 8 MiB. Local original PDF upload and
isolated parser input allow 64 MiB so ordinary long scanned documents need not
fit a small public-fetch ceiling. Existing parser CPU/RSS/object/page/raster/output
limits, 256 MiB aggregate artifact store and seven-day staging lifetime remain. This is selective exploration of potentially hundreds of pages, not
whole-document prompt injection or unlimited OCR. Ordinary text excerpts and
current tool results still use actual runtime budget admission. No conversation
compaction, replay, recurrent cache repack or historical image rewriting is added.

## Vision profiles and the four-image envelope

Production `mtp-serving-v1` is still text-only. Parser reads/search and reuse of
already retained observations work there. **Fresh page Vision requires the existing
standard-OFF Vision runtime**, either the Chat Server's runtime or explicitly
`DS41F_DOCUMENT_VISION_URL=http://…`. The status/UI exposes availability; an
unsupported profile fails before inference reservation. This configuration does
not expand production capability or relax any qualified limit. A configured
endpoint is trusted operator configuration, never a document-provided URL.

A stateless page request has one image and no historical image messages. It uses
existing Vision preparation/envelopes unchanged. Main retained chat's image count
therefore does not grow with inspected PDF pages. The default Browser/TUI tool
surface uses `document_pdf`, not legacy `fetch_pdf`; the latter remains available
on the server for explicit legacy clients with its ordinary Vision semantics.
Existing sessions' frozen declarations are not rewritten.

For rare direct visual confirmation, Browser's page dialog offers **Add this page
to ordinary chat Vision** only on a Vision-capable main profile. This explicitly
uses one of the existing four full-history slots, through the existing attachment
validation. A UI-only preview uses no slot and runs no model request.

## Browser workflow

1. Create a chat, **Add PDF** or drop/paste a PDF (64 MiB local original-byte ceiling), then send a
   question. A text reference announces document name, page count and capability
   once; no image or whole-document content is appended by upload.
2. Model acquires a public PDF or explores uploaded retained pages via the tools.
   The document panel shows name/count/state, referenced pages and observation
   state. Tool citations link to original-page previews; model observations have
   an explicit **NOT original source** label.
3. Open an arbitrary page by number, inspect parser-rendered source raster,
   parser text and separately labelled derived observation, or download original
   PDF bytes. Read more pages in later turns without re-upload/download.
4. Browser reload preserves document references in the existing UI journal, not
   native restore/replay. Removing a panel reference does not delete immutable
   source bytes or rewrite already committed messages.

Upload receipt ID is saved **before** POST. Lost acknowledgements are reconciled
with GET at that known ID; the upload is never automatically repeated. Refresh
is source/cache discovery, not another model observation. Extracted/indexed pages
are not falsely displayed as pages referenced in model tool results.

Main Stop/canonical reconciliation and tool-effect ledger remain the existing
protocol. Stop during an isolated observation pauses main continuation; an already
reserved bounded page request can finish and publish its single result. Explicit
Continue consumes that result, never resends the page inference. Unknown effects
remain blocked. Existing tool-batch/runaway circuit breakers are not increased;
expensive observations are best requested sequentially, one page per model turn.

## Real application acceptance

See [PDF acceptance](pdf-document-acceptance.md). Reproducible collectors:

```sh
PYTHONPATH=. .venv/bin/python tools/qualify_pdf_documents.py \
  --app http://127.0.0.1:8080 --phase text --public-paper
# On a ready, existing standard-OFF Vision runtime:
PYTHONPATH=. .venv/bin/python tools/qualify_pdf_documents.py \
  --app http://127.0.0.1:8080 --phase scan --paper-artifact <retained-paper-id>
# Dedicated headless Chrome/CDP, no npm/build framework:
PYTHONPATH=. .venv/bin/python tools/qualify_pdf_browser.py \
  --app http://127.0.0.1:8080 --chrome http://127.0.0.1:9223
```

Collectors save full local transcripts/receipts under ignored
`artifacts/pdf-documents/`. They do not retry uncertain POSTs. This is document
application acceptance, not runtime requalification or release promotion.
