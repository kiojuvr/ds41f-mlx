# Runtime-authoritative capability budget (standard-OFF)

Production defaults expose supported capability, not smoke-test quotas. Dynamic
capacity comes from runtime; ordinary defaults, resource/security hard ceilings
and qualified model envelopes are different policies. No R1/release/promotion is
implied by this implementation or its focused browser receipt.

## Output and admission

`max_tokens: "auto"` is a runtime HTTP extension. It is normalized to a minimum
positive placeholder solely for recipe conversion. The **actual request** is
converted/rendered/tokenized with the existing recipe/tokenizer, and its inline
images undergo the existing original-byte multimodal expansion. The prepared
request receives a resolved output reservation; the decoder uses that value,
not the placeholder. Explicit positive integer output reservations remain
supported. Omitted output retains the legacy API default 128 for wire compatibility;
the everyday UI defaults to Auto.

Shared preparation applies a total qualified envelope:

- Text: `min(checkpoint max_position_embeddings, 1,048,576)`.
- Vision (including retained historical inline images):
  `min(checkpoint max_position_embeddings, MAX_MULTIMODAL_CONTEXT=8192)`.
- `remaining = qualified total - len(actual expanded recipe token IDs)`.
- Auto reserves `remaining`; Custom must fit it. No invented fixed 8K/1K default,
  frontier-only estimate, char/token ratio, above-checkpoint extrapolation or
  separate unverified output ceiling is used. The checkpoint does not publish a
  separately admitted output ceiling here; the receipt reports that as null.

The prompt count includes history, new user input, complete tool results, fixed
reasoning/tool declarations and recipe structural tokens. The envelope includes
output as well, not a 1M-output guarantee or fresh 1M prompt plus generation.
Preparation rejects overlength before native mutation; custom reservations also
use this guard. It is newly explicit HTTP envelope enforcement, not a relabeling
of the older very-long-context maintenance/core proof.

`POST /v1/sessions/{id}/budget` accepts the same complete request as generation.
It uses **the same preparation and stateful admission validator** as actual
stream reservation: model alias, session busy/recovery state, exact generated
prefix and original image identities. It does not load/encode images on GPU,
create KV, append history, reserve a response, execute a tool or generate tokens.
It returns the current request_count and `binding: false`. A preview is observation,
never execution permission; actual generation prepares/rechecks again before its
existing busy/response reservation. Expected-request-count fencing remains final.
Web `/api/budget` additionally checks its observed count. No preview retry can
silently become a generation/effect retry.

The UI previews before submitting generation and journals the budget. A rejected
preview records “No generation submitted” and an explicit admission/capacity
boundary, not a fabricated unknown generation. Auto length termination is recorded
as `context_capacity`; Custom length as `output_limit`; Stop remains canonical
`user_stop`. Context-boundary Continue rechecks the full exact request and cannot
replay/drop committed history to make space. Exhausted histories require New chat
or an eligible saved state; partial ordinary responses/results remain visible.

## Tools and external resource limits

- Tools Auto has no ordinary round quota. It continues until final response, Stop,
  admission/capacity or tool failure. **128 rounds is a fixed runaway circuit
  breaker**, not an editable routine setting. It preserves pending calls, labels
  the reason and requires explicit Continue to reset. Ask/Off semantics remain.
- Thinking On defaults to **100 / max**. Supported choices remain 50/75/100, not
  arbitrary 1–100. Existing frozen conversation prefixes and saved preferences
  are not rewritten. Thinking Off and sampler defaults are separate controls.
- Search default **10**, max **20**: a provider latency/noise bound, not model
  capacity. Provider fallback/error attribution remains existing behavior.
- Fetch network body ceiling **8 MiB**, total transfer deadline **45 seconds**,
  at most **8 redirects**. Public URL/redirect checks remain and the actual socket
  peer is checked before GET. The connected socket is shut down at the absolute
  deadline, including stalled/trickled headers; DNS resolution remains subject to
  the OS resolver rather than a separately cancellable resolver. Compressed bodies are refused rather than introducing
  an unbounded decompression path. A body hitting the byte ceiling returns usable
  partial data with `network_truncated` instead of discarding a normal large page.
- HTML readable-text extraction excludes script/style/noscript/template; raw tags
  are not fed as body text. Non-HTML text remains text. Original Vision bytes are
  unrelated and never altered by this extraction.
- Model-facing excerpts have a separate **1 MiB UTF-8** memory/resource ceiling
  (JSON escaping and metadata add serialization overhead).
  It is not a token estimate or ordinary 4,000-character excerpt quota. Returned
  counts, offsets, truncation and termination reason are explicit. `offset` allows
  an intentional fetch continuation; no automatic redownload/effect retry occurs.

After a tool batch executes, the Web ledger publishes its completed actual effects
**before** context fitting. Original source results are retained in the bounded
per-session ledger until replacement/close. It constructs the complete next model
request with **all** batch results and checks actual runtime recipe admission.
When needed, only not-yet-model-observed fetch excerpts are reduced by bounded
candidate search; every candidate uses the runtime budget endpoint, never a local
token estimate. The selected candidate was actually admitted; tokenizer length
need not be assumed perfectly monotonic. Context truncation and next_offset are
included in model-facing JSON and visible tool results. Historical results and
committed token prefixes are never edited.

If even structural/non-fetch batch content cannot fit, completed effects remain
returned/observable with `budget_error`; no effect is re-executed. The UI pauses
explicitly and continuation must pass a fresh admission check. Transport loss still
recovers by GET observation; response consumption still needs explicit consent.
The ledger is application effect bookkeeping, not model/session authority.

## Evidence and limits

Focused final-source gate: **22 real Chrome/runtime checks PASS, 84 Python tests +
8 subtests PASS, 15 renderer/platform tests PASS**. Receipts, exact scope and retained
failed/older-source attempts are indexed in `artifacts/dynamic-budget/README.md`.

`tools/qualify_dynamic_budget.py` uses a private real Chrome context with an isolated
new-source standard-OFF runtime/Web pair, without restarting the user's original
process/session. It records source hashes, exact nonbinding preview, real Auto
admission, reload, safe Stop and fresh restore, with zero replay/full-cache repack.
It also checks a Custom-capacity rejection and Stop after preview without any
native request/count advance. CPU tests independently exercise real recipe tokenization,
original-byte Vision expansion without GPU encoding, model/qualified boundary
intersection, exact prefix guard and fitting each fetch candidate through actual
recipe preparation. These tests are not GPU/1M/full Vision/R1 evidence. Historical
Chat/LAN receipts remain historical source-specific receipts, not automatic closure
of this larger admission/tool-policy change.
