# Everyday local Chat interface

## Gate

**PASS: 68 current-source real Chrome/runtime checks, 90 affected Python tests +
8 subtests, 12 renderer tests.** The receipts and source gate are
`artifacts/chat-ux/browser.json` and `artifacts/chat-ux/qualification.json`.

The `133adc4` stateful Web/runtime qualification remains the semantic baseline,
not automatic acceptance of a changed UX. All five final static sources match the
browser receipt; all 95 tracked production Python sources are identical to that
baseline. A PARTIAL, RUNNING, failed collector or older-source receipt does not
close the gate. **R1 re-verification, release and ds41f-runtime promotion were not
performed by this task.** No runtime restart or user-conversation retirement is required
for the collector: it uses a separate private Chrome context and its own sessions.

## Main surface

One current conversation, without a session sidebar. The header contains the
model name, Ready / Generating / Using tool / Stopping / Recovering / Error,
New chat and Settings. Runtime identities, frontiers, reservations and persistence
information live under Settings > Diagnostics/Recovery. New chat explicitly warns
that the current native conversation will be closed; DELETE must succeed (or
confirm absence) before browser state is retired and a new session is created.
Active work disables destructive lifecycle actions. Uncertain idle outcomes are
not silently discarded: New chat warns about retiring the unresolved outcome.

The composer grows to 190px, then scrolls internally. Enter sends, Shift+Enter
leaves newline input to the browser, and composition/229 key events cannot send.
Image picker, paste and drop use the same preflight and original-byte reader;
previews have individual removal. There is no canvas, transcoding or recompression.
Historical image loading is restricted to supported inline original-byte data URLs.
Limits are explained on rejection, not advertised as a management dashboard.

Send becomes Stop while an operation is active. Stop uses the known admitted
request ID (or observation of the active ID), then disconnects live transport and
waits for canonical settlement. The header says Stopping; it is not a GPU abort.
Reload reconnects and waits on GET; it never sends a generation or tool effect.

## Model controls: exact mapping

- Thinking Off maps to `reasoning_effort: "none"`. On selects `low`, `high` or
  `max`, which map to the pinned V4.1 recipe's **50, 75, 100** respectively.
  Historical `xhigh` is retained in the request and displayed as 75. Thinking and
  effort are separate controls; effort is disabled while Thinking is Off.
- The pinned recipe uses an enum, **not arbitrary numeric 1–100 effort**. Its
  `deepseek-recipe-encoding/src/v4/dsv41.rs` maps Low→50, High/Xhigh→75, Max→100;
  Chat Completion conversion accepts enum names, not numbers. An arbitrary
  slider would falsely advertise ignored/unsupported values. The UI therefore
  exposes the actual three levels and explains this limitation. Adding a 1–100
  protocol would require a separate runtime/recipe change, explicitly outside
  this task.
- The original `protocol.reasoning` and tool declarations freeze before the
  first generation request is durably submitted. Locked controls cannot silently
  rewrite the conversation prefix. Use New chat to choose a different setting.
- Settings > Generation directly sends finite, nonnegative `temperature`,
  `top_p` in 0..1, and positive integer `max_tokens`. No top-k, ignored parameter
  or private sampler is introduced. Initial everyday sampling is 1.0 / .95.
  Reset to runtime defaults selects 0 / 0 / 128, exactly the fallbacks in
  `DeepSeekRecipeRuntimeBackend.make_sampler()` / `max_tokens()`; this does not
  claim a different official model sampler.
- Output presets: Auto, 4K, 8K, 16K, 32K, Custom. Auto is an explicit conservative
  reservation policy: 8192 tokens for text, 1024 if any historical/new image is
  present. It does not estimate remaining native context or bypass admission.
  Custom accepts the recipe's positive u32 domain rather than the old UI 4096
  cap. Runtime context/multimodal admission always remains final.
- Tools Auto executes certified canonical pending calls; Ask pauses before each
  batch; Off never starts a new effect. Changing mode changes client execution,
  not the frozen protocol declarations. The configurable ceiling defaults to
  **32 rounds** (1..128 UI range), not four. A round is one tool batch followed
  by a canonical assistant request. There is no hidden generation/effect retry.

## Rendering, scrolling and safety

The renderer creates DOM elements and text nodes only; it never invokes an HTML
parser or `innerHTML`. Paragraphs, headings, emphasis, lists, blockquotes, inline
code, fenced code with language/Copy, tables, rules and safe links are supported.
Raw HTML remains literal. HTTP(S) URLs with credentials and all other schemes are
rejected. Links use noopener/noreferrer; transcript images never load remote URLs.
This small renderer is intentionally not a complete CommonMark implementation
(e.g. nested lists, reference links and escaped table pipes are not promised).
No npm, build chain, framework or frontend dependencies were added. Inline parsing
is bounded; blockquote depth is capped at 16 and table separators are validated
cell-by-cell to avoid ambiguous whitespace regex backtracking.

Streaming uses an 80ms timer independent of token cadence. Persistent text nodes
receive new text; only newly completed Markdown blocks are parsed once. An
unfinished fence/paragraph remains literal until a safe block boundary. Final
canonical messages are rendered after reconciliation; unchanged message DOM and
open disclosures are reused by kind, not position, so new public reasoning cannot
steal a previously open tool disclosure. Opening live public reasoning survives
canonical completion. Empty live reasoning remains hidden. Adjacent assistant/tool turns are display-grouped
without modifying stored ordinary history, so the response remains the main
surface and tool JSON never takes over the conversation. Reasoning is collapsed
and consists solely of public `reasoning_content`. Tool rows contain names,
queries/URLs, result/error summaries and clickable sources. Copy response and
code Copy use the browser clipboard, reporting permission failure if necessary.

Following occurs only near the bottom. Scroll-up detaches; Latest jumps to the
end and reattaches. Disclosures detach deliberately, and layout/clamp events
cannot silently reattach an unfollowed reader. Actual scroll gestures back to the
bottom can reattach. Boundary renders restore a visible message anchor when
unfollowed; native browser anchoring handles asynchronous image geometry changes.
Image loads and composer resizes follow only while the reader still wants it. Default rendering remains bounded to the last 100 ordinary messages;
explicit earlier-history expansion adds 100 at a time. This bounds default DOM,
not user-retained IndexedDB history or native context.

## Termination and continuation

Reasons are shown below the response, not buried in Settings:

| State | Explanation / next action |
| --- | --- |
| Native finish `length` | Output limit reached → Continue response |
| Tool ceiling | Paused after N rounds → Continue tools / Stop / unavailable result |
| Ask / Off with pending calls | Approval / paused → Continue tools / unavailable result |
| User Stop | Recorded user intent, committed state reconciled; a small suffix may be recovered |
| Disconnect / unknown cancellation cause | Response interrupted, canonical state reconciled; never guessed to be User Stop |
| Stop too late | Response completed before Stop took effect |
| Tool error result | Tool failed with actual error → Continue without result; no Retry |
| Tool reservation outcome unknown | Not retried → observe/reconcile, or explicit unavailable result |
| Admitted/network outcome uncertain | No ordinary Continue → Reconcile / Details |
| Native busy after observation window | Still running → Reconcile / Stop |
| Unrecoverable/closed conversation | Cannot safely continue → New chat / Recovery |

Continue response sends the existing exact ordinary canonical history as the
next native incremental assistant request. It does not resubmit a user turn or
prefill/replay a prompt. Its text is a subsequent model turn, not a guarantee that
the model resumes the exact interrupted sentence. Continue tools executes only
outstanding certified calls, or, if their results were already observed, requests
the next assistant turn without executing those calls again. Stop on a paused
workflow leaves canonical pending calls intact and records a UI pause; it does
not manufacture results or perform effects. Continue without result is explicit
and retains the existing unavailable-result settlement path. These operations
are separate from GET-only canonical reconciliation.

Stop intent is an application journal annotation, not proof of generation
cancellation: the native cancelled outcome is required for that classification.
Stopping a client tool workflow is separately a recorded client pause, not a GPU
or generation abort. A known user Stop
which arrives after native completion is labelled too late. Unknown/legacy
cancellations remain neutral “Response interrupted”, never blamed on the user or
mislabelled as an output limit. A Stop before submission does not start a request.

The reservation is still saved before tool POST. Reload observes
`/api/tools/result`, never repeats POST. Results are deduplicated by tool-call ID.
`toolAwaitingResponse` is application workflow presentation/journaling, **not
execution permission**: a recovered completed result needs an explicit Continue
tools to ask the model to consume it. Normal Send stays blocked while canonical
calls, effects or a tool response remain unresolved. The Web server's canonical
certificate, effect ledger, SSRF/redirect/size/content-type guards are unchanged.

## Recovery and persistence

Settings is divided into Generation, Thinking, Tools, Recovery and Diagnostics.
Save requires canonical idle settlement and uses the existing native artifact /
application snapshot frontier pairing, including original images. UI finish/cause
and paused workflow annotations are retained; received results awaiting a model
response still have inline Continue tools after fresh restore. No reservation or
execution permission is manufactured by those annotations. Locked Thinking controls
always reflect the saved prefix, not the preferences of the conversation just closed. Restore uses
that saved entry, never an artifact path input; it closes the current native
conversation only after explicit confirmation, journals a known fresh restore ID
before I/O, and reconciles that same ID. It cannot automatically repeat restore.
Low-level discard/unavailable-result controls appear only in the relevant state.
A missing old runtime session is not reconstructed from history. Normal sends do
not enumerate old application sessions or load every saved image/history snapshot.
Recovery menus read small metadata projections via a readonly cursor only when
Settings is opened; IndexedDB schema/version and snapshot contents remain compatible.

There is no runtime/state-authority change: worker execution, committed delivery,
cancellation settlement, exact-prefix admission, canonical request identity,
no replay, tool reservation/certification, image identities and native persistence
remain the `133adc4` semantics. Only static client files changed in production.

## Qualification

Run `tools/qualify_chat_ux.py` against the already running Web and runtime and an
isolated Chrome CDP endpoint. It records incremental receipts and an actual screen
capture, checks production static-file hashes at both ends, uses real generation,
search/fetch, Stop, reload, offline network loss, effect-response loss/observation,
original image picker/paste/drop paths, save and fresh native restore. Failed and
older-source attempts must be retained and must not be interpreted as PASS.
Paste/drop input events and IME events are generated inside the real browser;
this qualifies those event handlers, not every OS clipboard/IME implementation.
Long-DOM testing is display-only synthetic data, never submitted as model history.

The final real-browser receipt covers:

| Qualification | Evidence |
| --- | --- |
| Text, Markdown, code Copy | Actual generated blocks and clipboard readback |
| Long output / continuation | 2048-token tutorial, native length stop, inline next request |
| Follow / reading / Latest | In-stream detached position and explicit reattachment |
| User Stop | Native cancelled settlement, no false output-limit label |
| Reload / uncertainty | Active reload, real offline transport loss, same application nonce; GET only |
| Tools >4 rounds | Seven sequential fetches, 5-round pause, Stop/Continue; default remains 32 |
| Search + fetch / failure | Actual sources and HTTP403 result; no Retry/effect replay |
| Lost effect response | Real effect executed once, reload result observation once |
| Paused workflow persistence | Received result saved before model observation, fresh restore, inline Continue |
| Approval | Ask pauses before POST, approved effect occurs once |
| Thinking | Actual low request; immutable prefix and public-only/possibly empty reasoning |
| Disclosure geometry | Display-only fixture preserves tool kind and live public-reasoning open state |
| Vision | Picker, remove/drop, exact-byte paste, historical display and zero old-image encodes |
| Image save/restore | Paired frontier **1167**, exact JPEG hashes, fresh native continuation with zero old-image encodes |
| Restore control truth | Off conversation → saved low prefix; locked UI reflects Think 50 |
| UI bounds / input | 2000 display-only messages → 100 DOM messages; composition events, Enter/Shift+Enter and 190px cap |

All recorded canonical snapshots have zero prompt replay and full-cache repack.
The screenshot is `artifacts/chat-ux/browser.png`. Failed collection assumptions
(background send awaited, premature observation, requiring nonempty public
reasoning), older-source attempts and the actual composition event-wiring failure
are preserved. The latter was fixed with real event listeners and verified before
any generation and again at the end; do not use those earlier attempts as current
acceptance evidence.

Affected Python gates cover live delivery/backpressure/retirement, cancellation,
multimodal cancellation, request policy, recovery and Web origin/tool boundaries.
The separate historical M35 evidence pin test fails on `internal_mtp.py` SHA256
(`6821d84…` versus its older receipt's `6a6169a…`); the actual source is identical
to `133adc4`. That pre-existing historical pin mismatch is recorded, not repaired
by rewriting a qualification receipt or claimed as an affected UX regression.
Node renderer tests are optional developer tooling, not a runtime/build dependency.
No 200K/1M, full Vision, R1, model restart or release qualification is implied.
