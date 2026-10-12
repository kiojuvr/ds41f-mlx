# Everyday local Chat interface

This page owns **Browser interaction and truthful controls/labels**.
[Application state](web-application.md) owns journals, effects and recovery;
[frontend policy](generation-policy-frontends.md) owns generation ownership;
[operations](operations.md) owns launch. Browser UI does not own runtime caches,
settlement or recipe grammar.

## Gate

The original everyday UI PASS at `9312cdd` is retained in
[Browser receipt](../artifacts/chat-ux/browser.json) and
[qualification](../artifacts/chat-ux/qualification.json). Its static/Python hash
comparisons and the earlier `133adc4` semantics baseline apply to those inputs,
not later network, budget, MTP, generation-policy or PDF changes. Failed/partial/
older-source receipts cannot close a changed boundary. Full original gate wording
and acceptance matrix remain in the source edition linked below.

## Main surface

One current conversation, no ordinary session sidebar. The header shows model
and Ready/Generating/Using tool/Stopping/Recovering/Error, New chat and Runtime / Details.
Diagnostics/Recovery holds runtime IDs, frontiers and persistence metadata.
New chat confirms retirement of the old conversation; DELETE must succeed or
confirm absence before browser identity is forgotten. Active work disables
destructive actions; uncertain idle outcomes are not silently discarded.

Enter sends; Shift+Enter inserts newline. IME composition/229 cannot send. The
composer grows to 190px before internal scrolling. OFF image picker/paste/drop
share original-byte preflight/removable previews, without canvas/transcoding.
Ordinary MTP disables chat image acquisition. PDF upload/drop and retained page
exploration are separate document operations: see [PDF workflow](pdf-document-workflow.md).
A page preview does not automatically consume an image slot or run inference.

Conversation takes precedence over composer, attachments and runtime/debug state.
The composer keeps input, attachment pickers, Think, Tools and Send/Stop visible;
effective workflow and sampling stay inside Runtime / Details. Retained PDFs have
one-line filename/page-count cards that open a PDF context dialog. Referenced
pages are summarized as ranges; the collapsed page list accepts number/range
filters such as `12, 12-18`. Source previews remain separate from chat images.
New user turns journal a UI-only `userPresentation` alongside exact ordinary
messages: the bubble shows the user's text and attachment cards, while generated
PDF instructions are collapsed under Inspect context. Save/Restore preserves this
presentation without adding fields to model requests. Older journals without this
metadata keep their literal bubbles; user text is never stripped heuristically.
Generating is also shown inside the live assistant card; reasoning stays collapsed
inside that card. The user restarted Chat Server and visually confirmed the UI
improvements. Focused Python tests passed (15), as did the Node frontend tests
(22). This user-reported visual check and developer regression coverage do not
refresh the historical Browser gate.

Send becomes Stop during active work. Stop observes/uses the authoritative ID,
requests safe cancellation and reconciles canonical GET; it is not immediate GPU
abort. Reload reconnects/observes without generation/effect POST replay.

## Model controls: exact mapping

- Require READY and supported `generation-policy-v1`/stateful ownership; no legacy
  fallback. All eight sampling types/ranges/defaults and reasoning efforts come
  from the runtime. Unknown or inconsistent capability rejects before generation.
- Thinking Off sends `reasoning_effort: "none"`; On selects an advertised enum,
  not percentages/aliases or frontend defaults. Token-budget thinking is unsupported.
  Reasoning freezes at submission, including unknown outcomes; New chat is needed
  to change it. Stateless reasoning is still request-owned.
- Generation has temperature/top-p/top-k/min-p; Advanced has penalties/seed.
  Reset uses runtime defaults without changing locked reasoning. Sampling is
  editable between user turns and serialized explicitly; tools/continuation keep
  their user workflow snapshot. Effective continuation and next-user-turn choices
  are distinguished when they differ, not silently inherited by the runtime.
- Auto/4K/8K/16K/32K/Custom output choices remain runtime-admitted reservations.
  Auto resolves actual recipe/image-expanded prompt against remaining context and
  the selected profile's output ceiling where present. Positive Custom must fit
  both. Preview observes only; generation rechecks. See
  [capabilities](capabilities.md) and [capacity policy](dynamic-capability-budget.md).
- Tools Auto executes certified pending calls; Ask pauses before a batch; Off
  starts no new effect. Mode changes application execution, not declarations.
  A fixed **128-round runaway breaker** preserves calls and needs explicit Continue
  to reset; it is not an ordinary round quota or automatic retry mechanism.

## Rendering, scrolling and safety

Markdown uses DOM elements/text nodes, never `innerHTML` or raw HTML execution.
Paragraphs/headings/emphasis/lists/quotes/code/tables/rules/safe links are supported,
not full CommonMark. Credential URLs/non-HTTP(S) schemes reject; links use
noopener/noreferrer, transcript images never fetch remote URLs. Inline parsing,
blockquote depth and table separators are bounded. No npm/framework build.

An 80ms render timer is independent of token cadence. Stable live text nodes
append; completed blocks parse once, unfinished blocks remain literal. Canonical
reconciliation renders final messages while preserving unchanged DOM and open
reasoning/tool disclosures by kind. Empty live reasoning stays hidden; only public
`reasoning_content` is shown, collapsed by default. Tool rows show arguments,
results/errors and sources without becoming the main response. Display grouping
never changes stored ordinary messages. Clipboard failures are reported.

Following applies only near bottom. Scroll-up/disclosures detach; Jump to latest explicitly
reattaches. Layout/clamp/image/composer changes do not silently reattach a detached
reader; real scrolling to bottom can. Boundary renders preserve visible anchors.
The default last-100-message view expands in explicit 100-message increments:
a DOM bound, not deletion/compaction of IndexedDB history or native state.

## Termination and continuation

| Observed state | Truthful action/meaning |
| --- | --- |
| Native `length` | Output/context admission boundary; Continue rechecks the complete exact request |
| Breaker / Ask / Off pending calls | Explicit Continue tools or valid unavailable-result action; no hidden effects |
| User Stop | Journalled intent plus canonical cancelled outcome; undelivered committed suffix may appear |
| Disconnect / unknown cause | Neutral interrupted label, not guessed User Stop or output limit |
| Stop too late | Native completion preceded Stop, not fabricated cancellation |
| Tool error | Actual result/error; explicit continuation without effect retry |
| Unknown reservation | GET observation or valid unavailable-result path, never another effect POST |
| Admitted/request identity uncertain | Reconcile/Details; ordinary continuation disabled |
| Busy / unrecoverable / absent | Observe/Stop or explicit New chat; no reconstruction by replay |

Continue response submits exact canonical history as a **new incremental assistant
turn**, not resubmission of the user request or a promise to resume the exact
sentence. Exhausted context cannot be repaired by dropping/replaying committed
history. Continue tools executes only certified outstanding calls or asks the model
to consume already observed results, without rerunning effects. Paused Stop leaves
calls intact; annotations alone grant no execution permission.

User Stop intent is not proof of native cancellation. Tool-workflow pause is
separate from generation abort. A Stop before submission starts no request.
`toolAwaitingResponse` records presentation/workflow state, not authority; recovered
results require explicit Continue. Normal Send remains blocked while requests,
calls/effects or their pending model response are unresolved.

## Recovery and persistence

Settings separates Generation, Thinking, Tools, Recovery and Diagnostics. OFF Save
requires canonical idle settlement, matched native artifact/frontier/application
snapshot and original images. Paused workflow/control/finish annotations survive,
but do not manufacture reservations or permission. Locked reasoning reflects the
saved conversation, not preferences of the one being retired.

OFF Restore selects a saved entry, confirms old-session close, journals a known
fresh ID before I/O and reconciles that ID; never automatically repeat restore.
MTP disables native Save/Restore. A lost runtime session is not rebuilt from the
Browser journal. Recovery menus load metadata lazily, not all saved images/history
on ordinary Send. Storage failure blocks unjournalled actions. The application
state owner defines exact count/identity/effect/restore fences.

## Qualification

The [earlier interface edition](archive/editorial-baseline/chat-interface.md.txt)
retains all original browser checks, measured counts, screenshot and failed
collector/IME/composition attempts, old reasoning/round labels and historical M35
pin observation. Those obsolete labels are not current controls or test failures.
The screenshot is [historical Browser capture](../artifacts/chat-ux/browser.png).

`tools/qualify_chat_ux.py` is the existing isolated-Chrome/current-static-boundary
collector, not authorization for a new campaign here. Real Browser event injection
proves tested handlers, not every OS IME/clipboard implementation; long-DOM data
is display-only, never model history. Later generation/PDF/network changes have
separate receipts. Renderer Node tests are developer tools, not runtime dependencies.
Compilation does not refresh model, image, long-context, R1 or release approval.
