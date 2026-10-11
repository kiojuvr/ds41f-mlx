# Single-user localhost Web application

This page owns **Chat Server application state, tool effects and recovery**.
[Chat interface](chat-interface.md) owns Browser interaction;
[operator Chat](operator-chat.md) owns TUI interaction;
[generation frontend policy](generation-policy-frontends.md) owns shared control
ownership. [Operations](operations.md) owns launch/environment instructions.
Application history/journals are not native state or fresh release qualification.

## Status and scope

Browser and TUI are separate frontends over the same Chat Server APIs. The server
proxies the selected runtime: OFF retained native Chat or ordinary-MTP logical
sessions. Profile capability is explicit, not inferred from an endpoint name.
MTP remains text-only with no native persistence/restore; OFF has its separately
bounded image/artifact contract. Runtime session limits still apply—equal
frontends do not grant simultaneous independent sessions on the one-session MTP lane.

Localhost is default. Explicit application `--allow-private-lan` has its own
[network policy](private-lan.md); it is not authentication or Internet/multi-user
qualification. Original OFF runtime/application and subsequent UI/frontend/PDF
acceptances are distinct source-scoped records, not an automatic PASS for HEAD.

## Architecture and launch

```text
Browser: ordinary history, original images, workflow/request journals in IndexedDB
TUI: in-process text history, draft and workflow/request reservations
  -> Chat Server HTTP proxy / certified application-side tools and ledger
  -> selected public runtime session API
  -> runtime-owned execution, canonical settlement and retirement
```

The Web process does not load the model, tokenize independently, mirror KV,
replay retained history or implement a sampler. HTTPX relays live bytes; blocking
control/tool work runs off the event loop. No frontend framework/npm build is
required. Use the separate application environment and already-ready installed
runtime described in operations, not the old M19 checkout model launcher.

## Everyday workflow

Create a conversation, Send, then reconcile its canonical GET outcome. Live
assistant content is provisional; full ordinary canonical history is submitted
for exact continuation, not interpreted as client executable state. New chat
explicitly retires the known old runtime conversation; failure does not forget it.

Tools Auto/Ask/Off governs Browser execution of certified calls, not frozen
protocol declarations. TUI uses Auto with explicit pause/continuation on errors.
Search/HTML acquisition and retained PDF operations are application effects; see
[tool capability](tool-capability-completion.md) and
[PDF workflow](pdf-document-workflow.md). Default document tools retain original
PDFs and selected interpretations, not a growing series of chat images.
Parser reads/search work on MTP. Fresh page observations require an existing OFF
Vision runtime; they do not add image capability to main MTP Chat.

OFF Browser image attachment preserves original bytes through subsequent turns.
MTP does not acquire ordinary-chat images. OFF Save/Restore is exposed only where
runtime capability permits it; a Browser journal alone cannot restore MTP sessions.
Stop uses the known runtime request ID, waits for settlement and can recover a
committed but undelivered suffix. Reload observes; it never automatically repeats
a generation, restore or tool effect. Unsupported/unrepresentable continuation
remains blocked. Exact controls and labels belong to the Chat interface.

## History, uncertainty and recovery

Browser `store.js` retains ordinary messages/original images, protocol settings,
workflow snapshots, pending-request journals, effect reservations, documents and
artifact-linked saves. These are application data, not native proof. Lost history
or old image bytes cannot be reconstructed from bounded GET `last_turn`.
localStorage presentation/selection preferences cannot authorize execution.
TUI has no shared IndexedDB or persistent transcript owner.

Before submission Browser durably freezes request body, base count and application
UUID; TUI retains equivalent correlation in its process. Chat Server forwards
expected-count/application-ID headers and the admitted native request ID.
Reconciliation requires exact next count and matching application/native identity
before canonical response enters history. OFF correlation and production-MTP
reserved-UUID fencing remain distinct runtime contracts. An uncertain request
cannot silently be retried or discarded; explicit nonadmission handling needs
positive GET observation of unchanged/nonbusy state.

Before effect POST, frontend reservation and the living Chat Server ledger fence
`(session, count, native request ID)`. The server obtains completed calls and
positive reconstruction permission from canonical GET, not supplied browser
arguments or partial JSON. Duplicate/concurrent calls reuse known results or
fail on uncertainty. Actual effect results publish before budget fitting; a
fitting failure must not make completed I/O retryable. Reload/Reconcile observes
`/api/tools/result` without effects. Missing ledger after server restart cannot
prove nonexecution: block reexecution, with Browser's explicit unavailable-result
path where valid. This is not crash-safe/distributed exactly-once execution.

For OFF native restore, Browser durably journals a known fresh ID before I/O.
Lost acknowledgement is reconciled by that same ID/frontier, never another restore.
Cold loading may exceed ordinary request expectations; the long restore timeout
is not a loading SLA. Tab close cannot safely imply native restore abort.
Missing/mismatched journals, artifacts, frontier or originals block continuation.
Production MTP restart loses logical state; application history does not permit
native replay/restore to recover it.

IndexedDB write/quota failure prevents an unjournaled action. Preserve the same
origin/profile, originals, native artifact and matching application save. Saves
are explicit user-retained data, not automatic unlimited snapshots. Default DOM
history bounds limit rendering, not stored history or native context.

## Security and limits

- Default loopback/TrustedHost, same-origin JSON mutations, cross-site rejection
  and CSP; application LAN opt-in is a separate documented boundary.
- Text-node/DOM Markdown, no HTML execution, safe HTTP(S) links and original inline
  transcript images; no automatic remote-image loading or client recompression.
- Acquisition checks public DNS and actual peers, redirects, transfer/time/MIME
  limits. PDF parsing is isolated/bounded with no action/form execution or
  attachment extraction. Source bytes and derived observations remain distinct.
- Runtime owns final image/request/sampling/capacity admission. Complete-request
  fitting, not excerpts or character estimates, determines usable tool results.
- Web ingress and aggregate effect ledger have explicit resource ceilings;
  namespace exhaustion refuses rather than evicting unresolved outcomes.
  Successful close removes that namespace. These bounds are not context limits.
- Bounded diagnostic summaries are distinct from live delivery, latest canonical
  outcome and user-retained data; no fixed total-memory promise is made.

Runtime profile values live in [capabilities](capabilities.md); acquisition and
resource values in [budget policy](dynamic-capability-budget.md) and PDF workflow.
Tools are not filesystem/shell/plugin execution. No application capability broadens
native persistence, image, concurrency, security or effect guarantees implicitly.

## Qualification

The [original Web edition](archive/editorial-baseline/web-application.md.txt)
retains the `133adc4` OFF baseline, matched artifact/image/frontier/soak results,
failed collectors, false quota PASS and exact old paths/counts.
[Original Web qualification](../artifacts/web-application/qualification.json)
and its stateful-live receipts remain scoped evidence, not acceptance of later
UI, MTP, policy or PDF changes.

Later owners are [Browser UX](chat-interface.md#qualification),
[frontend controls](generation-policy-frontends.md#installed-real-model-evidence),
[operator acceptance](operator-chat-status.md) and
[PDF acceptance](pdf-document-acceptance.md). Existing collectors are reproduction
tools, not instructions for a fresh campaign during compilation. No model restart,
R1, long-context, release or universal provider/browser/grammar claim is added.
