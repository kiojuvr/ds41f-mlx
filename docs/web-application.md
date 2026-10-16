# Local Web Chat application

Web Chat is the **sole Chat frontend**. The TUI is an [operator console](operator-chat.md).
[Browser interface](chat-interface.md) owns controls; [generation policy](generation-policy-frontends.md)
owns workflow snapshots; [operations](operations.md) owns deployment.

## Product and ownership boundary

```text
Browser tab: multiple conversations, per-conversation history/draft/workflow/journal
  -> Web Chat: HTTP proxy, application tools, bounded effect ledger and idle leases
  -> optional internal logical runtime session per conversation
  -> existing ProductionScheduler queue, ONE native-active BatchGenerator
  -> existing paired target+DSpark authority and canonical settlement
```

Conversation is not an executable resource. New Chat adds a tab-memory conversation
without destroying its predecessor or allocating native state. First Send creates
an internal known-ID runtime correlation. Switch changes only the visible
conversation; hidden requests continue, and all asynchronous outcomes, Stop intent,
attachments, tools and history remain attached to their originating conversation.
Delete removes an idle conversation from this tab; cleanup is an internal side
effect, not a button required to free a native slot. Users do not manage session IDs.

Ordinary MTP remains default. Logical retention allows **16** runtime conversations;
generations from distinct conversations queue through the existing single-active
lane. Each canonical conversation pin uses the SAME upstream paired authority,
with 16 possible logical pins plus two publication slots. There is no second
Scheduler/cache owner, batching, multi-active decode or numerical change. No pin
is evicted to replay an existing conversation. Resource exhaustion refuses safely.

## Lifetime and orphan cleanup

Conversations, original images, drafts and request/effect uncertainty journals live
in tab memory only. `store.js` no longer reads/writes IndexedDB history or native
artifact saves; reload/close starts a fresh UI, with no archive or restart restore.
Legacy browser storage is left untouched but is not read. No persistent database,
accounts or native MTP persistence are introduced.

A living tab renews its internal correlations every 30 seconds through `/api/keepalive`.
The Web process retires its idle/orphan correlations after **600 seconds**, checked
every 30 seconds, removes the effect ledger only after confirmed retirement/404,
and never retires while a tool effect lock is held. Busy runtime requests reject
retirement. Web shutdown attempts idle cleanup; unknown/busy cleanup is deferred.

The ordinary runtime independently retires nonbusy logical sessions after 600
seconds of inactivity, every 30 seconds and before new session creation. GET/admission
renews activity; active or queued requests are never idle candidates. This covers
Web process crashes as well as lost tabs. A pagehide JSON beacon asks for immediate
release, but delivery/DELETE success is never correctness authority. Lost beacons
cannot permanently occupy a slot. Browser suspension beyond the lease can expire a
conversation: keep history readable, refuse continuation and offer New Chat; never
rebuild native state from old messages or automatically resend.

## Canonical request and effect safety

Before submission, the tab freezes ordinary history, expected base count and a
cryptographic application UUID. Creation also has a known ID before I/O. Unknown
create/generation outcomes are never automatically repeated. Canonical GET must
match the exact next count plus application/native request IDs before a response
enters that conversation's history. Live SSE content remains provisional.
Switch does not change a request's frozen workflow, protocol envelope or identity.

Stop targets only that conversation's known native request ID and waits for canonical
settlement. Queued cancellation consumes its UUID but commits no invented turn.
Missing/unrepresentable state remains blocked. Replay/repack remain zero; losing a
paired pin cannot authorize prefill/reconstruction of retained history.

Before tool POST, the tab records a reservation. The living Web effect ledger fences
(session, count, native request ID); certified canonical GET, not browser arguments
or partial JSON, grants execution. Duplicate calls reuse known results or fail on
uncertainty. Effects publish before budget fitting. Result GET is observation only.
Lost server ledger does not prove nonexecution: a reserved tab observes or explicitly
returns unavailable results; it must not repeat the effect. This remains local
living-client safety, not distributed/crash-safe exactly-once execution.

Tool Auto/Ask/Off, 128-round breaker, complete-request budget fitting, retained PDF
source/derived-observation distinctions and isolated stateless page Vision remain
intact. Browser image acquisition and PDF observation use the connected runtime's
advertised bounded Chat Vision, including ordinary MTP; no separate OFF endpoint
or fallback is required. See [application Vision integration](web-mtp-vision-integration.md).
Native MTP persistence remains unsupported. Existing OFF API artifact support is
not a persistent Browser product; Web UI Save/Restore and archive controls have
been removed. Original-byte image authority is unchanged.

## Security and scope

Stdlib/no-build UI, DOM/text Markdown, CSP, same-origin JSON mutations, private-LAN
peer/Host policy, public acquisition validation and isolated PDF parser limits remain.
TUI Web launch defaults to LAN; direct Web CLI defaults to loopback. No authentication,
TLS, multi-user service, Internet deployment or new framework is introduced.

The old New Chat implementation awaited DELETE, removed old UI identity, then
created a replacement. A successful delete released the old Scheduler pin; New Chat
immediately occupied a new slot. Lack of orphan retirement and the now-removed TUI's
failed-create state explained stranded/conflicting flows. These are separate from
the new additive-conversation specification; obsolete TUI paths were removed, not
expanded into another Chat implementation.

## Evidence

Current product checks: `artifacts/operator-console/`, collected with current-source
`.venv` tooling and real Chrome/PTY/installed model. Historical Web/TUI/policy/PDF
receipts retain their original source scope and are not rewritten. See
[original Web qualification](../artifacts/web-application/qualification.json),
[historical Browser gate](chat-interface.md#qualification),
[historical TUI Chat acceptance](operator-chat-status.md) and
[PDF acceptance](pdf-document-acceptance.md). This delta is not runtime-core
requalification, a performance campaign, model execution or release promotion.
