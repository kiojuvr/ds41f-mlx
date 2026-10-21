# Web application — retired

The bundled Browser Chat application and `ds41f_mlx.web` Chat Server are retired.
No Browser frontend/assets, conversation/workflow manager, application leases,
effect ledger, Tool Loop, search/acquisition or PDF executor remain supported.
These facilities were removed, not moved into Runtime or replaced by a shim.

Runtime remains MTP-only for package-fixed Flash and Flash-1017, with native Chat
Completions, Messages, bounded inline Vision and production logical sessions.
The TUI is solely an [operator console](operator-chat.md). External clients own
agent orchestration and tool execution. No replacement Harness is bundled;
successful inference is not comprehensive Agent/Harness compatibility.

## Historical evidence

The following retained source-scoped excerpt describes the former application,
not current product checks or instructions. Original source is in Git history;
archival receipts are unchanged and removed Web tests are not Runtime qualification.

Current product checks: `artifacts/operator-console/`, collected with current-source
`.venv` tooling and real Chrome/PTY/installed model. Historical Web/TUI/policy/PDF
receipts retain their original source scope and are not rewritten. See
[original Web qualification](../artifacts/web-application/qualification.json),
[historical Browser gate](chat-interface.md#qualification),
[historical TUI Chat acceptance](operator-chat-status.md) and
[PDF acceptance](pdf-document-acceptance.md). This delta is not runtime-core
requalification, a performance campaign, model execution or release promotion.
