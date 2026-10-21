# Operator console

The TUI is solely a **Runtime operator console**, not a Chat/Agent application.
The bundled Browser Chat and Chat Server are retired; there is no replacement
Harness. External clients use Runtime's native Chat Completions and Messages.

| Key | Action |
| --- | --- |
| R | Start / gracefully Stop Runtime |
| S | Runtime launch settings |
| D | Diagnostics |
| ? | Help |
| Up/Down, PgUp/PgDn | Scroll logs or details |
| L / End | Follow latest logs |
| Q | Quit console; leave Runtime running |

Readiness, status, metrics and logs are observed through the existing loopback
Runtime control service. Conflict detection prevents launching over an occupied
service port. Shutdown waits for protected native work and retirement; it is not
a force-kill or permission to resend uncertain requests.

Runtime launches use the installed interpreter selected by
`DS41F_OPERATOR_RUNTIME_PYTHON`, the existing sibling installed MTP environment
when discovered from the development `.venv`, or the current interpreter.
The canonical isolated launcher is `python -I -m ds41f_mlx.ops start --profile
mtp-serving-v1`. Headless `ds41f start` and `ds41f_mlx.serve` remain available.

Settings preserves Default Reasoning (initially Max, unless overridden).
S then T cycles none/low/high/xhigh/max; the saved preference applies at next
Runtime start. Explicit request controls override the default. There are no Web
bind/port settings, Web child processes, Browser launch actions or Chat controls.

See [operations](operations.md#operator-tui-and-applications) and
[live observability](operator-live-observability.md) for operational details.
Historical Web/TUI Chat receipts qualify only their original inputs, not this
operator-only surface or any external Harness. Successful inference does not
establish comprehensive Agent/Harness compatibility.
