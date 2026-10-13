# Operator console (TUI)

The TUI is an **operator console, not a Chat client**. Web Chat is the sole
conversation frontend. This retained documentation URL replaces the former TUI
Chat page; historical receipts and archived editions are unchanged.

## Responsibilities and keys

| Key | Action |
| --- | --- |
| R | Start / gracefully Stop **LLM Runtime** |
| W | Start / gracefully Stop **Web Chat** |
| O | Open **Web Chat** in the default browser |
| S | Settings / return to Dashboard |
| Q | Quit **TUI**, leaving both services running |
| D / ? | Diagnostics / help |
| ↑↓ / PgUp/PgDn / L / End | Scroll logs / follow latest |

Dashboard labels explicitly say Start Runtime, Stop Runtime, Start Web Chat or
Stop Web Chat based on observed control state. F2/F3, Chat-mode Ctrl shortcuts,
Chat editor/transcripts, conversation ownership, tool/reasoning rendering and
New/Close/Continue workflows are removed from the TUI.

Metrics, logs, readiness/profile/network information and lifecycle diagnostics
remain scalar projections, not execution authority. Settings controls the next
Web Chat launch's LAN/loopback bind and port, and the next Runtime launch's
**Default Reasoning**. It does not modify running services. In Settings, **T**
cycles `none / low / high / xhigh / max`; the initial Runtime preference is **Max**.
The reasoning preference persists in `${XDG_CONFIG_HOME:-~/.config}/ds41f/operator.json`;
`DS41F_DEFAULT_REASONING_EFFORT` overrides it on TUI startup. Web bind/port changes
remain process-local. Per-conversation generation, thinking and tool controls
still belong to Web Chat. Explicit client reasoning controls override the Runtime
default; see [operations](operations.md) for CLI/environment configuration.

## Launch and process boundary

Use the installed environment for model execution. To check current source UI:

```sh
cd /Volumes/SDXC-512/ds41f-mlx
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
.venv/bin/python -m ds41f_mlx.ops
```

The checkout `.venv` is application/development tooling. In this local layout,
Runtime Start selects the sibling `ds41f-mlx_mtp-venv/bin/python`; alternatively
set `DS41F_OPERATOR_RUNTIME_PYTHON` to an already provisioned interpreter. Its
installed code is selected with `-I`. A normally installed TUI uses its own
interpreter. No staging environment or second model executor is created.
Web Chat uses the application interpreter. Application-only environment options
are not forwarded to the admitted model launcher.

Web Chat defaults to **LAN** (`0.0.0.0:8080`, explicit private-LAN permission).
`DS41F_CHAT_BIND=loopback` restricts the next launch; `DS41F_WEB_PORT` changes its
port. Direct `python -m ds41f_mlx.web` still defaults to loopback. The TUI discovers
existing services; opening it does not reconfigure them. Stop then Start to apply
launch settings. Runtime bind is separate: it defaults to `127.0.0.1:8000`.
For OpenCode on another trusted LAN machine, set `DS41F_HOST=0.0.0.0` before
opening the TUI (or use a specific private interface). This is independent of
Web Chat LAN mode. Private LAN has no authentication/TLS; do not expose to Internet.

On macOS, O uses nonblocking LaunchServices `/usr/bin/open`, not AppleScript.
The bounded helper has detached/null terminal descriptors and a 10-second timeout.
Success means an open request, not proof of browser display. The TUI never owns,
waits for or kills the browser process itself. Failures show a manually usable URL.

Q retires no conversation and signals neither service. Service shutdown uses
canonical graceful control and waits for disappearance, never force-kills.
The kernel TUI guard prevents multiple operator consoles. Per-conversation
generation settings are not duplicated in the TUI; Default Reasoning is a
Runtime launch preference, not a conversation editor.

See [operations](operations.md), [live observability](operator-live-observability.md),
[Web application](web-application.md) and [Browser interface](chat-interface.md).
[Historical TUI Chat acceptance](operator-chat-status.md) and its artifact JSON
remain historical evidence of the removed interface, not current instructions.
Current product-boundary receipt: `artifacts/operator-console/`;
collector: `tools/accept_operator_console.py`.
