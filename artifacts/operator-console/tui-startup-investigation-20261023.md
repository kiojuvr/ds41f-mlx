# TUI startup OSError investigation — 2026-10-23

**Result: NOT REPRODUCED; original cause unresolved.** This is not a fix or Harness/image acceptance.

## Scope and identities

User stopped the working headless Runtime and confirmed memory release before authorizing investigation. No model process or listener on 8000/18441/18442 was found before testing. Repository HEAD remains `64364270c14271a6cf710712e88e8e03b70a5dfe`, branch `Visual-Feedback-Loop`; existing Phase 0 report preserved.

Used `/Volumes/SDXC-512/ds41f-1017/bin/python`, Python 3.13.15, isolated installed imports (`-I`). Explicit checkpoint: `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`; saved TUI reasoning effort: `max`. No host override (test used the default loopback binding), no model-generation requests, and no Harness execution.

Installed file SHA-256:

| File under `lib/python3.13/site-packages/ds41f_mlx/` | SHA-256 |
| --- | --- |
| `execution_manifest.json` | `39a39c45df3a5bb61bca0e5147298b7849efc85a3d34c037c1641dd7672511b5` |
| `operator_tui.py` | `de642a48a9544287274c1e57ce6994a4ce58cda8a04aee177ba9c7d4c85e8efd` |
| `operator_control.py` | `d3bfd0ff0470d548346a508d674c7bda913cad6fed54214e127f38b3a1329b13` |
| `runtime_logging.py` | `6702a17f6e0cf134169f7e26d7f208c8e5805b2642bf0bc1820176e2dd620992` |

## Observation and diagnostic limitation

The supplied original TUI record reported `asyncio` / `OSError` immediately after READY and Application startup complete. Its sanitized message does not identify an errno, operation, or traceback. It does not by itself establish a failed server, incoming model request, or memory-exhaustion cause.

Installed `operator_tui.launch()` calls the canonical installed `ds41f_mlx.ops start`, which execs `ds41f_mlx.serve`. Differences from a terminal launch include detached session, stdin/stdout/stderr sent to DEVNULL, environment filtering, and the saved reasoning setting. TUI also polls local control status/logs. `runtime_logging.RuntimeHandler` retains sanitized warnings in its control history but writes original exception details only to stderr. Since the normal TUI launch discards stderr, those details cannot be recovered from that old TUI record.

For these tests only, an in-memory wrapper around `subprocess.Popen` redirected child stdout/stderr into a private temporary file (mode 0600). No installed/source Python file, persisted environment, TUI setting or model asset was edited. This deliberate diagnostic redirection differs from the original DEVNULL destinations, and neither test reproduces the user's exact terminal environment or possible LAN bind.

## Bounded experiments

1. **Installed TUI start/control functions without curses:** instantiated `Operator`, used `lifecycle('start')` and `poll()` every 0.4 s, observed READY for 15 seconds, then used `lifecycle('stop')`. Runtime PID 13635 exited 0. Elapsed 146.2 s. No ERROR or OSError in the captured stderr.
   - READY: 09:28:55.354 JST; shutdown complete: 09:29:10.651 JST.
   - Temporary log: `/var/folders/jk/0kx3l0t93zj6gwlgth99c6lc0000gn/T/ds41f-tui-start-diagnostic-j8g5k7uh.log`.
   - Log SHA-256: `f3a3ecc23c6a1b343cdb3c5aefca7de5c09d48fe940d8aa8d756f8a21149ccf8`.
2. **Actual installed curses TUI in a pseudoterminal:** TERM `xterm-256color`, 150 columns × 45 rows; invoked `operator_tui.main()`, including its singleton guard and ordinary event loop. Sent `r` to start, waited READY + 15 seconds, sent `r` to stop, then `q` after runtime retirement. TUI PID 19692 exited 0. Elapsed 144.9 s. No ERROR or OSError in captured runtime output.
   - READY: 09:32:04.320 JST; shutdown complete: 09:32:19.623 JST.
   - Temporary log: `/var/folders/jk/0kx3l0t93zj6gwlgth99c6lc0000gn/T/ds41f-real-tui-diagnostic-1xm62or7.log`.
   - Log SHA-256: `b355c658d1708eabf573a153bb359909c09270bb1562b02d21c303d7b40529de`.

Only control/status/log requests and graceful shutdown were used; no public inference POST. Raw TUI terminal bytes were drained, not persisted. Temporary runtime logs contain model-loading/lifecycle diagnostics, not conversation/image/credential payloads. Minimal sanitized results and hashes are retained here rather than duplicating verbose load logs.

After both experiments, no Runtime/TUI process or listener on 8000/18441/18442 was found. Servers were gracefully stopped; none was left running. No runtime package changes, rebuild/reinstall, checkpoint changes, commit or push.

## Follow-up: user's LAN bind configuration

User supplied the original invocation: activate `/Volumes/SDXC-512/ds41f-1017`, export the same checkpoint, export `DS41F_HOST=0.0.0.0`, run `ds41f`.

A third bounded actual-TUI pseudoterminal run set that host, VIRTUAL_ENV and PATH accordingly, retaining the usual DEVNULL stdout/stdin and redirecting **only stderr** for diagnostics. R start → READY at 09:46:03.894 JST → R stop → retirement at 09:46:19.398 JST → Q exit 0, elapsed 146.5 s. Runtime PID 25784, TUI PID 25783. Again zero ERROR/OSError occurrences. No public inference POST was sent. Final process/listener checks found no remaining Runtime/TUI on 8000/18441/18442.

Temporary log: `/var/folders/jk/0kx3l0t93zj6gwlgth99c6lc0000gn/T/ds41f-tui-anyhost-diagnostic-2tiu93jf.log`; SHA-256 `f72256dc785dcaea44b497e47793b2f53a5fefe5274db22f0dc73d810cb1e01a`.

Thus `0.0.0.0` alone did not reproduce the issue. The actual user's terminal, other inherited settings, timing and inbound client connections remain possible differences, not established causes. Diagnostic stderr redirection is also a deliberate difference from the original launch.

## Next discriminating evidence

Prepared `/tmp/ds41f-tui-capture-stderr.py`, a temporary diagnostic entry point for the user to run in the original activated terminal with original exports. It invokes the installed TUI and redirects only the Runtime launch's stderr to a fresh private temporary file, leaving installed files, settings, stdin and stdout unchanged. R starts/stops normally; Q exits TUI and prints the log location. Exiting TUI alone does not stop Runtime: stop with R first. After providing the wrapper, the user reported that the issue did not recur and Windows DeepSeek Harness conversation worked normally. No exception traceback or capture log was supplied. This is user-reported successful operation, not a separately observed wire capture.

Do not diagnose memory, change asyncio/uvicorn, or rebuild the installation based only on the sanitized exception class. Current evidence demonstrates successful installed TUI startup under the tested conditions, followed by user-reported successful Windows Harness conversation without recurrence. The original OSError's cause remains unknown; no fix was applied and no dependency update is justified. Stop reproduction attempts unless the error returns. This does not qualify the Phase 0 image transport path.
