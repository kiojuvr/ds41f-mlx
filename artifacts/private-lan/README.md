# Opt-in private LAN qualification

- `browser.json`: current-source PASS, 15 checks on a real insecure HTTP private-IP origin, isolated Chrome context and existing standard-OFF runtime. Includes real generation, Stop, reload, fresh restore and two-tab non-expiring transaction exclusion. Canonical replay/repack counts zero.
- `http-initial-pass.json`: preserved earlier PASS before adding paired optional TLS launcher flags; not final-source acceptance.
- `tests.txt`: 34 passed + 8 subtests, including network Host/peer/origin/forwarding boundaries and affected native delivery/recovery regressions.
- `renderer-platform-tests.txt`: 15 passed, including safe rendering, cryptographic HTTP UUIDs and unchanged secure-context Web Locks contract.
- `runtime-bind.json`: standard-OFF wildcard bind accepted by resolved --print-config; no model/server restart.

Source hashes cover web.py and six static assets. Runtime/serving/serve.py sources are unchanged from 9312cdd. The private test context and its native sessions were retired; the user's localhost browser/conversation was not modified. HTTP tests used this machine's real LAN interface, not a second physical PC. Optional TLS flag pairing/forwarding is unit-tested; certificate installation/physical remote PC access is not qualified. No R1, release or runtime promotion implied. See docs/private-lan.md.
