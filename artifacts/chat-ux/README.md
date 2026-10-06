# Current Chat UX qualification

**PASS**: `qualification.json` gates final five static assets against
`browser.json` (68 actual Chrome/runtime checks). The 95 tracked production Python
sources match `133adc4`. Scope: everyday localhost single-current-conversation,
standard-OFF Chat; not R1, release or ds41f-runtime promotion.

- `browser.png`: final actual browser surface after restoring a different Thinking prefix.
- `affected-tests.txt`: 90 passed + 8 subtests.
- `renderer-tests.txt`: 12 passed, including XSS/URL safety, incremental stable DOM,
  giant whitespace/table and quote-depth guards, public-only live reasoning.
- Native snapshots: zero prompt replay/full-cache repack; original two JPEG hashes
  remain exact; image save frontier 1167; no historical image encodes on text/fresh restore.
- Tool result received before model observation: saved and freshly restored at
  frontier 8625; inline continuation consumed it without another effect (11 tool
  result messages before and after).

Earlier attempts remain unchanged evidence, **not current-source acceptance**:

- `collector-await-failure.json`: the collector incorrectly awaited background Send.
- `initial-partial.json`: earlier partial native observations.
- `reconnect-observation-race.json`: premature collector observation during async recovery.
- `source-change-failure.*`: asset pin correctly refused acceptance after editing.
- `empty-public-reasoning-failure.json`: invalid assumption that enabled Thinking
  must emit nonempty public reasoning; actual protocol/locks were correct.
- `earlier-ui-pass.*`: an older-source PASS, before final UX corrections.
- `ime-event-wiring-failure.json`: real bug in composition event registration.
  Fixed with addEventListener and requalified from before the first generation.
- `prior-m35-pin-failure.txt`: pre-existing M35 historical evidence SHA pin mismatch;
  `internal_mtp.py` is unchanged from `133adc4`. No historical receipt was rewritten.

See `docs/chat-interface.md` for mappings, limitations and reproduction. The
collector uses a separate private browser context and only its own native sessions;
it never restarts the runtime or closes the user's conversation. OS paste/drop/IME
input events are generated inside actual Chrome; every OS implementation is not
claimed. Synthetic DOM/disclosure stress data is display-only, never model history.
