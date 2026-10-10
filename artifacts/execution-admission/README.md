# Installed admission migration qualification

Final published MTP wheel: `release/execution-package.json` (SHA-bound artifact in
`release/execution-packages/`). The same wheel was installed non-editably and
started by canonical `ops start --profile mtp-serving-v1` in isolated Python from
`/tmp`, on the current pcre2 10.49 host. `acceptance.json` is its bounded real-model
ordinary acceptance; shutdown completed cleanly. This is not fresh OFF, full
long-context, new-hardware or general release qualification.

- `boundary.json`: unchanged admitted recipe extension, historical pcre2 10.48
  versus actual 10.49 byte drift; no reseal/replacement approval. Installed Python,
  recipe and tokenizer changes are rejected; semantic-native digest corruption
  is explicitly simulated (no mutation of shared Homebrew). Removing source/build
  lookup locations and adding unrelated site content preserve the execution digest.
- `native-contracts.json`: 77 preview/nonmutation/determinism rows, exact 64-record
  official recipe parity and exact 28-row R1 recovery/representation contracts;
  197 protocol/admission tests and 56 state/resource tests in separate processes.
  The legacy state harness imports its OFF donor during collection, so combining
  it with installed-origin MTP tests causes pre-existing import contamination.
- `acceptance.json`: generic tools, real continuation with 374 cached tokens,
  branching, ordinary SSE and malformed-request refusal; READY health, empty queue.
- `server.log` / `final-installed.json`: all four final generations publish their
  canonical paired frontier; all have zero prompt replay and zero cache repack,
  with no errors/cancellation. Exact state/frontier/pairing/recipe guards remain
  unchanged. Final installed import origins and wheel/execution digests are recorded.

Model byte verification freshly scanned all official shards once; later admission
used only the unchanged file-identity-bound receipt. No checkpoint bytes, recipe
wheel, native numerical library or legacy `normal-local.json` was refreshed.

Build provenance records a source **base commit** plus the complete package-input
digest, since the migration wheel was built before the final source commit. This
is provenance, not installed startup equality. Historical source/native closure
inventories remain evidence. No pcre2 SHA is present in the new admission manifest.
