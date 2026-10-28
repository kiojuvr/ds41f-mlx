# Integrated Phase 4 execution evidence

**Result: C8 BLOCKED, six real reads/three offloads/eight settled native requests.**
See [the report](../phase4-integrated-native.md). The model was shut down after a
repairable read-only inspector failure; it was not reloaded.

`run.py` is the single task driver. It launches one admitted installed worker,
then invokes the official headless CLI sequentially with the frozen task texts
and supported exclusive Session adoption. It never submits inference itself,
dispatches tools, edits history, clears caches, or retries uncertain requests.

Simplifications relative to Step 4B-R2:

- `monitor.py` replaces the blanket other-Python rule. Fresh PID/birth/parent
  chains prove Supervisor descendants, including separate-process-group workers
  and transient fork-before-exec children, are task-owned. Unknown live Python /
  oMLX / ds41f candidates still block; no argv/path whitelist or model-ownership
  inference from process name. Partial observations are retained.
- One full installed/checkpoint/actual zero-weight proof in the worker before
  `ready`, rather than a separate full preflight followed by the same worker
  proof. Normal non-editable provisioning remains unchanged.
- A materialized worker and direct official read-only inspector replace nested
  source-execution/data-URL wrappers. Existing Runtime lifespan/native tracing
  and paired-state mechanisms are unchanged.
- Reuse the already qualified corrected observer and ten-effect guard by
  reference; no duplicate observer/translator, exhaustive readiness campaign,
  historical inventories, or new gate framework.
- Correct Python's handling of string-valued `surfaceOp` at the durable
  replacement check (only dictionary replacement operations qualify).

Authorities are referenced by digest in `frozen-inputs.json`. `execution-inputs.json`
binds the runner code before execution. `monitor-tests.json` tests real processes
and the complete sampler, not only a stand-alone classification predicate.
`effective-configuration.json` comes from the actual installed configuration and
real official read_image declaration; its registration preflight creates no Session.

Audit joins: `wire.jsonl` → `prepared.jsonl` → `trace-NN.json.gz` →
`lifecycle.jsonl` / `persisted-NN.json` / `gate.jsonl`. `audit.py` only reads these
receipts. Native events, image effects, summary attempts and adapter refusals are
separate counters. Resource samples, process provenance and shutdown are in
`resources.json`, `cleanup-lifecycle.json` and `process-cleanup.json`.

The live attempt used the text-only inspector now preserved as
`persisted-executed.mjs`; it failed on the first official offload event.
`persisted.mjs` was corrected **after normal shutdown** to mount the official
offload projection and supply the official projection registry to Session.create.
`postrun_checks.py` reproduces both paths on the same real 65-event history,
checks unchanged durable bytes, verifies the six stored attachment digests and
joins all three changed-projection native executions. `postrun-checks.json`
is read-only post-run evidence, not resumed native work or compaction success.
`execution-inputs.json` authenticates the executed inspector, not its correction.

`official-session.tar.gz` preserves the original official compressed Session,
attachment objects, Files mapping, profile and ready/go receipts. Disposable roots
were removed after validation (`cleanup.json`). To repeat the read-only inspector
check, restore this archive into a disposable root and point `location.json`'s
`temp` there; do not run the native driver or alter original Session events.
`audit.py` reads only repository receipts and runs without those disposable roots.

The frozen three-image budget, 8,192 automatic threshold, 1,024 retention, 2,048
summary cap and all request/output/wall/memory/swap limits remain unchanged.
No numerical kernel, Production admission, official package or Runtime ownership
change. A new invocation is never a replacement Session or uncertain resubmission.
Do not rerun `run.py`: this task permits at most one live load and one native Session.
