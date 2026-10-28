# Phase 4 completion evidence

**C8 BLOCKED:** seven real reads, three offloads, one successful automatic compaction,
durable replacement and post-compaction native continuation. Model end_turn after
G leaves H/I/new A unperformed. See [report](../phase4-completion-native.md).

One new load and one official native Session, starting at de1fc2416f7a61d5485a21ac9ff2629ddd237fe8.
The prior evidence directory is unchanged. Do not rerun the native driver.

`run.py` reuses the proven single-worker/official-CLI path. No direct inference
client, effect dispatcher, history edits or independent KV ownership. Policies,
tasks, image identities and ceilings remain the frozen Phase 4 authorities,
referenced by digest in `frozen-inputs.json`. `execution-inputs.json` binds the
actual task code. Production numerical/admission code is unchanged.

Only necessary task changes:

- Use the corrected official persistence inspector, including the owning image
  offload plugin and official Session projection registry. Before loading, the
  **exact inspector subprocess** successfully reconstructs the retained real
  65-event offload Session and 71-event compaction Session without changing their
  durable bytes (`diagnostic-preflight.json`). Temporary test copies are removed.
- A read-only inspection failure pauses further CLI activity, while the same
  Supervisor continues resource/time checks. Explicit repair/resume can repeat
  inspection only, after checking native settlement, Runtime health and no pending
  effect. No automatic native resubmission or Session replacement.
- Native trace snapshots use atomic replacement, avoiding partial diagnostic
  reads. The existing lifespan remains the sole shutdown owner.
- The passive collector records official `estimateMessage` pricing on replacement
  events; the actual pre-compaction meter surface supplies selected route tokens.
  No second tokenizer, summary framing or compaction implementation.

`wire.jsonl` → `prepared.jsonl` → `trace-NN.json.gz` joins actual canonical inputs,
image spans, paired roots, Vision, complete paired settlement and SSE. Official
lifecycle/persistence and unique tool results are in `lifecycle.jsonl`,
`persisted-NN.json` and `gate.jsonl`. `audit.py` and `acceptance.py` are read-only
post-run joins, not execution gates or simulated successes. `resources.json`
retains prospective PID/birth/ancestry ownership and physical-memory/swap samples.
The corrected terminal observer, task guard and resource predicates are reused by
reference, not copied/requalified in another framework.

`cleanup-lifecycle.json` separates request retirement/process exit from model
weakref destruction. C2 and C7 retain their independent qualification limits.
The small `official-session.tar.gz` preserves original official compressed storage,
attachments, Files mapping, profile and load handshake; only disposable task roots
are removed. No push, deployment, kernel change or Phase 4C.

Post-run reproduction (no model, no live roots needed): run `audit.py` then
`acceptance.py`. The latter reads authenticated attachment bytes directly from the
archive after cleanup and intentionally exits nonzero because the positive C8
workload is incomplete. Its native-integrity/no-uncertainty predicates still pass.
To repeat official read-only Session inspection, restore the archive to a disposable
root and supply `PHASE4_INSPECT_ROOT` and `PHASE4_INSPECT_OUTPUT` to `persisted.mjs`.
Do not relaunch the native driver.
