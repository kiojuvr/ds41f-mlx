# M24 — Operational Agent Soak and Long-Session Robustness Qualification

## Status

M24 adds a bounded operational agent-soak harness and a refreshed source-tree diagnostic run over the already-qualified M23/M20 runtime architecture.  It does not change the production model path: MTP, DSpark, DFlash and speculative decode remain OFF.

Canonical artifact:

- `artifacts/m24/operational-soak.json`

Supporting diagnostic tooling:

- `tools/run_m24_operational_soak.py`

## Harness

Pi Agent was not modified.  For this qualification run, a controlled OpenAI-compatible client/tool-loop harness was used instead so the exact request/response/tool-result transcript could be preserved without introducing Pi-specific ambiguity.  The runtime server executed no tools; tools were executed only in the client harness against a disposable workspace.

The harness supports starting a release/source server process, driving `/v1/sessions`, polling bounded diagnostics, persisting/restoring at an idle boundary, simulating client interruption, executing client-side tools, and preserving selected exact protocol envelopes.

## Workload refreshed in `artifacts/m24/operational-soak.json`

- Server path: source-tree server with M24 diagnostics enabled, using the same configured external checkpoint/oMLX/deepseek-recipe assets as M23.
- Model/runtime identity: `deepseek-v4.1-flash`, DENSE_P0_P7, P7 full-resident backbone SSD Engram, P5 zero-replay handoff, oMLX 0.7.0 GenerationBatch, MTP/DSpark/speculative decode OFF.
- Workload: disposable Python repository with `calc.py` / `test_calc.py` and client-side tools (`list_files`, `read_file`, `write_file`, `run_tests`, `intentional_missing_command`).
- Turns: 10 stateful Chat Completions turns.
- Tool cycles: 10.
- Persistence/restore: one idle persist at frontier 1024, clean server restart, restore to a fresh session id, then five additional turns.
- Cancellation/interruption: one client timeout at turn 3.  The server committed the turn after the client timed out; the harness waited for the idle boundary, recovered the canonical committed response from session metadata, and continued without retrying it as new work.
- Intentional failures and recovery: failing tests and a missing-command tool failure were recorded; later `write_file` and `run_tests` cycles recovered the workspace tests.

## Observed runtime invariants

For the canonical run:

- Final frontier: 1702 tokens.
- Replay/repack: total prompt replay count 0; total full-cache repack count 0.
- Cache/frontier: all recorded idle boundaries reported all 40 cache offsets equal to the token frontier.
- Continuation: every accepted turn was an exact-prefix extension; no hidden fresh-prefill fallback was observed.
- Persistence/restore: restored session resumed from the persisted frontier and continued through multiple additional tool cycles.
- Cleanup: restored session closed successfully; final diagnostics reported no locked backend mutex.

## Performance/resource observations

The run was deliberately bounded and is not a leak proof.  Within the workload:

- First-token latency remained approximately 50–53 ms across early/mid/late turns.
- Decode throughput remained approximately 19–20 tok/s across early/mid/late turns.
- RSS samples stayed in the approximate 12.4–13.2 GB range after model load; no monotonic unexplained growth was observed in this bounded run.
- System memory pressure sample reported 42% free memory.  Existing system swap usage was 158 MB of 1024 MB; the run did not establish new swap growth.

## Repetition/loop classification

No unclassified runtime repetition incident was observed in the canonical run.  The run did include normal model/tool-loop behavior where the model inspected files over several turns before editing.  Recorded diagnostics classify this as client/model task progression, not runtime replay: transcript hashes were recorded, frontier advanced coherently, replay/repack remained zero, and cache offsets stayed coherent.

M24 does not claim the model is loop-free.  The harness preserves exact samples sufficient to distinguish:

- model-behavior repetition with coherent runtime state;
- client/transcript repetition;
- runtime/session-state defects;
- performance-induced retry/cancellation churn.

## Defects found/fixed

No runtime defect was found in the canonical run.

One harness issue was found during an earlier smoke attempt: after a client timeout, the harness retried immediately while the original server request was still active.  That produced an expected `active request` conflict and ambiguous cancellation evidence.  The harness was fixed to wait for the server to return to an idle boundary and, if the timed-out request committed, recover the exact committed response instead of replaying it as new work.

## Qualification outcome

M24 qualifies the tested MTP-OFF runtime for ordinary bounded sustained agent use at the scale exercised by this run: exact continuation, zero hidden replay/repack, coherent cache ownership, usable client-side tool/result loop, recoverable client interruption, persistence/restore/resume, successful cleanup, stable bounded resource behavior, and diagnosable transcript/runtime state.

This is an operational-soak claim for the tested workload, not a new long-context maximum and not a claim that all possible model thought loops are absent.
