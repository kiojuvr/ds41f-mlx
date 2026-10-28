# Phase 4 E2E execution and post-run audit

One authorized native load / one official Session. Do not rerun `run.py`.
Prior evidence is unchanged. See [report](../phase4-e2e-native.md).

Reuse of the completion attempt's worker, provenance Supervisor, official CLI,
corrected read-only persistence inspector, terminal observer and evidence formats.
`execution-inputs.json` binds task code; `frozen-inputs.json` binds unchanged
policy, sources, task texts, ceilings and inherited safety predicates.

The driver treats normal end_turn as a completed turn, not a completed workload.
It checks the completed effect prefix and requests only unfinished reads in the
same Session. Intermediate reuse/offload/Vision assertions move to the post-run
acceptance audit. The task guard no longer requires acceptance offload/compaction
milestones before permitting ordinary work. Authentication, effect order and
uniqueness, uncertain-effect/protocol fences, failed/unclosed/second compaction
fences, paired integrity, admission, frozen headroom and resource/output/time
budgets remain enforced. No native request is retried by this driver.

`driver-checks.log` covers the actual H/I/A continuation function and changed
pure guard seams; `diagnostic-preflight.json` reuses actual durable histories.
Provisioning completed wheel installation before its shell timeout; the final
model-free installed payload inspection was then completed explicitly. Initial
Supervisor import failed on missing psutil before starting any worker; it was
launched using the newly provisioned environment (no load or Session consumed).
Both preparation incidents are retained, not native failures/retries.

Post-run: `audit.py`, then `acceptance.py`, join actual wire, Recipe, native paired
state/Vision/settlement, official events and persistence. `official-session.tar.gz`
preserves original compressed storage and authenticated attachment bytes.
`cleanup-lifecycle.json` distinguishes request retirement/process exit from
same-process Backend/Model destruction. C2/C7 and broader C6/C9 stay independent.
No Production changes, push, deployment or Phase 4C.
