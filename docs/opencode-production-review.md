# Mac OpenCode ordinary production review

## Scope and execution

Real Mac OpenCode, built-in **Plan** agent, unchanged system prompt/tool set,
actual `ds41f-mlx` source tree and official DeepSeek-V4.1-Flash checkpoint.
No fixtures, client rewrite, soak, Windows smoke or default promotion.

The pre-existing uncommitted ordinary source-text admission fix was retained.
No old serving process was present at initial inspection. The normal-local
server was started from this working tree on `0.0.0.0:8000`; admitted checkpoint
`/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.
`ops inspect --profile mtp-serving-v1` and application startup passed.

The installed OpenCode provider initially sent `DeepSeek-V4.1-Flash`, which is
not a published model alias: HTTP 400 `fixed model alias required`.
**B — client configuration mismatch**, not a general request grammar defect.
For this invocation only, provider model configuration used the advertised
`omlx/deepseek-v4.1-flash`, retaining the full context/output limits and ordinary
Plan environment. Server aliases and client source were not changed.

Session: `ses_edecf390cffejrVn3DyyIIUjXi`.
Initial request: 「リポジトリのレビューをお願いします。まず全体をみて、優先度の高いレビュー対象を選定してください。」

OpenCode naturally inspected git log/branches/status, repository structure,
README, current diff, source and searches. Its Read of `mtp_profile.py` returned
the actual source containing `<|` / `｜`; subsequent source/search/model turns
continued normally. No `raw special-token source unavailable` rejection.
This closes the known defect on the actual OpenCode production path.

OpenCode selected active admission/serving changes as high-priority targets and
reported review observations and passing affected tests. A follow-up asked it
not to re-investigate the now-closed source-text fix, and to complete detailed
review of recent production-serving changes without edits. It reviewed the
`b5e60b6` → `46bfca5` line across `production_mtp.py`, `paired_checkpoint.py`,
`internal_mtp.py`, transport, capacity, settlement and related tests, then
returned a detailed result with code locations and proposed concerns.

## Disposition of review concerns

Review output is evidence of completed work, not defect authority. Checks of
its concrete claims did **not** establish a new production defect:

- Short-prompt negative tap (H1): the subtraction branch requires
  `prompt_capture - frontier >= 8192`; short prompts return `frontier`.
  The claimed negative short-prompt slice cannot follow that branch with the
  admitted 128-row rings. Do not raise the ordinary minimum prompt length.
- Acquire/capture boundary (H2): eligibility deliberately reserves a genuine
  append plus P5 terminal holdout. An N-row stored frontier is reusable for an
  N+2-row prompt; the existing adoption regression demonstrates this. This is
  not replay or recurrent trimming of a cached frontier.
- Poisoned publication (H3): **both** prompt and final publication are inside
  `if not rec.poisoned`. Settlement exceptions propagate into retirement's
  error path, set fatal containment and clear checkpoints. The report's claim
  that poisoned prompt checkpoints still publish is not borne out by code.
- Ordinary Host relaxation (M1): already explicitly documented for trusted
  private-LAN/single-operator serving. Authentication/Internet/multi-user
  capability is unsupported (**B**, not a reason to restore singleton Host
  restrictions).
- Pump race (M2): there is no await between the no-work check, break and
  `_pump_task = None`; the proposed interleaving is not an asyncio race.

No new A/general production API or runtime blocker was exposed. No additional
production code change, session protocol or client-specific workaround was
introduced. Stop after this completed workload, rather than manufacture probes
for speculative findings.

## Validation and ownership

Every recorded serving settlement for this review had `error=None`,
`replay=0`, `repack=0`, canonical publication and no cancellation. These logs
confirm this workload, not a new cancellation qualification. Scheduler remains
sole lifecycle owner; paired target/DSpark state, canonical settlement before
publication and fail-closed mutation handling are unchanged. Existing affected
regressions include adoption/publication/burn and cancellation paths.

Affected regression command:

```sh
.venv/bin/python -m pytest -q \
  tests/test_ordinary_text_parts.py tests/test_production_mtp.py \
  tests/test_runtime_capacity.py tests/test_generic_mtp_tools.py \
  tests/test_production_context.py tests/test_mtp_bind.py
```

Result: **127 passed**. Same set in the admitted normal-local environment:
**126 passed, 1 deselected** (`test_budget_http_is_observation_only_and_count_fenced`,
a standard-OFF test with its separate native recipe pin). Running that OFF test
in normal-local before exclusion reproduced its known identity failure; it
passed in `.venv`. This is not an ordinary MTP failure. The preceding fix's
recorded 203-test validation is retained, not claimed as a new run here.
Singleton contract/limits and raw-token qualification restrictions are unchanged.

The known `preview_certified_eof_tokens` dependency qualification issue is not
an observed blocker here. Normal-local startup checks that capability and passed;
the admitted native recipe completed the real production workload. Other
qualification environments are not repaired or requalified by this exercise.

Operational evidence was captured in `/tmp/ds41f-review-preflight.json`,
`/tmp/ds41f-review-server.log`, `/tmp/ds41f-opencode-review.jsonl` and
`/tmp/ds41f-opencode-review-continued.jsonl` (local transient logs, not release
certificates). Server remains running from the reviewed source.

## Remaining boundary

This closes the source-text blocker and completes one real repository review,
not default promotion. Windows end-to-end smoke remains unverified; the existing
separately scoped release/packaging, full R1 and promotion gates remain open.
No additional genuine ordinary-production blocker was demonstrated by this
workload. Do not convert speculative review concerns or the unrelated EOF
qualification environment into new default-promotion blockers.
