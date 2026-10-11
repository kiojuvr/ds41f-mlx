# Runtime-authoritative capability budget

This page owns **actual-request output reservation and post-effect result fitting**.
It originated in standard-OFF; ordinary MTP uses the same capacity machinery with
its explicit output ceiling. [Capabilities](capabilities.md) owns current profile
limits/defaults; [frontend policy](generation-policy-frontends.md) owns controls;
[application state](web-application.md) owns request/effect/recovery journals.
Historical OFF Browser PASS does not qualify later tool/PDF/MTP changes or maxima.

## Output and admission

`max_tokens: "auto"` is a runtime extension. A positive placeholder reaches recipe
conversion only; real rendering/tokenization and original-byte image expansion
resolve `prepared.resolved_max_tokens`, which execution uses. Explicit positive
reservations remain supported. API omission resolves its default, not Auto;
everyday Chat negotiates Auto from runtime defaults.

`serving/capacity.py` owns the profile-aware decision:

```text
total = min(checkpoint range, qualified text or OFF-image envelope)
prompt = actual expanded recipe token count
remaining = total - prompt
OFF Auto = remaining
ordinary MTP Auto = min(remaining, ordinary output ceiling)
Custom = requested positive allowance, within remaining and profile ceiling
```

History, new input, full tool results, frozen reasoning/declarations and recipe
structural tokens all count. No char/token estimate, frontier-only subtraction,
above-checkpoint extrapolation or output-sized allocation is implied. Total
context is prompt **plus** reserved output, not a maximum prompt plus unlimited
generation. Invalid/exhausted capacity rejects before native mutation. MTP stays
text-only; an image-bearing request cannot borrow the OFF envelope.

`POST /v1/sessions/{id}/budget` prepares the same complete request and checks the
selected retained-session admission: alias, busy/recovery state, exact canonical
prefix, reasoning/envelope and OFF original image identities where applicable.
It reports count and `binding:false`; no native generation/cache append/response
reservation/effect occurs. CPU recipe/image validation is not GPU image encoding.
Actual generation prepares/rechecks before reservation. Chat Server additionally
checks the observed count. Preview retries never become generation/effect retries.

Rejected pre-submission preview is “No generation submitted”, not invented unknown
admission. Execution finish/cancel classification comes from the profile's
canonical outcome, not merely the Auto selector. OFF records Auto length as
context capacity; ordinary MTP has its own outcome projection. Continue must
recheck the full exact request; it cannot drop/replay committed history to fit.
Exhaustion needs an explicit new conversation or eligible OFF saved state.

## Tools and external resource limits

Model capability, operator defaults and resource ceilings are separate:

- Tools Auto's **128-round runaway breaker** preserves pending calls and needs
  explicit Continue, not routine round quotas or automatic effect retry.
- Thinking uses advertised effort enums; obsolete 50/75/100 percentage aliases
  are not current controls. Sampling and frozen retained reasoning have separate
  ownership; no prefix/saved preference is rewritten to change it.
- Search default 10/max 20 bounds provider latency/noise, not model context.
- Public fetch: 8 MiB acquired bytes, 45s absolute transfer deadline, up to eight
  redirects. Validate public DNS/actual peer and each redirect. Deadline closes
  stalled/trickled socket work; OS DNS resolution is not independently cancellable.
  Refuse compressed bodies rather than unbounded decompression. Usable byte-bound
  partial results carry explicit network truncation.
- HTML readable extraction excludes scripts/styles/noscript/templates; original
  image/PDF bytes are separate retained sources, not rewritten by text extraction.
- `fetch_url` default discovery is **4096 characters**, not a hard context or
  memory ceiling. Explicit length/retained artifact offset can select more without
  automatic redownload. The old fixed 1 MiB fetch-excerpt cap was removed by
  [tool capability completion](tool-capability-completion.md); acquired bytes and
  aggregate serialized batch bounds now own resource limits, then runtime fitting
  chooses model-consumed text. Call serialization and result serialization are
  different ceilings; PDF per-page/parser limits are separate.

Exact acquisition/batch/artifact/PDF policies belong to tool capability completion
and [PDF workflow](pdf-document-workflow.md). They do not expand native MTP image,
context, concurrency or effect guarantees.

After effects, Chat Server publishes actual completed results **before** fitting.
It retains source results in its bounded living ledger until replacement/close.
`web_budget.fit_tool_results` forms the complete next request with **all** batch
results and asks runtime admission. Only new, not-yet-model-observed fetch/page
text may be excerpted; old messages, committed token prefixes and image bytes
never change.

Bounded candidate search sends every candidate through runtime budget, not a local
token estimator or monotonic-tokenization assumption. A selected candidate has
actually passed. Truncation, returned counts, next offsets and `has_more` remain
explicit in model-facing JSON/display. Retained source continuation is deliberate,
not automatic reacquisition.

If structural/non-fetch content cannot fit, completed effects stay observable with
`budget_error`; pause and explicitly recheck, never execute again to get smaller
results. Lost response recovery is GET-only ledger observation. Fitting is
application bookkeeping, not native model/history authority or effect consent.

## Evidence and limits

[Original budget evidence index](../artifacts/dynamic-budget/README.md) and the
[pre-compilation edition](archive/editorial-baseline/dynamic-capability-budget.md.txt)
retain exact OFF Chrome/runtime/source counts, older failures, deadline follow-up
and prior Thinking/excerpt wording. Those source-specific results do not silently
close later tool/PDF/frontend/ordinary-MTP boundaries.

`qualify_dynamic_budget.py` is an existing reproduction collector, not permission
to start an isolated model pair during compilation. CPU tests cover recipe counts,
CPU image expansion, checkpoint/profile intersection, nonbinding preview and
actual-request fitting; model/Browser evidence stays at its recorded scope.
No maximum-context/GPU/full Vision/R1/release qualification is added.
