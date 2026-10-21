# Application capability budgeting — retired

The Browser application, acquisition tools and their budgeting/workflow manager
are retired. No application budget manager or tool executor was moved into Runtime.
Runtime admission/capacity and generic tool-call protocols are unchanged; current
limits belong to [capabilities](capabilities.md), not the historical policies below.

## Historical policy excerpt

The following unchanged excerpt records former application policy only. Its
controls, tools and executors no longer exist in the bundled product.

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

[Original budget evidence index](../artifacts/dynamic-budget/README.md) and the
[pre-compilation edition](archive/editorial-baseline/dynamic-capability-budget.md.txt)
retain exact OFF Chrome/runtime/source counts, older failures, deadline follow-up
and prior Thinking/excerpt wording. Those source-specific results do not silently
close later tool/PDF/frontend/ordinary-MTP boundaries.
