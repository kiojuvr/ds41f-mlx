# Shared sampling ownership transfer — candidate qualification

Authority: implementation-ownership feasibility (`c1d1f716:docs/implementation-ownership-feasibility.md` in Git)
and `DS41F-STABLE-e8c719c`. This is one source/maintenance transfer, not stable
promotion or removal of the external oMLX engine.

## Bounded change

`ds41f_mlx/sampling.py` is the complete admitted `omlx/utils/sampling.py`,
verbatim except six attribution comments. Donor SHA256:
`8cd19ae59dd5802b38ae1825098f0ecdfdb13c7ed821bfcab32a24864b6fc131`.
The exported donor and full revision are in `third_party/mtp/sources.json`.
Apache-2.0 attribution and the identical license at
`ds41f_mlx/model_execution/APACHE_LICENSE` remain delivered in the wheel.
The shared backend factory imports the owned module; production MTP inherits
that factory. Capability/trace authority labels identify the new source location;
policy version, defaults, validation, parameters and semantics do not change.
No physical engine, cache, kernel, processor or recipe implementation changes.

## Qualification plan (declared before request measurements)

Use the frozen native MTP and OFF wheels as baseline, and separately built,
non-editably installed native candidates in the existing provisioned candidate
environments. Inherit each profile's exact dependency/native/recipe identities;
never repair development admission or use a validation bypass. Existing stable
receipts cover unchanged portable dependencies and long-context/physical owners;
this is not a new performance campaign or maximum-context qualification.

- Exact donor source identity, helper outputs/errors/shapes/dtypes, all admitted
  filter-presence combinations and boundary values; filtered MTP logits/densities,
  metadata, rowwise/batched draws, RNG state and draw order, worker/stream use.
- Maintained generation-policy, MTP lifecycle/semantic/owner and OFF P5/forward
  regressions. Development admission failures remain failures, not runtime bugs.
- Existing installed native MTP tools/append/branch/SSE/paired-session acceptance
  and installed OFF P5/P6/resource acceptance.
- Matched fresh-server greedy and seeded stochastic short HTTP requests, three
  repetitions per profile/implementation, same policy/prompts/output budgets.
  Compare response semantics, token/draft counts, alignment/replay/repack and
  available memory/latency metrics. One warm-up request per policy precedes the
  measured three. Exact semantic/state/sampling differences fail. Investigate
  a median decode/TTFT regression exceeding both 15% and 0.1 seconds, or peak
  memory growth exceeding both 2% and 256 MiB; these are bounded practical
  screening budgets, not an SLA. Noisy results are inconclusive, not PASS.
- Matched active stream disconnect and subsequent recovery; cancellation frontiers
  are timing-dependent, but settlement/alignment/no-replay and retirement must hold.

## Result: boundary complete; native candidate qualified, not promoted

The [machine-readable qualification](../artifacts/sampling-transfer/qualification.json)
binds both candidate wheels, frozen wheel identities, exact payload deltas, raw
receipts, commands/collectors and comparison results. Reproduce comparisons with
`.venv/bin/python artifacts/sampling-transfer/compare.py`. Candidate manifests bind
`d2a24ad` as the **base** commit and the exact changed source-payload digest; they
do not claim clean `d2a24ad` contains the transfer. Only four ds41f package members
change: the added module, shared factory import, and two diagnostic authority
labels. Every other package member and all dependency/native/semantic/model
identities match the respective frozen native wheel. The repository's generated
execution manifest was restored after building; installed candidates retain their
own manifests. Stable wheels, tag, historical receipts and release metadata are
unchanged.

### Exact behavior

`tests/test_owned_sampling.py` compares against the admitted attributed export,
whose module hash also matches both frozen wheel manifests. Source bytes match
apart from the six comments. Comparisons cover filter-presence combinations,
ties, top-p/min-p cutoffs, invalid inputs, zero/large/chunked top-k, full 129280
vocabulary, 1D/2D/3D and float32/float16/bfloat16; callable parameters, MTP keys,
filtered logits, float32 densities and rowwise/batched draws. Worker/new-stream
calls preserve exact outputs, exception type/message, RNG state, categorical/
uniform order and shapes; greedy makes no draws. Installed routing tests confirm
production MTP inherits the same owned factory. External MLX/MLX-LM primitives
and all physical engine bytes remain unchanged.

Real MTP baseline/candidate tools, append, branch and SSE have identical semantic
records (only request-derived IDs differ), cache frontiers, token counts and
proposal/acceptance counts. Paired logical-session frontiers are **18 and 37** in
both; aligned, replay=0, repack=0. Each measured short request produces **64**
identical tokens under greedy or seeded stochastic combined top-p/top-k/min-p.
MTP greedy draft counts are **51 proposed / 31 accepted**, stochastic **50 / 28**
(the raw receipts are authoritative). Active disconnect settles aligned at C=20,
with 3 generated tokens and 2 proposed / 1 accepted draft; explicit recovery is
identical, with no automatic resend.

Real OFF baseline/candidate HTTP greedy/stochastic choices and usage match
exactly. Tool-call arguments/content and tool-result SSE text match. Installed
P5/P6 acceptance retains the same 40-cache list, P5 once, all-layer frontiers,
17-token suffix, native-call counts, replay=0/repack=0 and allocator/wired lease
restoration; all 16 token reports match except latency. OFF's existing control
projection exposes no per-request/cancellation metrics: disconnect plus explicit
serialized successful recovery and the direct/maintained lifetime evidence are
used, not invented MTP-style telemetry.

### Practical screening and retained evidence

The predeclared three measured repetitions per policy/profile pass the warm
latency budget. Exact medians are in the qualification JSON. OFF sampled process
RSS differs by about 2.2 MiB; this is **not** peak MLX memory. MTP's final reported
model-load allocations are identical (281.1 GiB active, 5.2 GiB cached at the last
reported group); no new peak-memory claim is made. Verbatim operator graphs add
no tensor operation, draw, synchronization, state/cache owner or weight allocation.
Unchanged long-context, native/portable, resource and failure boundaries retain
the stable baseline's evidence at its recorded scope, not a fresh campaign.

The single uncached MTP tools TTFT was 3.248s fresh baseline versus 3.844s
candidate. Investigation places the difference in **unchanged prefill** (1.729s
versus 2.252s suffix append), before the transferred factory; decode is faster.
The frozen stable identical fixture recorded 3.727s TTFT. These cold observations
are noisy/inconclusive, **not a cold-start performance PASS**; candidate remains
within the frozen practical observation and matched warm sampling/TTFT is steady.
No baseline value or budget was rewritten.

An early collector expected a nonexistent `cancelled` control field (the existing
field is `finish_reason=abort`). Rerunning it had warmed the baseline cancellation
prefix. The first candidate run therefore had a different initial cache frontier,
not different sampling; it is retained as `candidate-mtp-http-initial.json`.
The final comparison uses the same warmed prefix/frontiers in both implementations;
all measured outputs/counts were already identical on the initial run. OFF's first
candidate warm-up includes lazy model load; warm-ups are excluded as declared.

### Tests, admission and retirement

- Source differential/authority tests: **215 passed**. The focused source suite
  repeats **79 passed, 14 subtests passed, 4 admission failures** recorded by
  feasibility; no bypass or development-environment repair. Additional source
  P5/cache-release/recovery seams: **26 passed, 14 subtests passed**.
- Independent installed candidate tests pass for owned sampling, generation
  policy/authority, production session owner/sessions, MTP binding/lifecycle,
  recovery, profile rules, OFF target/forward and context-copy contracts. All
  **four installed ordinary HTTP route** cases pass. This is not a full-suite PASS.
- Attempts to run source-only tests in isolated installed Python retain their
  failures: source `tools`/`tests` imports unavailable, checkout-negative test
  requiring the checkout refusal rather than installed execution, and a fake
  backend with an unset explicit checkpoint. The affected source seams pass;
  these are test-context limitations, not repaired runtime behavior. Raw logs
  are retained, including all development admission failures.
- Native installed startup/admission and existing `accept_native_mtp.py` /
  `accept_native_off.py` pass for frozen baseline and candidates. Each uses its
  separately admitted recipe/native bytes. No validation rules changed.
- All model processes started here exited through graceful shutdown; candidate
  environments contain the qualified wheels, and normal operator installations
  were not changed. No automatic stable promotion occurred.

**Remaining concrete blocker: none for this bounded ownership transfer.** This
qualification does not transfer Scheduler, BatchGenerator, PagedCache, MTP physical
execution, target/cache layout, kernels or recipe semantics, and does not authorize
a subsequent ownership boundary.
