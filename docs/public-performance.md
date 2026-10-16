# Ordinary MTP public benchmark — 2026-10-16

Small canonical practical-performance receipt for the current ordinary default
`mtp-serving-v1` implementation. **Original baseline: PASS, six real-model
requests**, preserved unchanged. The follow-up below adds synchronized prefill
measurements and matched historical-condition controls: **EXPECTED / EXPLAINED**.
No runtime changes or benchmark-specific production tuning. This is a performance
observation, not a new release promotion, numerical qualification, endurance test
or SLA.

## Environment and exact implementation

- Mac Studio, **Apple M3 Ultra, 512 GB unified memory** (549,755,813,888 bytes),
  macOS **26.5.2**, Python **3.13.15**. One model-active request at a time.
- **Official DeepSeek-V4.1-Flash**, original 48-shard checkpoint at
  `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`, with the normal
  SSD-backed Engram assets. Admission verifies checkpoint metadata, source
  revision/LFS digests and full weight bytes (unchanged-file verification cache).
  Exact checkpoint inventory/tokenizer hashes are in the raw identity receipt.
- Source **`7c5c42ccb9a58c41cc2493bae7904f396b64d0dd`**. No runtime-source edits.
  The only new executable code is the external benchmark driver.
- Fresh **non-editable** installed `ds41f-mlx==0.23.0`, wheel
  `ds41f_mlx-0.23.0-py3-none-any.whl`, SHA256
  **`89cf312b6e6f86cf641bb49923e0c7ae84b7b52cb35d0913e0a40f7991160bf5`**.
  Execution identity:
  `91885864c6c57cf8ccd30e2b1c80bced99d5c84e90e3d6602532f7d9d8d8ba0a`.
- **Native GLM**, ordinary `ProductionMTPBackend` / `ProductionScheduler`,
  `DENSE_P0_P7`, MTP / DSpark / speculative decode **ON**. Not singleton serving,
  an archived prototype, development executor or the old M41 package.
- Pinned **oMLX 0.7.0** with the current admitted compressed-growth,
  selected bit-exact-kernel, template-reuse and production-safe context-copy
  overlays. MLX / MLX-Metal **0.32.2**; MLX-LM
  **0.31.4.dev132+g94cdcae13**; deepseek-recipe **0.1.1**, exact qualified native
  recipe wheel SHA256
  `439d9eaab77a0edb45ad250cb937ecf425e2b315370489085eb02e2315e592e2`.
  Transformers **5.17.0**, tokenizers **0.23.2**, FastAPI **0.142.2**,
  Starlette **1.7.0**, httpx **0.28.1**. Full dependency inventory, native
  compatibility admission, origins and payload hashes are retained below.

The existing released/operator environment was not modified. The canonical
provisioner built this fresh current-source package; its shared MTP distribution
supports ordinary serving despite the provisioner's internal `mtp-singleton-v1`
build-profile name. Runtime config and execution in this benchmark are ordinary
`mtp-serving-v1`. Provisioning's generated checkout manifest was saved as evidence
and restored, not published as a release-manifest update.

### Effective generation policy (all six requests)

| Setting | Effective value |
| --- | --- |
| temperature | 1.0 |
| top_p | 0.95 (trace float representation 0.949999988079071) |
| top_k / min_p | 0 / 0.0 |
| repetition / presence / frequency penalty | 1.0 / 0.0 / 0.0 |
| seed | unset (`null`), ordinary stochastic sampling |
| reasoning effort / thinking | `none` / disabled (headless default, explicit in body) |
| requested output | 256 tokens; actual 256 on every request, finish `length` |

Sampling fields were omitted from request bodies to use runtime defaults, not
silently replaced with greedy generation. `execution_sampling=stochastic` was
asserted on every run. Outputs are intentionally length-capped and not scored as
answer quality or proof of tool execution. No OFF reference was taken.

## Reproduce

From the recorded source checkout, with [provisioning prerequisites](operations.md)
and the official external checkpoint, choose **absent** absolute environment/build
paths. Commands used on this host:

```sh
/opt/homebrew/bin/python3.13 -m ds41f_mlx.mtp_setup \
  --venv /Volumes/SDXC-512/ds41f-public-benchmark-mtp-venv \
  --build-dir /Volumes/SDXC-512/ds41f-public-benchmark-mtp-build

export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
/Volumes/SDXC-512/ds41f-public-benchmark-mtp-venv/bin/python -I \
  tools/bench_public_mtp.py \
  --output artifacts/public-performance/measurements.json
```

[`tools/bench_public_mtp.py`](../tools/bench_public_mtp.py) is the exact workload.
It invokes the installed production `create_app` through FastAPI `TestClient`,
using ordinary `/v1/chat/completions` **JSON** requests. No socket, special cache
injection, mocked state, executor monkeypatch, logical-session protocol, restore
shortcut or automatic resend is used. Admission occurs before physical-engine
imports, as in canonical launch. An initial driver setup attempt used the wrong
import order and failed admission before model load; this was corrected in the
driver, without changing runtime/admission. No completed timed request was
excluded or retried. This measures production handler/engine work, not Uvicorn,
TCP, SSE buffering or remote-client latency.

### Prompt construction and run count

- Fixed sequence: one **8K**, one **32K**, one **256K cold** prompt, followed by
  **three consecutive continuations** of the 256K conversation. Here K means
  1024; actual official-recipe-encoded counts, not byte estimates, are below.
- Each initial user message begins with a size-specific maintenance-journal
  header, repeats a fixed ledger sentence about checkpoint/branch coherence,
  and asks for a detailed engineering plan with at least twenty recommendations.
  The driver sizes the repetition count with the official recipe tokenizer and
  authoritative `prepare_request` outside request timing. The resulting lengths
  are within 14 tokens of the named sizes; none are replaced with tiny fixtures.
- Every continuation appends the **actual preceding assistant response** plus
  the numbered request for at least twenty further recommendations. Each adds
  **41 recipe prompt tokens** beyond the preceding settled frontier. The full
  conversation still passes through normal recipe conversion/encoding/lookup.
- Each cold request has **zero cached tokens**, asserted, without artificially
  clearing/replacing production state. The size-specific headers prevent hits
  on previous initial prompts.
- Model load is outside request timing (**75.1 s**). "Cold" is conversation-state
  cold, **not** cold model/disk/Metal pipeline startup. All rows share one process
  and normal compiler caches; the first row can include first-shape costs.
- Cold runs are deliberately single observations to bound long-prefill cost.
  The primary continuation row in README is the **independent per-column median
  of three consecutive turns**, not three copies of the same request. Decode
  range: **19.6–21.1 tok/s**. Acceptance range: **50–56%**. No confidence interval,
  all-prompt guarantee or statistical significance is claimed.
- Prompt construction is deterministic; unseeded stochastic responses are not.
  Raw responses/canonical IDs allow reconstruction of the measured continuations,
  and each exact request has a stored SHA256. Re-running may change acceptance,
  generated text and timings. This repetitive synthetic journal is a reproducible
  session-state workload, not a claim to represent all natural agent transcripts.

## Timer and counter definitions

All seconds are monotonic host `perf_counter` intervals. The original baseline
uses existing runtime instrumentation. The follow-up adds only original-call-once
observational wrappers at the P6 transaction and native-owner boundaries, as
described below; no invasive tracing infrastructure or kernel changes were added.

- **TTFT:** `trace.total_ttft_s`, from handler arrival (`arrival_t0`) to the first
  canonical generated token returned by `_next`. Includes request preparation,
  lookup/restore, suffix prefill, prompt checkpoint capture, P5 handoff and first
  decode call; model is already loaded, no competing requests. This is **not**
  first client-visible text/chunk: canonical/protocol tokens may precede text.
- **Prefill/P5:** `trace.prefill_handoff_s`, from resource-lease acquisition before
  suffix execution to completed P5 generation handoff, including prompt capture
  and terminal bootstrap. Excludes earlier preparation/lookup/paired restore and
  subsequent `_next` decode. It is an inclusive phase, **not pure bulk-prefill
  compute**, so no ambiguous prefill tok/s is published.
- **Suffix append:** `trace.suffix_append_s`, synchronized append execution and
  DSpark tail initialization/update, excluding separately timed prompt checkpoint
  capture and later P5 handoff. Cold append consumes prompt-minus-one tokens;
  retained append consumes 40 tokens with one terminal token held for P5.
- **Decode:** actual canonical generated tokens (**256**) / `trace.decode_s`.
  The denominator sums `_next` / `owner.next_token` calls, including first call,
  native MTP proposal/verify/rollback and semantic-guard work inside those calls.
  Excludes recipe preparation, prefill/P5, paired restore, asynchronous delivery
  gaps and final settlement/publication. It is **not** 256 / wall time or a
  post-first-token network generation rate.
- **Wall:** external `client.post` entry to complete JSON response, including
  preparation, generation, settlement/publication, response delivery and local
  test-client overhead. Runtime `elapsed_s` is also retained, but is not exactly
  this external interval. Nested timers must not be added to their parents.
- **Drafts:** proposed = `copy_proposed + native_proposed`; considered =
  `copy_verified + native_verified`; accepted = `copy_accepted + native_accepted`.
  **Acceptance = accepted / considered**, not accepted / all generated tokens.
  Ordinary context-copy proposals remain enabled. Unverified end-of-budget
  proposals explain proposed > considered. Legacy `depth_drafted` is not used
  as a substitute for verified-token counts.
- **Memory:** existing MLX `get_peak_memory()` sampled after each request,
  **cumulative process allocator peak**, not per-case reset, total system memory,
  RSS, SSD residency or a sampled physical footprint. Process peak was
  **327,969,726,252 bytes (305.4 GiB)**. No broader memory instrumentation added.

## Measurements

All rows generated **256 actual tokens**. Rounded values below; raw floating-point
intervals/counters are preserved in JSON, not claims of equivalent precision.

| Workload | Prompt tokens | TTFT s | Prefill/P5 s | Suffix append s | Decode s | Decode tok/s | Wall s | Drafts proposed / considered / accepted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K cold | 8,178 | 19.83 | 19.59 | 18.02 | 10.46 | 24.5 | 30.35 | 247 / 244 / 129 | 53% |
| 32K cold | 32,764 | 38.99 | 38.90 | 38.73 | 10.63 | 24.1 | 49.79 | 249 / 248 / 136 | 55% |
| 256K cold | 262,132 | 321.53 | 321.25 | 321.05 | 12.53 | 20.4 | 334.32 | 250 / 248 / 133 | 54% |
| continuation 1 | 262,429 | 0.846 | 0.546 | 0.453 | 12.14 | 21.1 | 13.08 | 248 / 245 / 138 | 56% |
| continuation 2 | 262,726 | 0.844 | 0.549 | 0.454 | 13.05 | 19.6 | 13.99 | 250 / 248 / 124 | 50% |
| continuation 3 | 263,023 | 0.834 | 0.538 | 0.449 | 12.30 | 20.8 | 13.27 | 247 / 245 / 133 | 54% |

Continuation per-metric medians: **0.84 s TTFT, 0.55 s prefill/P5,
20.8 tok/s decode, 13.3 s wall, 54% acceptance**. Pooled acceptance is separately
**395 / 738 = 53.5%**, not the median aggregation used in README.

### Primary result: retained paired-prefix continuation

| Turn | Restored paired frontier | New prompt suffix | Settled target / DSpark frontier | Paired restore time |
| --- | ---: | ---: | ---: | ---: |
| 1 | 262,388 | 41 | 262,685 | 7.4 ms |
| 2 | 262,685 | 41 | 262,982 | 7.5 ms |
| 3 | 262,982 | 41 | 263,279 | 7.5 ms |

Each restored frontier is **exactly the preceding request's settled frontier**,
including its 256 generated tokens, not merely a hit on the original journal.
Ordinary `PairedCheckpointAuthority.acquire` restores the compatible complete
pair; target-only reuse is not admitted. At every settlement all **40 target
cache offsets and all three DSpark ring offsets** equal the canonical frontier;
cache publication succeeds, prediction is retired and queue is empty.

For **every cold and retained request**:

- `prompt_replay = 0`, `full_cache_repack = 0`;
- quiescence `history_replay = 0`, `full_cache_repack = 0`,
  `new_verify_cycles = 0`, `new_proposals = 0`;
- quiescence performs its normal one terminal target forward and DSpark append,
  not a hidden replay of the full history;
- no fatal error, cancellation, active/queued request remains after completion.

These existing counters plus the exact restored frontiers and 41-token suffixes
establish incremental execution. They **do not mean zero copying/allocation**:
ordinary immutable paired snapshot/restore and packed-state growth can copy
logical arrays. Zero full-cache repack does not assert a zero-copy cache path.

## Authoritative prefill execution (added diagnostic pass)

The original inclusive `prefill_handoff_s` was **not an execution-only prefill
measurement**. The follow-up supplies that missing stage measurement on the exact
same installed wheel, checkpoint, defaults and deterministic cold prompt bytes.
All three cold request SHA256s match the original receipt. The original public
TTFT/decode observations remain unchanged, not replaced by these new samples.

[`diagnose_public_mtp.py`](../tools/diagnose_public_mtp.py) times the actual
**`DeferredPrefillAppend.execute_all()`** call, from after an entry synchronization
to completed exit synchronization on `generation_stream`. Each original is called
once. The interval includes its begin/segment execution/final seal, deferred
encoder/decoder work, segment setup and materialization; it is host elapsed time
for real P6 work, **not a GPU-only kernel sum**. Token count is the sum of the
actual transaction `T-C` frontiers, not total context divided by TTFT. The raw
receipt records every transaction, segment mode and frontier.

| Case | Actual tokens executed by P6 | P6 execution s | P6 tok/s | Inclusive suffix append s | Prefill/P5 s | Canonical TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K cold | 8,177 | 8.88 | 921 | 16.62 | 17.98 | 18.19 |
| 32K cold | 32,763 | 33.33 | 983 | 38.64 | 38.81 | 38.90 |
| 256K cold | 262,131 | 315.53 | 831 | 320.86 | 321.06 | 321.34 |
| ~256K continuation | **40 new** | **0.250** | **160 incremental** | **0.451** | **0.539** | **0.852** |

The prompt contains one additional held-out terminal token, executed by P5.
The retained turn restores **262,388** paired tokens, then processes **40 P6 +
1 P5 prompt tokens**; the restored history is **not** in the throughput numerator.
For cold 8K, P6 transactions contain 8,176 + 1 tokens; 32K: 32,634 + 128 + 1;
256K: 262,002 + 128 + 1. These are genuine production handoff boundaries.

**P6 execution is not the entire prefill pipeline.** The wider synchronized
`suffix_append_s` also includes owner creation/planning, completing the observed
hidden-tap reductions and DSpark rings, and inter-transaction work. Checkpoint
capture and P5 are outside that timer and inside `prefill_handoff_s`. In
particular, the 8K case has a large prefill-pipeline residual outside the P6
transaction interval; quoting only 8.88 s as full prefill latency would be wrong.
The table therefore publishes the inclusive intervals alongside execution rates.
The wrappers do not separately attribute the P6-external residual to kernels.
Entry/exit synchronization prevents lazy GPU debt inside the transaction from
being charged to decode. No P6 throughput was derived from TTFT, and no runtime
execution math, topology, sampling or cache policy was changed.

README's new P6 column comes from **this single diagnostic pass**; its original
TTFT/decode/acceptance columns retain the original observations/three-turn medians.
Compiler caches were not forced cold in either pass, and the new synchronization
boundaries mean phase times must not be substituted silently into the original
raw receipt.

## Historical gap diagnosis: EXPECTED / EXPLAINED

**The 40+ → 20–25 tok/s comparison did not hold workload/acceptance fixed.**
Current ordinary production still executes the exact historical rewrite at
**43.78 tok/s**, with all **384 canonical tokens and MTP counters identical** to
the original historical run and its newly reconstructed environment. Its
contemporaneous historical replay measures **44.02 tok/s**, a **−0.55%** current
throughput difference, not a material factor-of-two implementation regression.
Relative to the original 43.87 measurement, the current difference is **−0.20%**.
These bounded differences are not claims of statistical equivalence to arbitrary
precision; the observed factor-of-two loss does not survive matched work.

### Recovered authoritative historical contracts

All used the official checkpoint on M3 Ultra / 512 GB, MLX **0.32.2**, oMLX
**0.7.0**, MLX-LM **0.31.4.dev132+g94cdcae13**, recipe **0.1.1**. These version
names alone do not mean identical source/distribution: overlay and payload hashes
are retained in [historical-contracts.json](../artifacts/public-performance/historical-contracts.json).

| Historical evidence | Actual contract and boundary | Throughput / acceptance |
| --- | --- | --- |
| [M41 final control](../artifacts/m41/final-performance.json), implementation `c5eb83f` (full commit in extracted receipt) | `make_prompt` in `run_m26_upstream_mtp_bench.py`: repeated Python tokenizer/parser code, truncated to **4,096 raw prompt IDs**; **128 output**, greedy argmax, no thinking/chat-recipe prompt conversion; configured MTP depth **5**. Stock `BatchGenerator.next` + synchronize, guarded MTP; timer starts **after upstream prompt prefill**, includes guard/loop work, excludes HTTP, Scheduler/P6 and final quiescence. One preceding MTP warmup; fresh native cache per run. | **43.20 tok/s**, **103/109 = 94.5%** considered acceptance; 23 cycles. Same direct target-only control **19.47 tok/s**. Not current HTTP throughput. |
| [M53R matched application sanity](../artifacts/m53r/performance.json.gz), source `1b1f867a3bb30a6a0f670c63e37b89fa4361413b` | Old **`mtp-singleton-v1`**, 242 recipe prompt IDs: “Write OK exactly forty times separated by spaces and nothing else.” **64 requested, only 41 canonical tokens actually generated**, greedy, reasoning none, configured max draft depth **3**. One warmup + three measured fresh sessions; preloaded process, SSE. Rate = canonical tokens / sum of `_next`, not HTTP wall. | Mean **40.38 tok/s**, **29/30 = 96.7%** acceptance, 10 cycles per request. HTTP wall ~2.45 s (~16.7 canonical tok/s), distinctly narrower than the headline decode rate. |
| [Context-copy closure](context-copy-closure.md#performance-recheck-after-edge-closure), source base `4e2e862b4146be4bbc897511dd50a25cb65f7942` + frozen final candidate wheel | Already **ordinary production**, not singleton serving. Exact checked-in `bench_template_reuse.py` file rewrite (`prefix` → `kernel_prefix`), **1,110 recipe prompt IDs**, **384 output**, greedy, headless reasoning none, max draft depth **3**, context-copy enabled. Warmup, then alternating native/copy/native/copy controls in one loaded process; the two copy hits restore 1,108 tokens. Same `_next` decode-s boundary as today's public benchmark, JSON wall separately recorded. | **43.87 tok/s**, **286/288 = 99.3%** accepted, 96 cycles; 252 copied drafts verified / 251 accepted, native 36 / 35. Its ordinary-explanation control was already only **23.18 tok/s**, **172/382 = 45.0%**, greedy, 210 cycles—not 40+ for every prompt. |

M41 and M53R are recovered contracts, **not fresh executions of those old
implementations**. Their recorded payloads have no GLM native extension; a serving
profile name does not establish today's native distribution. The context-copy
final wheel is likewise extension-absent/portable GLM (MLX GPU, not CPU), and was
reproduced exactly as described below. None of these historical numbers described
a 256K accumulated conversation, ordinary stochastic journal, or 256 output
under the current benchmark contract. M53R's requested 64 must not be mislabeled
as 64 actually generated. M41/M53R acceptance uses their retained legacy
`accepts / sum(depth_drafted)` **decision-prefix** denominator, not today's
`copy_verified + native_verified` full verified-token counter. Those percentages
must not be treated as numerically interchangeable without that caveat. The
primary matched context-copy comparison uses the **same modern counters** on all
three implementations/receipts.

### Matched current controls and historical implementation replay

The current side remains the **unchanged non-editable public-benchmark native
wheel**, SHA256 `89cf312b…1160bf5`, with ordinary `ProductionScheduler` and default
max draft depth **3** (checkpoint `n_mtp_layers`). Greedy is an explicit diagnostic
request setting only; no README/runtime default changed. Context-copy remains
normally enabled; no artificial depth/acceptance tuning was performed on serving.

| Ordinary production workload | Policy | Decode tok/s | Accepted / considered | Verification cycles | Canonical tokens / cycle |
| --- | --- | ---: | ---: | ---: | ---: |
| Current 8K journal, cold reconfirmation | defaults | 24.50 | 134 / 245 (54.7%) | 120 | 2.13 |
| Current 8K journal, paired hit | defaults | 24.91 | 137 / 245 (55.9%) | 117 | 2.19 |
| Current **same** 8K journal, paired hit | greedy | 25.21 | 137 / 247 (55.5%) | 117 | 2.19 |
| Current ~256K journal continuation | defaults | 21.29 | 139 / 246 (56.5%) | 115 | 2.23 |
| Current exact historical rewrite, two warm hits | greedy | **43.78 mean** | **286 / 288 (99.3%) each** | **96 each** | **4.00** |
| Current exact historical rewrite, warm hit 1 | defaults | 42.72 | 284 / 291 (97.6%) | 98 | 3.92 |
| Current exact historical rewrite, warm hit 2 | defaults | 43.32 | 286 / 288 (99.3%) | 96 | 4.00 |
| Reconstructed historical wheel, same rewrite, two warm hits | greedy | **44.02 mean** | **286 / 288 (99.3%) each** | **96 each** | **4.00** |
| Reconstructed historical wheel, same 8K journal | greedy | 25.45 | 136 / 245 (55.5%) | 118 | 2.17 |

**Sampling alone does not explain the low acceptance.** Changing just the
journal's policy from current defaults to greedy changes 24.91 → 25.21 tok/s
(**+1.2%**), with acceptance essentially unchanged. Conversely, leaving defaults
unchanged but using the historical rewrite yields 97.6–99.3% acceptance and
42.7–43.3 tok/s. The repeatable rewrite's already-present code gives ordinary
context-copy proposals a useful source; the journal asks for new recommendations,
not copied text. The retained ~54% acceptance is therefore supported as a
**workload-dependent result**, not a global stochastic-policy ceiling or evidence
that draft depth was accidentally reduced. The existing adaptive controller
considers fewer later-depth tokens after rejection; max depth stays 3 on both
current serving workloads. M41's older max-depth-5 figure is a separate contract,
not a reason to tune today's production controller.

The original historical candidate environment no longer existed. Its **exact
final non-editable wheel and complete dependency source export did exist**:

- wheel: `artifacts/context-copy-closure/final-runtime-wheels/ds41f_mlx-0.23.0-py3-none-any.whl`,
  SHA256 **`6ae234f0279e32c11186db2947f292b56fc2362ca1b7c3cad6f76eb0882a5b5e`**;
- reconstructed separate environment:
  `/Volumes/SDXC-512/ds41f-public-historical-mtp-venv`;
- installed the original normal-local dependency lock, frozen native recipe wheel,
  the exact `artifacts/context-copy-closure/candidate` oMLX source export and this
  historical runtime wheel; copied the unchanged recipe tokenizer resource;
- **unchanged original installed-payload admission passed**, with exact original
  identity **`303077956acb20149b91de8f944dabad77853c1958f480772cf29e90be1317d4`**.
  No source allowance, manifest reseal or mutation of the current environment.
  All **689 original oMLX files** also match the current canonical build export
  byte-for-byte; today's installed oMLX only adds the three frozen native GLM
  artifacts. This directly rules out an unrecorded oMLX Python/controller overlay
  change between this historical candidate and the measured current package.
  The wheel's source-base field is not a claim that uncommitted candidate deltas
  were absent; its frozen payload/identity is the implementation authority.

Historical replay uses the same request hashes, output budgets and JSON production
boundary. The two rewrite runs have **identical complete canonical IDs and
proposed/verified/accepted/depth/cycle counts** across original historical,
reconstructed historical and current native runs. The journal's greedy
trajectories differ between old portable and current native distributions, so its
25.45 vs 25.21 comparison is a matched request/policy observation, **not** an
identical-token math-equivalence assertion. This task does not reopen numerical
qualification or claim cross-distribution bit identity.

### Quantitative attribution: useful output per native verification cycle

The mechanism is visible without a new profiler:

- Current greedy rewrite: **384 / 96 = 4.00** canonical tokens per cycle,
  about **91.4 ms/cycle**, yielding **43.78 tok/s**.
- Current default 8K journal hit: **256 / 117 = 2.188** tokens/cycle,
  **87.8 ms/cycle**, yielding **24.91 tok/s**.
- Thus **(2.188 / 4.00) × (91.4 / 87.8) ≈ 0.569**, accounting for the
  measured **24.91 / 43.78 ≈ 0.569** throughput ratio. The journal's verification
  cycles are not ~2x slower; they deliver far fewer useful tokens per cycle.
- Current default ~256K continuation: **2.226** tokens/cycle at **104.6 ms/cycle**,
  yielding **21.29 tok/s**. The same decomposition relative to rewrite gives
  **(2.226 / 4.00) × (91.4 / 104.6) ≈ 0.486**, matching **21.29 / 43.78**.
  Longer-context per-cycle cost is material but is not the main acceptance gap.
  These are observed operation counts/intervals, not a proposed acceptance model
  or a universal speedup law.

The original ~20.8 tok/s retained median is consistent with this reconfirmation;
new unseeded samples are not required to reproduce the identical acceptance rate.
For example the added cold 256K run is 19.47 tok/s at 48.4% acceptance, while its
next retained turn is 21.29 at 56.5%. Nothing was discarded to improve the public
baseline.

### Native generation versus serving/owner costs

The follow-up wrapper also accumulates the existing per-call
`OMLXMTPGenerationSession.last_native_decode_s`: actual `BatchGenerator.next()`
**plus its normal stream synchronization**, inside the owner. This interval
includes native target/draft work and the semantic horizon/recipe guard invoked
by it; it is **not pure GPU time**, nor does it exclude sampling by definition.
The wider public `decode_s` includes owner history/connection/observation checks.

| Current case | Native BG s | Public decode-call s | Outside BG, inside decode s | Native-only rate |
| --- | ---: | ---: | ---: | ---: |
| 8K default cold | 10.410 | 10.450 | 0.039 | 24.59 tok/s |
| 8K default hit | 10.238 | 10.277 | 0.039 | 25.00 tok/s |
| Rewrite greedy hit 1 | 8.753 | 8.769 | 0.016 | 43.87 tok/s |
| ~256K default continuation | 10.957 | 12.026 | 1.069 | 23.36 tok/s |

The 256K owner residual is **8.9% of decode-call time**, not hidden full-history
model replay. The source boundary includes canonical-history tuple construction
and connection/prefix/observation checks after native generation; those checks
scale with accumulated history. This identifies the residual's bounded layer,
not a newly introduced regression or permission to optimize it. Even omitting
**all** that owner cost leaves only **23.36 tok/s**, not 40+.

For the retained turn, runtime elapsed is **12.960 s**, prefill/P5 **0.539 s**,
decode **12.026 s**, cleanup **0.059 s**, outer JSON wall **13.019 s**. The remaining
**0.395 s** after subtracting prefill/P5 and decode includes full-recipe preparation,
lookup/restore, scheduler/delivery gaps and settlement/publication. It is an
inclusive residual, **not an isolated Scheduler profiler measurement**. HTTP
wall throughput (~19.7 tok/s here) is lower than decode throughput, as expected;
HTTP costs were never included in the original public decode denominator.

Built-in telemetry is also retained: for the 8K default hit, backbone **9.038 s**,
MTP-head enqueue **0.338 s**, sampling enqueue **0.015 s**, cache ops **0.033 s**;
rewrite hit 1: backbone **8.447 s**, head **0.084 s**, sampling **0.016 s**, cache ops
**0.008 s**. These host/enqueue/synchronization regions overlap and do **not** form
an additive GPU-time breakdown. They support the operation-count explanation,
not an attribution of every remaining millisecond. No ~2x cost appears in
Scheduler, HTTP, sampling enqueue or settlement. M53R's narrower decode-vs-HTTP
boundary must be distinguished, but **context-copy's 43.87 and the public rates
already share the same decode-call boundary**: timer naming alone cannot explain
that particular gap.

### Narrow target-only reference and limits

One direct stock `BatchGenerator` reference in the **same current native model**,
with speculation disabled, the exact **8,178 recipe prompt IDs**, **256 output**,
greedy argmax and reasoning none, measures **19.73 tok/s**. It uses normal upstream
prefill, not P6, and times synchronized generation after prefill, without the
production recipe guard/Scheduler. It is a **target-only diagnostic**, not a
`standard-off` serving installation, and does not change production settings.

A companion direct control retained the recovered M41-style **configured depth 5**
and measured **23.85 tok/s** on this same journal (132/261 accepted; 122 cycles).
It is **not current ordinary depth-3 serving**, is not used to attribute serving
overhead, and is not a favorable headline substitute. Direct target-only, direct
speculative and served greedy trajectories are not token-identical; direct
upstream prefill/verification and serving P6/guard are different boundaries.
The direct pair's observed aggregate rate ratio is **1.21x**, not M41's 2.22x
on its high-acceptance code fixture. Consequently these references establish the
scale of target-only cost but **do not establish a clean identical-token or
ordinary-serving MTP speedup ratio**. The current served journal
is around 25 tok/s, not the old code fixture's 43+; the old M41 2.22x acceleration
must not be advertised for this workload. No OFF comparison campaign was opened.

### Commands, run counts and audit trail

Current diagnostic command (same admitted environment as the original benchmark):

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
/Volumes/SDXC-512/ds41f-public-benchmark-mtp-venv/bin/python -I \
  tools/diagnose_public_mtp.py --output artifacts/public-performance/diagnostic.json
```

Historical reconstruction and replay used these retained local inputs in a new
absent environment:

```sh
uv venv --python /opt/homebrew/opt/python@3.13/bin/python3.13 \
  /Volumes/SDXC-512/ds41f-public-historical-mtp-venv
uv pip install --python /Volumes/SDXC-512/ds41f-public-historical-mtp-venv/bin/python \
  -r third_party/mtp/requirements-normal-local.lock
uv pip install --no-deps \
  --python /Volumes/SDXC-512/ds41f-public-historical-mtp-venv/bin/python \
  third_party/mtp/deepseek_recipe-0.1.1-cp310-abi3-macosx_11_0_arm64.whl \
  artifacts/context-copy-closure/candidate \
  artifacts/context-copy-closure/final-runtime-wheels/ds41f_mlx-0.23.0-py3-none-any.whl
mkdir -p /Volumes/SDXC-512/ds41f-public-historical-mtp-venv/share/ds41f-mtp/recipe/static/tokenizers/v41
cp /Volumes/SDXC-512/ds41f-public-benchmark-mtp-venv/share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json \
  /Volumes/SDXC-512/ds41f-public-historical-mtp-venv/share/ds41f-mtp/recipe/static/tokenizers/v41/
/Volumes/SDXC-512/ds41f-public-historical-mtp-venv/bin/python -I \
  tools/compare_public_mtp_history.py --output artifacts/public-performance/historical-replay.json
```

For a fresh public checkout, the historical runtime wheel is now also
[archived with this evidence](../artifacts/public-performance/historical-wheel/ds41f_mlx-0.23.0-py3-none-any.whl).
The ignored old `candidate` directory is not needed: use the current canonical
provisioner's build export
`/Volumes/SDXC-512/ds41f-public-benchmark-mtp-build/omlx` in its place, and this
archived wheel instead of the old `final-runtime-wheels` path. All 689 admitted
oMLX files were verified byte-identical between those exports. This reconstructs
the original extension-absent environment without stripping files from or
mutating an admitted native environment. Admission must still pass unchanged.

**11 current production requests** (three cold, one long continuation, two matched
8K hits, one rewrite warmup and four rewrite hits), **two direct references**, then
**six historical production requests** (journal cold/greedy hit, rewrite warmup,
two greedy hits and one sampled hit). Only the explicitly identified rewrite
warmups are excluded from paired rewrite averages; all completed measurements
remain in raw receipts. Production loads were serial, never simultaneous.
No large-sample statistics or broad context/prompt sweep. The original six public
measurements remain the canonical baseline.

- [Raw current diagnostic traces (gzip JSON)](../artifacts/public-performance/diagnostic.json.gz),
  [log](../artifacts/public-performance/diagnostic.log).
- [Raw historical implementation replay and full admission identity (gzip JSON)](../artifacts/public-performance/historical-replay.json.gz),
  [replay log](../artifacts/public-performance/historical-replay.log),
  [reconstruction log](../artifacts/public-performance/historical-setup.log),
  [payload-admission log](../artifacts/public-performance/historical-inspect.log).
- [Derived execution rates, native/owner intervals and matched comparison](../artifacts/public-performance/diagnostic-summary.json).
- [Recovered historical contracts/counters and original evidence hashes](../artifacts/public-performance/historical-contracts.json).
- [Exact current/historical wheel identities and new script/evidence digests](../artifacts/public-performance/diagnostic-package.json).

The two large new raw JSON receipts are stored compressed; decompression restores
the exact generated `.json` bytes and recorded digests. Commands intentionally
produce the uncompressed working files consumed by the replay driver.

**Classification: EXPECTED / EXPLAINED for the apparent factor-of-two gap.**
The old high-acceptance rewrite still runs at its historical rate on the current
ordinary implementation; ordinary low-acceptance explanation was already ~23
tok/s historically. Workload-dependent useful output per verification cycle,
plus the bounded extra long-context native/owner cost, accounts for the observed
rates. Sampling policy and narrower historical HTTP boundaries are explicitly
controlled rather than assumed to explain the gap. No remaining material
implementation regression is demonstrated by matched evidence; this conclusion
is workload/configuration bounded, not a universal absence-of-regressions claim.
No optimization, topology/default change, numerical requalification, release
promotion or 1M campaign follows from this diagnosis.

## Evidence and bounded interpretation

- [Raw measurements and full admitted environment identity](../artifacts/public-performance/measurements.json):
  all six traces, responses, canonical token IDs, usage, timers, MTP stats,
  offset lists, quiescence receipts, request/script digests and dependency versions.
- [Derived rows/medians](../artifacts/public-performance/summary.json).
- [Wheel/package and evidence digests](../artifacts/public-performance/package.json).
- [Exact generated package execution manifest](../artifacts/public-performance/execution-manifest.json).
- [Provisioning log](../artifacts/public-performance/provision.log) and
  [completed run log](../artifacts/public-performance/run.log).

The original sample alone did not explain the historical throughput gap; the
matched diagnosis above now does. There was no OFF serving comparison,
performance optimization, new release acceptance or operator payload replacement.
The benchmark does not
establish a universal MTP speedup, real tool-effect correctness, natural-transcript
quality or long-term retention/endurance.

**256K retained-session operation is the practical-performance focus** because
this runtime targets developed long-running agent/session workloads. Cold 256K
prefill remains a substantial ingestion latency (~5.4 minutes here). Existing
**capacity/correctness validation up to 1,048,576 tokens** is a separate boundary,
not an implication of comparable 1M latency. No 1M request was run here. See
[performance interpretation](performance.md) and [capabilities](capabilities.md).
