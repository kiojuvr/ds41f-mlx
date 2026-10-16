# Ordinary MTP public benchmark — 2026-10-16

Small canonical practical-performance receipt for the current ordinary default
`mtp-serving-v1` implementation. **PASS: six real-model requests**, no runtime
changes or benchmark-specific tuning. This is a performance observation, not a
new release promotion, numerical qualification, endurance test or SLA.

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

All seconds are monotonic host `perf_counter` intervals from existing runtime
instrumentation; no invasive instrumentation or kernel changes were added.

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

## Evidence and bounded interpretation

- [Raw measurements and full admitted environment identity](../artifacts/public-performance/measurements.json):
  all six traces, responses, canonical token IDs, usage, timers, MTP stats,
  offset lists, quiescence receipts, request/script digests and dependency versions.
- [Derived rows/medians](../artifacts/public-performance/summary.json).
- [Wheel/package and evidence digests](../artifacts/public-performance/package.json).
- [Exact generated package execution manifest](../artifacts/public-performance/execution-manifest.json).
- [Provisioning log](../artifacts/public-performance/provision.log) and
  [completed run log](../artifacts/public-performance/run.log).

No concrete runtime correctness or severe performance regression was discovered
in this bounded sample. There was no OFF comparison, performance optimization,
new release acceptance or operator payload replacement. The benchmark does not
establish a universal MTP speedup, real tool-effect correctness, natural-transcript
quality or long-term retention/endurance.

**256K retained-session operation is the practical-performance focus** because
this runtime targets developed long-running agent/session workloads. Cold 256K
prefill remains a substantial ingestion latency (~5.4 minutes here). Existing
**capacity/correctness validation up to 1,048,576 tokens** is a separate boundary,
not an implication of comparable 1M latency. No 1M request was run here. See
[performance interpretation](performance.md) and [capabilities](capabilities.md).
