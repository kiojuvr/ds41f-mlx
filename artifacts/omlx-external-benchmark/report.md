# oMLX External Benchmark — 2026-11-01

**PASS.** Official external warmup and all eight single-request trials completed
against unchanged Flash-1017 Production. No test size was omitted or retried.

## Standard and identities

- Upstream: <https://github.com/jundot/omlx>, exact revision
  **`17fce1d939f65a3cb92616862710c3007ea32b95`** (0.7.3 HEAD at acquisition).
  Measurements use its **unmodified** `omlx/admin/benchmark.py` and
  `omlx/admin/external_api.py`. [Revision](upstream-revision.txt),
  [source/corpus SHA-256](upstream-sha256.txt), [commands](commands.md),
  [configuration](configuration.json), [client dependencies](client-requirements.txt).
- Host: **Mac Studio / Apple M3 Ultra / 512 GB** (549,755,813,888 bytes), 32 CPU
  cores, macOS 26.5.2 / 25F84. Client and server are on the same host, over
  loopback HTTP at `http://127.0.0.1:8000/v1`, model ID `deepseek-v4.1-flash`.
- Production: `/Volumes/SDXC-512/ds41f-1017`, `ds41f-mlx==0.23.0`, installed
  source **`960036e9d831f25d1fa948e3d932cbce302427f1`**, **`mtp-serving-v1`**,
  **native GLM, paired MTP**, numerical profile `flash-1017-pre-woa-fp8-v1`.
  Installed execution-manifest SHA-256:
  `4962e642d11c5126b41551130210e4c4a78de65116f2e69397b9d3117dfc72b1`.
- Actual checkpoint: **DeepSeek-V4.1-Flash-1017** at
  `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`.
  The running process's environment and open checkpoint shard paths confirmed
  this path; it is not the original Flash checkpoint from the earlier README.
  Metadata/tokenizer hashes and file sizes/mtimes are in the identity receipt.
  No full checkpoint rehash, checkpoint conversion or model load was performed.
- **Corrected Recipe 0.2.0**, source `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`
  plus the installed native-semantic-boundary patch; admitted Recipe wheel
  SHA-256 `be5299f4649675ce2e6159a87bcc4a89c05bf24aa3b0bc7ba3f58d5124df38bb`.
  Production Python 3.13.15, MLX 0.32.2, MLX-LM
  `0.31.4.dev132+g94cdcae13`, subordinate oMLX 0.7.0. This installed substrate
  is distinct from the **0.7.3 external benchmark client revision**.

## Methodology and request settings

The bounded [invocation script](run.py) calls the upstream prompt builder,
external streamed client and single-test measurement function, without copying or
changing any measurement algorithm. It follows upstream external orchestration:

1. Official **quick warmup**: approximate prompt target 32, generation cap 8,
   temperature 0. Actual usage: **61 prompt / 8 completion / 0 cached tokens**;
   content observed, `finish_reason=length`, `[DONE]`. This also verifies
   `stream_options.include_usage` support before long requests.
2. One sequential single-request trial per standard target:
   **1,024; 4,096; 8,192; 16,384; 32,768; 65,536; 131,072; 200,000**.
   Context profile **`code_python`**, requested generation **128**, temperature
   **0**, no ANE alignment, no batching, no retries or repeated trials.
3. Each prompt uses upstream's fresh `BENCH-<UUID>` prefix and bundled corpus,
   estimated at four characters/token. These are **approximate target lengths**,
   not exact tokenizer lengths. Actual counts come exclusively from streamed
   endpoint usage, not SSE event counts.
4. Official `extra_body={"reasoning_effort":"none"}` keeps benchmark requests
   non-thinking. Production's default is `max` and is **not changed**. The
   installed Recipe converter confirms effective **temperature 0, top-p 1.0,
   thinking false**, top-k/min-p 0, repetition penalty 1, presence/frequency
   penalties 0, seed unset. See [effective settings](effective-settings.json)
   and the read-only [conversion probe](settings.py). No tools, stops or other
   generation-policy overrides are sent.

Production was already model-ready and idle. There is no cold model-load phase,
cache flush, session reuse request, native diagnostic pass or optimization.
Read-only health checks between trials require no active/queued requests before
sending the next trial. HTTP transport capture buffers bytes in memory and writes
raw files between timed trials. The client package namespace bootstrap avoids
unrelated admin-server imports; no server dependencies or fake benchmark modules
were substituted. See commands for the client-only import failures and resolution.

## Results

| Target prompt | Actual prompt / completion tokens | TTFT (s) | Processing estimate (tok/s) | Generation (tok/s) | End-to-end (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1,161 / 128 | 2.901 | 400.1 | 29.5 | 7.247 |
| 4K | 4,514 / 128 | 9.925 | 454.8 | 28.9 | 14.362 |
| 8K | 8,307 / 128 | 17.625 | 471.3 | 28.6 | 22.103 |
| 16K | 15,916 / 128 | 23.059 | 690.2 | 29.0 | 27.469 |
| 32K | 31,032 / 128 | 38.497 | 806.1 | 29.7 | 42.801 |
| 64K | 61,935 / 128 | 74.786 | 828.2 | 28.2 | 79.330 |
| 128K | 121,763 / 128 | 155.107 | 785.0 | 24.7 | 160.284 |
| 200K | 184,990 / 128 | 245.073 | 754.8 | 24.1 | 250.387 |

All eight streams returned HTTP 200, usage matching these actual counts,
`finish_reason=length`, and `[DONE]`. All reported **0 cached tokens**, 128
nonempty content deltas and no reasoning deltas. Content-delta counts are a wire
sanity check, **not the source of token counts**. Heartbeat comment lines ranged
from 0 to 16 per trial and were ignored by the upstream client. All TTFT,
processing and generation metrics passed upstream's observed-timing validity
conditions. [Wire validation](validation.json), [published table](published-table.md),
[verification log](verification.log). Each `pp<target>.request.json`, `.sse` and
`.http.json` retains the exact request, original response bytes and HTTP status;
`warmup.*` retains the same for warmup. No measurement failed or was omitted.

## Metric definitions and limitations

- **TTFT:** upstream `first_content_time - start_time`, where start precedes the
  HTTP request and first content is the first nonempty content/reasoning delta.
  It includes HTTP, server preparation, scheduling and delivery overhead. SSE
  comment heartbeats and role/empty deltas do **not** establish first content.
- **External processing estimate:** actual `prompt_tokens / TTFT_seconds`, rounded
  by upstream to 0.1 tok/s. This is **not direct native prefill/P6 throughput**.
  It is publishable only when content timing was actually observed. Zero reported
  cached tokens does not make it a synchronized native-prefill timer.
- **Generation throughput:** upstream `completion_tokens / (last_content_time -
  first_content_time)`, rounded to 0.1 tok/s; the numerator includes the first
  completion token (not `tokens - 1`). It is valid only for at least 16 completion
  tokens and a positive observed content span. It measures delivered generation,
  not accumulated native decode-call time.
- **End-to-end:** upstream `end_time - start_time`, including stream consumption
  through `[DONE]`, rounded to 0.001 seconds. README TTFT seconds are the official
  `ttft_ms / 1000`, formatted to three decimals. Other published values are copied
  unchanged from upstream result dictionaries.
- [Raw official metric outputs](results.json) and [execution log](execution.log)
  are the timing authority. Per-trial unrounded `StreamStats` timestamps are not
  returned by the upstream single-test function and are not retained; the
  warmup's returned timestamps are retained in [warmup stats](warmup.stats.json).
  [Verification/formatting script](verify.py) cross-checks every published count
  against raw usage and checks the processing estimate allowing upstream rounding.
- These are single observations under the stated conditions, not confidence
  intervals, universal throughput guarantees or an SLA. External Benchmark does
  **not directly measure MTP acceptance**, device peak memory or native execution
  phases. No historical native P6/decode/acceptance counters are mixed into these
  measurements.

## Documentation and final Production state

Replaced README's entire old performance table, values and methodological
footnotes with the verified external table and explicit benchmark conditions.
Updated `docs/performance.md`, the measurements index in `docs/README.md`, and a
short supersession notice in `docs/public-performance.md` so earlier native
receipts cannot be mistaken for current README methodology. No unrelated
historical evidence was deleted; no new historical benchmark campaign was added.
No Runtime source, generation policy, checkpoint, Recipe, installed dependency
or client configuration was changed.

Production's original server **PID 14750** (parent/operator **14746**) remains
running, model-ready, native GLM / `mtp-serving-v1`, with **0 active and 0 queued
requests** at completion. Its default reasoning remains **max** and all default
sampling settings, generation policy and admitted dependency identity match
initial health. The isolated benchmark client completed and is no longer running.
No restart, unload, cache-clear, cancellation or operator-setting write was issued.

[During-run identity](identity-during.json.gz) and
[final identity](identity-final.json.gz) agree on process start time/command,
installed versions, complete non-bytecode installation size/mtime inventory,
operator settings/hash, checkpoint file inventory/metadata hashes and execution
manifest hash. [Actual installed binary hashes](installed-sha256.txt) and
[upstream source integrity](upstream-integrity.log) are retained.
[Final state verification](final-state.json), [initial health](initial.health.json)
and [completion health](final.health.json) provide the operational receipt.
The metadata inventory was first captured during the run; it is not a full
before/after cryptographic rehash of every installed file or model weight.

**Transient cache state naturally changed:** no compatible prefix was restored;
aggregate counters increased by 18 queries, 18 misses and 18 evictions, with no
new hits (two lookups per warmup/trial). Existing cache contents were not cleared
or artificially restored. Preserved operational state means the same ready
process, installation, model and configuration—not bitwise restoration of live
cache contents after ordinary requests.
