# Long-frontier suffix comparison investigation

Status: **CLOSED — DIFFERENT GLM EXECUTION PROFILES, NOT A SAME-PATH RUNTIME REGRESSION**.

The installed-profile statements and measurements below are frozen investigation
evidence. Subsequent bounded promotion made GLM native the local MTP default;
OFF publication remains portable. See [current distribution admission](runtime-admission.md#glm-execution-distributions).
No historical measurements or correctness conclusions are requalified here.

The historical native suffix path has not lost performance in the bounded exact
control. Current installed MTP/OFF use the admitted **portable** GLM profile;
the historical long-context authorities used **native** GLM. Their target
attention/MoE execution and numerical trajectories are not identical. The
original unresolved restore-probe mismatch was a profile mismatch, not damaged
saved state. No runtime, numerical, precision, cache or resource-policy change
was made, and no native bundle was added to either installed runtime.

This closes the attribution question, not a new optimization, native-profile
publication, or long-context qualification campaign. The original OpenCode
request's raw IDs/receipt were unavailable: its exact 11.29 s was not replayed.
Nearby matched production-owner reproduction and an exact historical native
control establish the execution-profile distinction; they do not decompose every
millisecond of that unavailable request.

## Authority and timing boundaries

Historical authorities are `standard-off-200k-production-qualification.md` and
`very-long-context-production-qualification.md`, not fresh-prefix throughput.
The 2,049-token OFF collector's `bootstrap_s` surrounds exact-prefix admission,
P6 on 2,048 tokens, and P5/target terminal bootstrap, including completion debt.
Decode and settlement are separate. Historical receipts do not split its GPU
debt further.

Current ordinary MTP uses ProductionScheduler -> paired acquisition ->
`InternalMTPQualificationBackend._start` with a checkpoint callback:

| Work | Current accounting |
| --- | --- |
| Recipe conversion/render/encoding | `recipe_*_s`, `request_prepare_s`, before native append |
| Paired lookup / materialized restore | `cache_lookup_s` / `paired_restore_s`, before append |
| Wired lease acquisition | before `append_t0`, within coarse `prefill_handoff_s` |
| P6 through N-2, DSpark projections, ring evaluation, synchronization | `suffix_append_s` |
| Immutable prompt capture at N-2 | `prompt_checkpoint_capture_s`, explicitly subtracted from append |
| Completing one-token P6 to N-1, ring update/evaluation and synchronization | `suffix_append_s` |
| P5, native constructor, held-out terminal bootstrap | `p5_handoff_s`, outside append |
| First native proposal/activation and generation | decode / first decode / TTFT |
| Quiescence, final capture, publication and retirement | after generation, outside append |
| Request wall / first emitted canonical token | `elapsed_s` / `total_ttft_s`, not append |

Dashboard Prefill is
`max(0, prompt_tokens - cached_tokens - 1) / suffix_append_s`.
It is not pure P6 kernel throughput. P6 `execute_all()` return is not a GPU
completion boundary either. Time in `mx.eval(ring.keys)` includes incoming target
computation, not just DSpark-specific compute. The existing startup investigation
already established that distinction; its optimization campaign was not rerun.

The common end point is **terminal bootstrap ready**: OFF `bootstrap_s` versus
MTP `suffix_append_s + p5_handoff_s`. Account for MTP capture and wired acquisition
separately for whole startup. Timer definitions alone do NOT erase the original
comparison gap: the historical OFF timer was broader, not narrower.

## Current installed production-owner comparison

Both installed environments passed inspection with the explicit official
checkpoint. No runtime was listening on localhost:8000 at investigation start.
Model processes ran sequentially; installed payloads were not edited/resealed.

The tokenizer-derived maintenance-journal fixture is a token-level core probe,
not a reconstructed OpenCode conversation or a cache-compatibility investigation.
MTP uses ordinary preparation, ProductionScheduler, paired authority,
P6/checkpoint/P5, generation and settlement. OFF uses first-party prefill, real
P5 retirement to idle without consuming output, then ordinary M8 continuation.
Observers add no evaluation/synchronization barriers inside measured work.

The seed's generated pair matched four extra fixture IDs. Actual reuse was
209,004, not the harness's expected 209,000. The MTP request settled successfully;
only the observer's post-settlement expected-frontier assertion failed. Its raw
FAIL receipt is retained, not relabeled PASS. OFF uses exactly the actual restored
canonical IDs and remaining suffix (`method.json`, original/effective fixtures).

| Measurement | First-party OFF, portable GLM | Ordinary paired MTP, portable GLM |
| --- | ---: | ---: |
| Frontier / suffix / appended tokens | 209,004 / 1,769 / 1,768 | same |
| P6 host return, bulk | 5.603 s (whole append) | 5.539 s (1,767-token bulk) |
| Subsequent bulk ring evaluation | not a separate OFF phase | 4.517 s, including target debt |
| Completing 1-token append | part of OFF bulk | 0.0283 s host + 0.0304 s ring evaluation |
| Displayed append interval | OFF field is not the MTP metric | 10.124 s / 174.6 tok/s |
| P5 interval | includes outstanding target debt | 0.0807 s |
| **Common suffix-to-terminal-ready interval** | **10.1698 s** | **10.2049 s** |
| Common appended-token rate | 173.8 tok/s | 173.2 tok/s |

MTP's net difference is **0.0352 s / 0.35%** in this single case, not a precise
isolated DSpark cost or statistical speed claim. Lookup/restore total
**0.06285 s**, prompt capture **0.00651 s**, and request wall **10.464 s** are
separate. The 4.52 s ring evaluation is not removable MTP overhead: OFF pays
essentially the same completion debt after P6 host return, inside P5.

Both branches generated `[6319, 17840]`. OFF's 40 offsets equal final frontier
210,775. MTP settled at the same target/DSpark frontier with no queued prediction
and published its pair. Both report replay=0 and repack=0.

## Resolved blocker: physical state matched, execution profile did not

The original historical restore control correctly refused comparison:
280 saved slots matched shape/dtype/raw-byte SHA256, history and probe suffix
matched, but greedy output diverged at token 17 (index 16): current portable
373 versus historical native 4995. Both thread/stream placements gave the same
portable sequence. Saved arrays alone did not establish identical execution.

`profile-audit.json` now identifies the missing prerequisite:

| Runtime/control | GLM extension | GLM execution profile |
| --- | --- | --- |
| Historical 200K and very-long OFF receipts | present | native |
| Installed MTP environment | absent; manifest also declares no GLM bundle | portable |
| Installed OFF environment | absent; manifest also declares no GLM bundle | portable |
| Exact historical native control, current first-party core | present; all three files match existing admitted native hashes | native |

Portable means MLX GPU fallbacks, **not CPU inference**. Both installed runtimes
lack the extension, dylib and metallib. `has_symbol` is false for native packed
attention, grouped-expert and MXFP4/affine block GEMM symbols. This is not a built
extension failing to load: the immutable delivered inventories contain no bundle.
OFF resource admission explicitly supports both `native` and `portable` profiles;
ordinary installed payload admission accepts its exact portable inventory.
Thus admission PASS did not imply the historical native numerical/performance
profile. No pin was relaxed to run the native control.

The first profile-dependent production branch in the ordinary multi-row P6
suffix is target **layer-0 Attention**: `Attention.__call__` uses
`glm_fast.deepseek_v41_packed_attention` for eligible BF16 rows when its symbol
exists, otherwise `packed_sparse_attention`. Native MoE block/pair/grouped-expert
selection also differs later. Precision and packed representation are unchanged,
but the algorithms/dispatch are not bit-identical across profiles. This is a
structural first execution divergence inside target/P6, not a per-kernel GPU-time
attribution. It precedes MTP context preparation, checkpoint capture and P5.

## Exact historical native control

The control uses the current first-party core with its existing hash-admitted
native resources. Its core files match installed OFF. The preserved legacy
resource environment supplies the exact pinned native bundle and dependencies;
this is a first-party resource-admitted core diagnostic, **not** a new installed
native wheel or ordinary MTP/native deployment acceptance. The recipe tokenizer
comes from the existing admitted installed resource. No production environment
was changed and no model processes overlapped.

Starting from the historical saved frontier 213,150:

- History, probe suffix, and all **280 saved slots** match.
- Probe's **32 greedy tokens and all 280 post-probe slots** match historical bytes.
- The three preceding short turns' **768 generated tokens** match the receipt.
- The first 2K turn reaches exactly the historical **214,239** frontier and exact
  2,049-token suffix digest. Its first two generated tokens also match.
- All 40 final offsets agree; replay=0, repack=0.

| Matched historical-native workload | Past native bootstrap | Current native bootstrap |
| --- | ---: | ---: |
| 1-token suffix | 0.07086 s | 0.06884 s |
| 17-token suffix | 0.32724 s | 0.32682 s |
| 257-token suffix | 1.02142 s | 1.02292 s |
| **2,049-token suffix at frontier 214,239** | **8.63645 s** | **5.63640 s** |
| One subsequent same-shape 2K continuation | not an exact historical match | **5.63246 s** |

The comparable current native 2K rate is **363.4 appended tok/s**, versus
**237.1 tok/s** in that historical turn. Its P6 host interval is 3.449 s;
completion/bootstrap remainder is 2.188 s. These bounded measurements show no
loss on the historical native path. Existing intervening OFF resource-lifetime
changes include acquiring/retaining the wired budget before model allocation;
the 200K authority predates that very-long-context repair. We did not toggle or
reinvestigate that repair, nor attribute every millisecond of the improvement.

Historical 512K/768K/near-1M 2K results remain native-profile authorities
(7.59–7.63 / 9.12–9.19 / 10.42–10.51 s), not portable-profile performance promises.
They were reviewed, not rerun.

An earlier portable control used the same historical 2,049-token suffix directly
at saved frontier 213,150, measuring **11.668 s**. It lacks the historical
warm-up/frontier match and is not substituted for the exact native comparison.
It nevertheless illustrates why an unlabelled portable result cannot be used to
declare regression against the historical native receipt. No numerical identity
or exact native-versus-portable speedup is claimed for that unmatched control.

## Root cause, owner and disposition

- **Classification:** different admitted execution topology/profile, not a
  same-path runtime regression. Existing native suffix performance is preserved
  (faster in the exact bounded control). Portable MTP and OFF agree at a common
  boundary; seconds of MTP-specific extra preparation are not the cause.
- **Root cause:** the comparison treated source/version/admission/profile name
  agreement as native-dispatch agreement. Current source-export/provisioned
  payloads omit the optional GLM native bundle; historical core qualification
  retained it. Silent portable selection also explains the old-probe mismatch.
- **Owning layer:** dependency export/provisioning and execution-profile
  interpretation/disclosure, with a telemetry phase-definition distinction.
  Not OpenCode reconstruction, prefix lookup, P6 lost performance, DSpark
  initialization, P5 or checkpoint copying.
- **Repair:** **no runtime repair required for the current admitted portable
  delivery**. Do not inject native binaries, reseal payloads or switch profile
  to recover a headline rate. That would change the current numerical trajectory
  and need a separately explicit publication/affected-correctness decision.
  This investigation does not start such a campaign.
- **Before/after:** no runtime patch, hence no repair before/after claim. The
  affected matched portable OFF/MTP and historical-native comparisons above
  are the evidence. No rejected/deferred optimization was reopened.
- **Correctness/state:** no changes to model semantics, precision, canonical
  state, P5/P6 ownership, MTP consistency or resource lifetime. Native historical
  state equivalence is now proven for the exact probe; current portable paired
  lifecycle remains internally coherent. Cross-profile bit identity is not
  promised, and no cache compatibility rule was relaxed.
- **Operation:** displayed 157 tok/s measures a portable suffix interval that
  includes target completion debt and ring update, excluding capture/P5. It is
  not evidence that native 200K–1M suffix performance regressed. Seconds-scale
  portable continuation remains a real interactive-latency limitation. The
  exact reported 11.29 s was not replayed; its raw workload was unavailable.
- **General-runtime readiness:** no new same-path regression blocker from this
  observation; no new native-profile promotion or broad qualification claim.
  Historical native throughput must not be advertised as current portable
  throughput. Numerical continuity across those profiles must not be inferred
  from identical saved slot bytes. Existing accepted portable delivery is
  unchanged. Stop performance exploration here.

## Receipts, tools and validation

`artifacts/suffix-continuation/` contains installed inspection, original/effective
fixtures, `method.json`, `mtp.json`, `off.json`, `profile-audit.json`,
`historical-native.json`, `closure.json` (including the exact native-control command
and receipt hashes), rejected historical portable controls, `topk.json`,
logs and `tests.txt`. The native control is PASS; historical portable controls
remain FAIL, correctly retaining their numerical-equivalence rejection.

The raw MTP receipt retains its observer-only frontier assertion failure. The
first OFF harness incorrectly tried idle append directly from sealed fresh
prefill without P5 producer retirement; it was rejected/burned and exited.
Neither harness issue was a repaired runtime failure. Later OFF uses actual P5
retirement. An isolated top-k specialization check found equal values/IDs and
roughly 0.6–0.8 ms warm times, not a seconds-scale regression; no overlay changed.

Reproduction tools: `tools/diagnose_suffix_continuation.py`,
`tools/diagnose_historical_suffix.py` (now records/asserts the GLM profile before
comparing), and `tools/diagnose_suffix_topk.py`. Installed controls use Python
`-I`; the preserved native resource control uses the historical exact dependency
origins and current first-party core as explicitly described above. Never overlap
model processes or mutate installed payloads during a model lifetime.

66 affected unit tests and 12 subtests passed; Python compilation,
`git diff --check` and repository self-containment/typed documentation references
passed. The authority-label checker cannot pass its existing
repository-wide source-identity prerequisite: unrelated candidate-consumer and
compressed-KV receipts lack hash-method metadata and include a source-span hash
mismatch. Those receipts were not changed or counted as passes.
