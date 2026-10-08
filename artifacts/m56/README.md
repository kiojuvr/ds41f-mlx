# M56 evidence — canonical accepted-prefix block execution

Base: master `5bad422cf9b72f0096e0ce63f3132f9b97fd967a`.
Decision: **PASS bounded canonical block/state/application boundary; coherent
practical local operational candidate; no profile/default/release promotion**.
See `docs/milestone-56-canonical-accepted-prefix-block.md` and `summary.json`.

## Determining final receipts

- `final-all-prefix-state.json`: final runtime, official checkpoint F=4095/B=8,
  independent row-OFF all 280 slot/metadata byte hashes, every prefix 0–8,
  retained full-logit and same-forward tap identity, repeated 8/4/1/0/6 accepted
  spans, compression/eviction, two cancellations and 15 injected barrier faults.
  OFF continuation, exact-list identity, burn and SSD/resource retirement pass.
- `all-widths-state.json`: final runtime F=255, real greedy inputs, B=8 and
  every prefix of widths 2–7 (33 additional trials), all-state/tap/logit equality,
  cancellation, 15 faults and frozen M50 OFF state/latency regression pass.
- `final-proposal-loop.json`: real first-party DSpark, greedy and explicit-key
  stochastic OFF parity, rejection/low acceptance, cancellation and producer/
  target failure, legal continuation/warm-up, child/parent retirement. Its main
  loop observes 12 physical forwards / 67 physical rows, 64 generated inputs.
- `real-operational.json` + `.effects.jsonl`: actual M54 socket/tool/effect/retry/
  re-entry envelope, including held consuming-EOF worker result-loss. Seventeen
  turns, five effects, duplicates zero; every generated-ID sequence/frontier
  equals the M55 row receipt. 142 forwards / 260 rows / 5,680 layer regions;
  forbidden execution zero. No fake tool/model generation or extra effect ledger.
- `final-affected.log`: **498 PASS**. Reduced tests distinguish arithmetic/state
  fixtures from checkpoint evidence; no test/admission contract weakened.
- `matched-{off,first-party-mtp-development,candidate}.json`: unchanged M55
  request bytes/output/frontier, one excluded warm-up and three sequential fresh
  sessions per lane. Model load excluded, no concurrent loaded real models.
- `topology-{first-party-mtp-development,candidate}.json`: separate extended
  profiling trials, one warm-up excluded, never substituted for ordinary rates.
  Actual input widths and target/layer/attention frames are observed without
  additional model work or synchronization. Seven canonical blocks (6×6+1×5),
  280 backbone layer regions, 748 canonical-width attention subregions; candidate
  ten four-row blocks, 400 layer/attention regions. NOT GPU dispatch counts.
- `full-r1.json` and direct `full-r1-gates` logs/receipts: **25/25 PASS**, both
  profiles and real model, exactly one run after stable implementation and
  operational/matched evidence. Extracted OFF source/cache payloads are not
  committed. This does not admit first-party MTP persistence/restore.
- `admitted-environment.json`, `seal.log`, `inspect.log`: normal local Python
  3.13.15 identity on the existing M54 recipe wheel; no candidate admission bypass.
- `summary.json` / `report.log`: strict final source-hash, oracle, operational,
  matched-output, observed-topology and full-R1 reconciliation.

Final serving replay / full-cache repack / hidden target re-execution = **0/0/0**.
Offline independent qualification execution is not serving replay. Selected
already-packed prefix bytes and bounded dense numerical RHS scratch still copy;
no zero-memcpy claim. All-layer publication/settlement remains the M51 barrier.

## Failed/intermediate evidence (NOT PASS / NOT rate measurements)

- `real-block-state.*`, `real-block-matrix.*`: naive block arithmetic exposes
  real shape-dependent reduction differences (53 state slots on first trial).
- `geometry-block-matrix.*`: uncompiled normalization integration still differs;
  later source editing during this intermediate process correctly triggers M47
  resource-change rejection. This receipt is excluded, not final qualification.
- `compiled-geometry-block-matrix.*`: HC fixed; exact taps but FP32 compressor
  remainders differ. Final compressor uses the same existing batched GEMV geometry.
- `final-block-state.*`: successful intermediate 4K primitive, before final
  sparse-width integration. Preserved, not relabeled final-source evidence.
- `real-proposal-loop.*` and `greedy-six-state.*`: actual short-context greedy
  parity failure. Per-layer trace identifies layer-2 attention's leading invalid
  padding moving valid keys between BF16 softmax tiles. Final code subdivides
  only attention at canonical sparse-list width changes, not target forwards.
- `final-greedy-six-state.*`: all numerical/state/cancel/fault checks pass, but
  old driver modulo-8 indexing fails on a six-token fixture. Corrected driver
  changes no runtime contract; determining final widths receipt above completes.
- `excluded-preflight-state.*`: killed load after reduced tests exposed a local
  variable shadowing typo; final code separates input width from empty-slot width.
- `excluded-trace-observer.*`: diagnostic observer confused a Block local with
  an integer layer ID; no runtime/admission change to hide its failure.
- `excluded-width-driver-retirement.*`: optional extra-width driver initially
  ran after backend retirement. Existing admission rejects; driver ordering is
  corrected before the final successful run, with retirement still asserted.
- `excluded-{seal,inspect}-environment.log`: normal setup rejected missing OpenCV
  pkg-config path / explicit checkpoint; corrected environment then passes.

## Reproduction

Use the admitted `/tmp/ds41f-m55-py315` local development environment (Python
3.13.15, MLX 0.32.2, M54 consuming-EOF native wheel and existing delivered
candidate dependencies). `DS41F_CHECKPOINT` explicitly names the official asset.
No PYTHONPATH, model/admission override or profile relaxation.

```sh
P=/tmp/ds41f-m55-py315/bin/python
$P -m tools.probe_m51_accepted_prefix --execution block \
  --strategy first-party-mtp-development --output <all-prefix.json>
$P -m tools.probe_m51_accepted_prefix --execution block \
  --strategy first-party-mtp-development --initial-frontier 255 \
  --greedy-inputs --width-matrix --output <widths.json>
$P -m tools.probe_m53_proposal_loop --matrix --output <proposal.json>
$P -m tools.probe_m54_eof_application --output <application.json>
# Sequential, one lane/model at a time; no topology instrumentation in rates:
$P -m tools.measure_m54_matched_application --lane off --output <off.json>
$P -m tools.measure_m54_matched_application --lane first-party-mtp-development --output <mtp.json>
$P -m tools.measure_m54_matched_application --lane candidate --output <candidate.json>
# Separate diagnostic trials for MTP and candidate:
# --topology --repeats 1
$P -m ds41f_mlx.reference --profile both --real-model --output <r1.json>
$P -m tools.report_m56_block_execution
```

No `ds41f-runtime` update, release/package work, new kernels, default/profile
selection or capability expansion occurred. Normal-local-profile approval of
the bounded envelope and consuming-EOF native dependency is a separate next
checkpoint, not a blocker in the now-qualified canonical block producer.
