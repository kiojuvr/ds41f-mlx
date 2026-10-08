# M55 evidence

Decision: **NO / milestone not closed**. The bounded on-device materialization
repair is qualified. The dominant remaining physical blocker is canonical
accepted-prefix block production/undo: 41 one-row forwards versus the candidate's
10 four-row forwards on the same request. No default/runtime/release promotion.

See `docs/milestone-55-target-verification-topology.md` and `summary.json`.

## Qualification

- `pre-change-r1.json`: final M54 source, admitted Python 3.13.15, full R1 both
  profiles/real model, 25/25 PASS before runtime performance edits.
- `post-change-r1.json`: final modified runtime source, same full R1 once,
  25/25 PASS. Gate logs and JSON receipts are retained; extracted OFF source
  and temporary KV cache payloads are not committed. Runtime source hashes
  are checked by the report tool, independent of the receipt's pre-commit HEAD.
- `admitted-environment.json`: normal local identity inspect on baseline source.
- `real-prefix-state.json` and `real-prefix-with-taps.json`: independent OFF
  byte oracle for all 280 slots/metadata, every bounded prefix, compression/
  eviction, cancellation, faults and exact-list publication. The latter enables
  the actual first-party same-forward capture path, not a proposal substitute.
- `real-operational.json`: final, actually held EOF-worker socket/tool/effect/
  retry/re-entry matrix. Effects log is an observable fixture effect, not a
  persistence capability. 260 forwards, forbidden execution 0, five effects,
  duplicate effects 0; child/parent retired and receipts/producers empty.
- `bounded-regressions.log`: 260 affected tests; final additional copy/lifetime
  checks are in `final-device-copy-regression.log` (6 PASS).

## Measurements

`matched-{off,first-party-mtp-development,candidate}.json` retain one excluded
warm-up and three fresh-session samples. Outputs/request SHA/frontier agree.
Extended topology trials have one excluded warm-up and one diagnostic sample;
their counters/timings are not substituted for the three-sample rates.
`intermediate-host-copy-topology.json` demonstrates that consolidation alone
moves much of the waiting into row completion, not eliminating GPU execution.
`residual-phase-detail.json` observes the remaining M52/receipt/ring boundaries.
All real-model lanes ran sequentially, never with concurrent loaded models.

## Exclusions (not PASS receipts)

- `excluded-environment-r1*`: dependencies initially outside the new prefix;
  candidate admission correctly rejected the origin.
- `excluded-checkpoint-environment-r1*`: explicit checkpoint environment setting
  was missing; candidate admission correctly rejected it. Identity provisioning
  was completed before the final admitted baseline PASS.
- `excluded-route-observation-race*` and `excluded-unrearmed-eof-gate*`: original
  race harness confused disconnect cancellation with explicit-route admission,
  and an earlier case had already released its shared EOF gate. The final probe
  rearms the gate and observes the real cancel route before worker release.
  No runtime cancellation/admission predicate was relaxed.

Excluded gate logs are retained, but neither extracted OFF source nor KV payloads
from these setup attempts are included in the commit.

## Reproduction

Use the locally admitted Python 3.13.15 environment with the M54 native recipe
and delivered candidate dependencies. No PYTHONPATH or profile override bypass.
Set `DS41F_CHECKPOINT` to the official external asset. Use normal identity
`seal --wheel <M54-wheel>` and `inspect` after provisioning/after runtime edits.
This is development qualification, not a distributable environment.

```sh
P=/tmp/ds41f-m55-py315/bin/python
$P -m ds41f_mlx.reference --profile both --real-model --output <receipt.json>
$P -m tools.probe_m51_accepted_prefix --output <state.json>
$P -m tools.probe_m51_accepted_prefix --strategy first-party-mtp-development --output <tap-state.json>
$P -m tools.probe_m54_eof_application --output <application.json>
# One lane at a time; original M54 matched request is unchanged.
$P -m tools.measure_m54_matched_application --lane off --output <off.json>
$P -m tools.measure_m54_matched_application --lane first-party-mtp-development --output <mtp.json>
$P -m tools.measure_m54_matched_application --lane candidate --output <candidate.json>
# Optional diagnostic counts or residual boundary profiling, separate receipts:
# --topology --repeats 1 / --phase-detail --repeats 1
$P -m tools.report_m55_topology
```

The report asserts matched request/output/frontier identity, qualification PASS,
executed topology counters, and exact final runtime source hashes.
