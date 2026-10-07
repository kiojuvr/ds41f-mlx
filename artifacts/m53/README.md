# M53 evidence — NO / NOT PASS

Base commit: `6d7c936` (M52 PASS), parent `40bd187` (M51 PASS).

This is a **proposal-boundary assessment**, not first-party DSpark integration.
No actual proposals ran. Empty M52 cycles are controls, not an MTP workaround.
See `../../docs/milestone-53-first-party-dspark-proposal-boundary.md` for the
missing same-forward committed tap receipt and admitted proposal child boundary.

## Fresh commands

```sh
.venv/bin/python -m tools.probe_m53_proposal_boundary \
  --output artifacts/m53/proposal-boundary.json
.venv/bin/python -m pytest -q \
  tests/test_m47_resource_admission.py tests/test_m44_target_generation.py \
  tests/test_m51_accepted_prefix.py tests/test_m52_generation.py
```

Both exit records are **0**. Probe: **316.97 s**; regressions: **121 passed / 36.65 s**.
The probe loads the official full-payload M47-admitted checkpoint at the normal
configured location. It does not load donor MTP stages or a native scheduler.
A Python main-thread execution profile guard rejects donor MTP function calls
throughout owned prefill and decode controls; the observed call list is empty.
No standard selector or admission-policy bypass is used.

## Measured observations

- Official checkpoint: 2,401 `mtp.*` entries, three stages, width 5, taps
  37/38/39, window 128. Loaded canonical MTP stage/parameter counts: **0**.
- First-party `preserve_mtp=True` request rejects before loading; using the
  alternate donor-loading branch is deliberately not attempted.
- Canonical prefill proposal context is **None**; owned M51 output is two
  vocabulary-logit rows, without DSpark hidden receipt.
- M51 prefix-zero cancellation and empty M52 post-materialization cancellation:
  exact all-slot hashes/history/lookahead unchanged; child retired.
- Empty M52 control consumes one anchor and accepts zero proposals. This is
  explicitly not real DSpark acceptance.
- Two coherent same-list continuation starts: F=260, F=263. Terminal F=279:
  all 40 target frontiers equal committed history, stop reason `length`.
  No new independent deterministic or stochastic OFF parity trial is claimed.
- Tentative target fault: all aliases burned, journal retired, continuation
  rejected. Proposal failure cannot be tested without a proposal producer.
- Replay **0**, full-cache repack **0**, initial P5 handoffs **1**.
- Cleanup: admission retired, SSD Engram stores closed, proposal allocations **0**.
- Profiled **OFF-only** decode: 16 / 0.82635 s = **19.36 tok/s**. Actual MTP
  throughput and acceptance rate: **N/A**. This is not a throughput comparison
  with the historical native candidate's 38–40 tok/s.

## Files

`proposal-boundary.json/log/exit` is the determining fresh boundary run;
`owner-regressions.log/exit` is the affected baseline suite.
`source-audit.json` records reviewed core/policy/probe and donor source hashes;
these are source-audit identities, **not** a new resource-admission policy.
The default OFF checkout and installed isolated candidate have different
scheduler identities and are recorded separately; neither scheduler was executed.
`decision.json` binds the source/evidence bytes and marks all unmet M53 gates.

## Unmet qualification

First-party weights/math/rings ownership, actual proposal generation,
all/partial/first reject, acceptance statistics, derived-state advancement,
reset/retirement, private proposal RNG, canonical stochastic proposal cycles,
proposal-side failures, real-proposal cancellation/terminal/continuation and
candidate proposal/result comparison are **not qualified**.

Nothing in these controls authorizes runtime/application integration. Implement
and qualify the additive proposal primitives within M53 before advancing. The
existing uncommitted M49/docs-index/Chrome changes are excluded from this commit.
