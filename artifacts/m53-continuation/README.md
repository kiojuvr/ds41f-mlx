# M53 continuation evidence — YES / PASS, development core only

Decision: `decision.json`; implementation/contract:
`../../docs/milestone-53-proposal-primitives-continuation.md`.
No application/runtime promotion and no native scheduler execution.

## Determining commands

```sh
.venv/bin/python -m tools.probe_m53_proposal_loop --matrix \
  --output artifacts/m53-continuation/loop.json
.venv/bin/python -m pytest \
  tests/test_m44_target_generation.py tests/test_m47_resource_admission.py \
  tests/test_m51_accepted_prefix.py tests/test_m52_generation.py \
  tests/test_m53_proposal_primitives.py tests/test_m53_proposal_producer.py -q
.venv/bin/python -m pytest \
  tests/test_prefill_fp8_mlx_p5.py tests/test_prefill_fp8_mlx_p6.py \
  tests/test_prefill_fp8_mlx_p7.py tests/test_m29_mtp_lifecycle.py -q
```

`loop.json/log/exit`: **exit 0**, 368.28 s. Full official checkpoint admitted,
first-party child loaded, actual DSpark seed/proposals/verification/settlement.
All admitted source files stayed static during this determining lifetime. The
final handoff type guard distinguishing retained legacy candidate context from
first-party receipts is separately covered by the P5 regression; it does not
change the first-party receipt branch exercised by the checkpoint run.

`regressions.log/exit`: **exit 0**, **145 passed** (40.18 s).
`prefill-regressions.log/exit`: **exit 0**, **51 passed + 26 subtests** (1.04 s).
Reduced proposal tests use real MLX but stub proposal arithmetic where explicitly
labeled: they are lifecycle/ownership evidence, not actual checkpoint proposals.

## Checkpoint observations

- Seed `[126,254)`, `[1,128,15360]`, followed by actual bootstrap to F=255;
  receipt retired. Three physical rings use post-layer means from 37/38/39.
- First proposal `[28231,3939,260,6341,6623]`: 5 accepts, 6 consumed inputs.
- 11 unmodified cycles: 64 tokens; 53/55 offered drafts accepted (96.36%). Ten
  all-accept cycles plus final length-truncated prefix, not natural rejections.
- Deliberately perturb **real** producer outputs: first reject consumes 1;
  partial reject accepts 2/consumes 3; both match independent sequential OFF
  all-slot/history/lookahead snapshots. Low-accept controls accept 0/25 across
  five cycles and still advance rings solely by committed anchors.
- Greedy and canonical MLX explicit-key temperature-0.7 stochastic sampling:
  32 tokens each, exact OFF final state. No stochastic proposal sampler exists.
- Cancellation restores exact canonical target snapshot with consumes=0;
  producer failure before verify and after partial derived-ring append leave
  coherent target state and allow only explicit idle disable, never implicit
  mid-cycle fallback. Target tentative failure burns all aliases/derived state.
- Two same-list continuations destroy prior rings, warm new rings with 128
  legitimate canonical forwards and resume real proposal cycles (F=395,535).
- Terminal F=319: all 40 cache frontiers = history; rings retire, receipts=0.
  Explicit child revoke releases stages/shared read handles and rejects entry;
  parent close leaves inactive child, producers=0, receipts=0.
- Replay=0; full-cache repack=0; hidden seed target reexecution=0. Profile guard
  rejects diagnostic `_forward` and donor model/MTP paths. Initial decode target
  calls = 67, exactly bootstrap + all eleven tentative six-position spans.

Measured profiled decode: **64 / 4.6360 s = 13.81 tok/s**. Coarse totals:
proposal math 0.1859 s; target verify + settlement 4.4119 s; receipt transport/ring
0.0376 s. No matched native candidate benchmark or direct numerical oracle run.
The historical ~38–40 tok/s candidate is only a reference; no speed-parity gate.

## Exploratory records (not determining PASS)

`initial-matrix.*`: earlier successful 356.34 s execution, before additional
rejection/low-accept/resource-revoke/forward-count gates and explicit read-handle
release. Its narrower observations are not used to claim the final gates.

`exploratory-source-change.*`: an earlier load rejected at 315.38 s because an
admitted proposal source was edited during the expensive payload/load phase.
M47 correctly rejected the changed source before publication. This is **not** a
proposal arithmetic blocker or a qualified decode run; the determining run was
restarted with static admitted resources. A preliminary direct script invocation
(without module/package path) also failed to import the project; commands above
use `python -m`. Neither exploratory failure is counted as PASS evidence.

`source-audit.json` records final implementation and donor identities. Donor
V4.1 DSpark math and V4 physical-ring algorithms are MIT numerical/algorithmic
sources only; native scheduling/lifetime is neither copied nor executed.
