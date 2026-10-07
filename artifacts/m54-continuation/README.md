# M54 continuation evidence

Decision: **NO / M54 NOT PASS**. The two former primitives are implemented;
partial standard socket execution qualifies. Next primitive: exact non-mutating
EOF-sensitive tool preview matching standard M11's decision.

Authority: [continuation](../../docs/milestone-54-semantic-authorized-continuation.md).
Base: M54 assessment `2b68a69`, M53 PASS `bf8305d`.

## Official checkpoint, actual standard HTTP sockets

```sh
.venv/bin/python -m tools.probe_m54_standard_application \
  --output artifacts/m54-continuation/standard-mtp.json
.venv/bin/python -m tools.probe_m54_standard_application --strategy off \
  --output artifacts/m54-continuation/standard-off.json
```

These are sequential real full-payload loads using M47 admission and official
checkpoint defaults. The first-party selection binds an OFF target then admits
the M53 proposal child. Normal standard FastAPI routes, real loopback uvicorn
socket and HTTPX client; no candidate scheduler. Output/exit records are in
`standard-{mtp,off}.{json,log,exit}`.

MTP: exit 0, **321.48 s**. First JSON turn and subsequent SSE turn each consume
nine inputs and terminate on EOS. The first turn executes two real M52 cycles;
the short-ring continuation uses protected canonical steps. A fresh max-token
case consumes eight inputs in two cycles. Partial SSE socket disconnect drains
two cycles / twelve inputs and freezes the cancellation response. Full frozen
retries are byte-identical, including the exact delivered prefix, and perform no
new target calls or request-count publication. Native reconstruction is
representable/exact-prefix; post-disconnect new model generation was not run.
Tools reject HTTP 400 before reservation or target mutation.

43 target calls = four bootstraps + planned decode inputs 10 + 9 + 8 + 12.
Consumed reports 38; 6 real cycles, 22/23 accepts (95.65%). Replay=0,
full-cache repack=0; guard observed forbidden diagnostic/donor calls=0. All forty
offsets match history at terminal frontiers 251/275/249/253. Child receipts and
producers are zero at idle; parent and child inactive after retirement.

OFF control: exit 0, **322.24 s**; 33 target forwards, 9/9/8 inputs for the
first three cases, three inputs at disconnect, exact frozen retries,
representable/exact-prefix reconstruction, and parent inactive after retirement.
First-party/OFF first-three-case token/message/frontier parity is exact. The
socket disconnect boundary differs (protected cycle versus single-token drain),
so this is not a matched performance workload. `decision.json` records this
limited parity and the explicit remaining qualification gaps. `source-audit.json`
records checkpoint config/index and integration source identities and confirms
M51–M53 generation/settlement/proposal/receipt core files unchanged from `bf8305d`.

`exploratory-asgi-mtp.*` is earlier development evidence using ASGI transport and
short OK/YES outputs: proposal math ran but **zero M52 speculative cycles**.
It is not the determining operational execution evidence and is not a speed
measurement. The later socket probe explicitly requires real cycles.

## Regression command

```sh
.venv/bin/python -m pytest -q \
  tests/test_m44_target_generation.py tests/test_m45_target_forward.py \
  tests/test_m46_state_production.py tests/test_m47_resource_admission.py \
  tests/test_m48_model_execution.py tests/test_m51_accepted_prefix.py \
  tests/test_m52_generation.py tests/test_m53_proposal_primitives.py \
  tests/test_m53_proposal_producer.py tests/test_m54_authorized_cycles.py \
  tests/test_m54_failure_boundaries.py tests/test_m54_response_reservation.py \
  tests/test_m54_semantic_boundary_assessment.py tests/test_m54_tool_eof_horizon.py \
  tests/test_runtime_config.py tests/test_stateful_live_backpressure.py \
  tests/test_stateful_live_delivery.py tests/test_stateful_live_retirement.py \
  tests/test_stateful_request_policy.py tests/test_web_application_boundary.py \
  tests/test_m36r_recovery.py tests/test_m11_tool_boundary_contract.py
```

`regressions.txt` / `regressions.exit`: **247 passed + 8 subtests**, 46.69 s,
exit 0. Reduced real MLX, M53 stub proposal math, actual recipe/tokenizer, standard
transport doubles, and actual Web effect-ledger/counter tool have distinct labels.
Effect-then-transport-loss test: executions=1, duplicate effects=0; not an MTP tool
workflow. Final observation snapshots, direct-entry reservation, startup-error
burn, retired-producer, budget and reasoning guards were finalized after socket
probes and covered by regressions, not a new checkpoint/socket qualification. Actual recipe proof: standard EOF completion ordinal 31 versus native
DSML terminal ordinal 37. Tool-enabled requests remain fail closed.

No matched operational OFF/first-party/candidate benchmark: full correctness/tool
envelope is NOT PASS. Coarse timings in JSON are diagnostics only; execution
combines verification/sampling/settlement, not an isolated phase benchmark.
No full R1, default change, candidate removal, release/promotion, Vision/restore,
long-context, batching or kernel research. Existing Chrome log changes excluded.
