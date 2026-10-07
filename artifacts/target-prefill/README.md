# Target prefill investigation (base `2f928cd`)

Decision: **NO PRODUCTION EXECUTION CHANGE**. See
[`docs/target-prefill-investigation.md`](../../docs/target-prefill-investigation.md).

- `native.json`: full native length ladder; later serialized probe failed on
  few-row grouped GatherQMM. Ladder timings completed; not a full PASS receipt.
- `final-controls.json`: corrected native PASS; BM8 exact-state controls,
  continuation, actual dense/Expert operand replays and serialized producers.
- `scaling*.json`, `producers.json`: admitted **portable** controls, before fixing
  diagnostic import order. Not normal native production throughput.
- `cone-controls.json`: unselected topology controls; output IDs match but state
  differs. Planner remains untouched.
- `suffix-controls.json`: unselected native attention reuse, source/history exact,
  changed decoder windows. At8192 raw continuation differs after first EOS.
- `candidate-integrated.json`: temporary integrated helper (without tool patch),
  about1.85s saved; helper restored before final commit.
- `candidate-suffix-math.patch`: exact unselected integrated candidate, applicable
  to the production package at the base. No runtime selector/promotion authority.
- `decode-controls.json`: same-resident-model, four repetitions per mode,
 32 steps; warm median1.594→1.630s (2.3% slower), identical32 initial IDs.
- `native-state-qualification.json`: native candidate8192/32768 ownership,
  same-list append, zero replay/repack and exact idle save/restore checks PASS.
- `r1-off.json` / root files in `r1-off-gates/`: full OFF R1 candidate PASS,
 24 gates, CONFORMANT. Frozen OFF source selects portable admission, so this is
  not a native P6 arithmetic-identity proof. Extracted frozen source/idle payloads
  are not duplicated here; authoritative R1 source archive is unchanged.
- `tool-reproduction.json`: final tool-only candidate against unchanged package.
- `summary.json`: regenerated from the above; explicit no-selection decision,
  prefix gain, state/trajectory caveats and paired decode regression.
- `source-dispatch.json`: inspected MLX/native dispatch source hashes and paths.
- `trace.log`: interrupted, memory-heavy Instruments attempt, no valid GPU event
  duration/count evidence. No trace timing claims or binary capture committed.
- `unit-tests.log`, `mtp-resource-tests.log`, `mtp-source-seal.log`: targeted
  checks and normal source-clone operator sealing against unchanged recipe wheel.

Use the documented sequential commands; model payload hashing/loading precedes
all measured prefill timers. Never run two model loads concurrently. The final
package execution files are unchanged from the base; diagnostic controls live only
in `tools/bench_target_prefill.py`. R1 verifier/material, MTP numerical paths,
resource guarantees, SSD Engram, Web and release/promotion remain untouched.
