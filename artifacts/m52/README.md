# M52 generation qualification evidence

Base: `40bd187392db0dc7c50f36c1bcfdfc90f8af124d` (M51 PASS).

**YES / M52 PASS**. Fresh checkpoint run: **404.5 seconds, exit 0**.

Commands (repository root):

```sh
.venv/bin/python -m tools.probe_m52_generation --output artifacts/m52/checkpoint.json
.venv/bin/python -m pytest -q \
  tests/test_m44_target_generation.py tests/test_m45_target_forward.py \
  tests/test_m46_completion.py tests/test_m46_engram_reads.py \
  tests/test_m46_state_production.py tests/test_m47_resource_admission.py \
  tests/test_m48_model_execution.py tests/test_m51_accepted_prefix.py \
  tests/test_m52_generation.py
```

`checkpoint.json/log/exit` is the determining fresh checkpoint run, including
native MLX RNG and EOS id 1 fixtures. `owner-regressions.log`: **178 passed**.
`checkpoint-initial.*` is a failed earlier attempt: numerical, generation,
cancellation and publication-fault assertions passed, but the final SSD observer
used a nonexistent `_engram_embed` attribute. It is retained as FAIL, not PASS
evidence. The observer now checks admitted layers' embeddings and prefetch
coordinator, and retirement through the live admission capability, as M51 does.
`tests-initial.log` is an earlier 78-test subset, not the final regression total.

The independent canonical OFF oracle is run first and its executable lists are
released before each speculative trial. No native scheduler is executed. The
proposal fixtures consist only of recorded independent OFF token IDs, with a
chosen mismatching proposal for reject/partial cases. The authoritative acceptance
count is always computed by the production session, never supplied to M51 by the
qualification driver.

Real initial frontier 255 (beyond the 128-row chronological local window), verify
span 8 consumed inputs. Every consumed prefix 1..8 is checked; materialized
cancellation checks consumed prefix 0. Repeated cycles retain 8,4,1,1,7 inputs,
then OFF continuation and a fresh-owner exact-list continuation. Greedy, scripted
deterministic, private NumPy stochastic RNG and seeded MLX categorical RNG compare
all 280 slots plus admission metadata bytes, committed history/frontier,
generated tokens, pending lookahead, sampler-input hashes/state and RNG state.
EOS/stop and max-token trials compare terminal state with independent OFF.
Max-span 32 and mid-list Python-publication/reentrant-sampler failures have reduced
state regression evidence, not checkpoint claims. This is not long-context MTP
qualification or a proposal-model admission benchmark.

Cancellation checks before verification, during tentative mutation and after
materialization restore prefix zero without a sampler call. Post-sampling cancel
is deferred until a coherent eight-input commit then exact-list idle retirement.
Injected generation faults at verify, tentative, materialized, acceptance,
frontier hook, post-settlement, history/lookahead, terminal and response boundaries
must burn all aliases and reject further continuation/publication.

No prompt/history replay or full-cache repack; M51 packed-prefix gather copies are
unchanged. Idle return preserves the exact list. Child borrow markers retire,
Engram futures drain, and closing the backend revokes resource admission.

Source SHA-256 at determining run:

- `ds41f_mlx/runtime/target_generation.py`: `b2130b1261e28006ec59d26bf5460fbab5eea4a0dc023a7fb931bbcd4e7958e8`
- `tests/test_m52_generation.py`: `643804d2ae57699b6e0baaa8844f9ac300bfe12e4e6179f8cdcb199c3e32dc88`
- `tools/probe_m52_generation.py`: `7084f45b13749fb591971d0c039d6385c31504878926a674f10094347fcfcc65`
