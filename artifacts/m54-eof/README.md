# M54 EOF-sensitive operational continuation

Decision: **YES — development operational integration PASS**, no promotion.
See `docs/milestone-54-eof-sensitive-operational-continuation.md` for the full
architecture, authorization/ordinal/provenance, cancellation, effects, resource,
performance and scope report.

Authoritative final evidence:
- `native-provenance.json`, `recipe-consuming-eof-preview.patch`: incremental
  native dependency patch, exact source/installed binary/wheel identity.
- `canonical-{base,modified,parity}.json`: 22 exact consuming-protocol cases.
- `regressions.txt`: 266 tests + 8 subtests PASS after worker race repair.
- `standard-mtp.json`: final 17-turn official-checkpoint standard socket + Web
  application matrix, including deterministic in-worker tool EOF cancellation,
  lost async worker result, exact SSE argument reconstruction and once-only effects.
- `standard-off.json`: nominal OFF control, first 11 completed-turn generated
  token/frontier/message cases matched by `matched-correctness.json`. Final extra
  worker race is real MTP evidence + common-path regressions, not a repeated OFF
  real socket trial. Intentional precompletion drain endpoints are not matched.
- `matched-performance.json`: exact request/generated tokens/message/finish/
  frontier comparison and three fresh-session samples after one excluded warm-up
  for OFF / first-party MTP development / historical candidate. Integrated recipe
  decode: 15.88 / 10.62 / 31.79 tok/s; complete HTTP: 11.11 / 8.17 / 14.94 tok/s.
  Inclusive overlapping phase times are not additive. No speedup/promotion claim.
- `full-r1.json` and gate logs: **one full run, FAIL**, 24/25 gates PASS; existing
  public MTP profile rejects Python 3.13.14 vs required 3.13.15 before model load.
  This run predates the final worker-batch/terminal cancellation race repair.
  Final source therefore has no full R1 PASS, and no second full R1 was run.

Excluded attempts are retained under exploratory/pre-worker/excluded/rejected
names. Candidate missing JSON schema dependencies were installed at existing
`third_party/mtp/requirements.lock` versions; a measurement-only executor self-join
was corrected. No candidate arithmetic or environment qualification gate changed.
No performance result from a failed cleanup or before correctness closure is used.

No binary wheel, model weights, source-release copy or persistence tensor payload
is committed here. Native wheel remains an external reproducible development
artifact, not an upstream or release promotion. Full R1's inherited OFF restore
fixture does not expand first-party persistence scope. Unrelated Chrome log is
not part of this change.
