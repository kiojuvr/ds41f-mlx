# Multimodal-Runtime — Production completion

**Disposition: OPERATIONALLY COMPLETE**, 2026-10-28. Explicit operational closure
under the [CURRENT contract](../../docs/multimodal-runtime-plan.md), not all-C1–C9
qualification. No implementation blocker remains in the declared envelope.

## Payload and admission

- Promoted source commit: **`977c57e3088d2a2204101c5f9a693707aec4bcb4`**,
  branch `Multimodal-Runtime`. Subsequent closure changes are documentation/evidence
  only; this record's Git commit identifies the final closure revision.
- Wheel: `ds41f_mlx-0.23.0-py3-none-any.whl`, SHA256
  **`b1760064b1b7c2d99c7539f7cd09a78bdc42b05fcbbdbcd778682f13607398b7`**.
- Existing target reused: `/Volumes/SDXC-512/ds41f-1017`, owner `kioju:staff`.
  Previous/new distribution version both **0.23.0** (version alone is insufficient).
  Previous build source `aa8c5717b42aa0a216fef3134e72c6db7ca49287`; new build source
  is the commit above. Full old/new RECORD hashes and provenance are retained in
  [evidence](production-promotion-evidence/).
- Installed execution identity remains
  `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
  This admission digest does not seal Python source; all **124** promoted package
  wheel members were independently compared byte-for-byte with installed files.
  Isolated executable imports resolve inside this venv, not the checkout.
- Every admission manifest field except provenance is unchanged: dependency locks,
  native GLM/MLX/Recipe resources, tokenizer, checkpoint inventory, variant and
  `flash-1017-pre-woa-fp8-v1` numerical profile. All installed distribution
  names/versions/origins compare equal before/after. No dependency update occurred.
- Matching official checkpoint:
  `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`.
  Existing full-byte SHA256 admission of the exact 48-shard inventory passes before,
  after installation and after shutdown; cache use is bound to unchanged file
  identities. Checkpoint and original Flash installation were not modified.

## In-place procedure and rollback

Before mutation: verified clean expected HEAD, actual package/import origins,
manifest/dependencies/checkpoint, environment ownership and process inventory.
No active Runtime/model process or port-8000 listener existed; no active work
needed interruption. Existing package admission passed. Its recorded temporary
wheel path no longer existed, so all **135 RECORD-owned installed files** were
hash-verified and archived before replacing anything (not development source).

Retained outside the repository, **not another runtime environment**:
`/Volumes/SDXC-512/ds41f-1017-promotion-rollback-977c57e3/` contains the previous
package archive, previous-package identity JSON, wheel comparison, final verification
and `promoted-wheel/` with the exact new wheel. Archive SHA256:
`e7a37c237b7dbb9cfa0cb16fe4fb9bec6db49a7c6b604230d58cfbf466aa2d71`.
Every archived member was independently rehashed against the pre-install inventory.

Used the existing publisher, without provisioning or resealing:

```sh
ENV=/Volumes/SDXC-512/ds41f-1017
SAVE=/Volumes/SDXC-512/ds41f-1017-promotion-rollback-977c57e3
"$ENV/bin/python" tools/build_execution_package.py \
  --checkpoint-variant flash-1017 --output "$SAVE/promoted-wheel"
uv pip install --python "$ENV/bin/python" --no-deps --reinstall \
  "$SAVE/promoted-wheel/ds41f_mlx-0.23.0-py3-none-any.whl"
```

Rollback, **only with the owned Runtime gracefully stopped and port released**:
verify archive SHA above; `uv pip uninstall --python "$ENV/bin/python" ds41f-mlx`,
then `tar -xpf "$SAVE/previous-package.tar.gz" -C "$ENV"`. Verify every restored
file against `previous-package.json`, then run ordinary installed `ds41f inspect`
with the matching checkpoint before starting. This restores the exact prior
installed package/entrypoints/metadata, including its original provenance; it
is not copying checkout source into site-packages. Dependencies remain untouched
and are recorded both in the snapshot and exact admission manifest. Rollback was
validated by archive member hashes, not performed over the accepted installation.

No `mtp_setup`, persistent second venv, editable install, bypass, manual admitted
asset change, model/kernel/Scheduler modification or unrelated package change.

## Installed acceptance

Evidence: [final verification](production-promotion-evidence/final-verification.json),
[ordinary acceptance](production-promotion-evidence/ordinary-acceptance.json),
[Messages smoke](production-promotion-evidence/messages-smoke.json),
[Runtime log](production-promotion-evidence/runtime.log) and
[shutdown](production-promotion-evidence/shutdown.json).

- Normal installed `inspect` before/after/final: **PASS**, native GLM, correct
  checkpoint and isolated installed origins. Ordinary loopback `mtp-serving-v1`
  process **82787**, owned by `kioju`, reached `model_ready:true` and exposed
  Recipe 0.2 native Messages and the expected execution identity.
- Existing `ds41f accept`: **PASS**, generic tool call/result, text completion,
  paired continuation/branch, Chat SSE and malformed requests refused. Four
  generations plus two expected pre-native 400 refusals.
- Three bounded public native Messages requests: **PASS**. Model requested
  `read_image(smoke.png)`; one real file read returned a 384×384 red-circle PNG
  under that exact tool ID; model identified red/circle and continued correctly.
  Settled frontiers **320 →530 →550**, restored **C=320 /530**. All seven
  generated requests log canonical publication, no errors/cancellation and
  **replay=0/repack=0**. This focused HTTP check is not another test Harness or
  a repeat of C8. Native Messages JSON completion passes; nineteen-stream native
  Messages `message_stop` evidence remains in the prior authoritative E2E report.
- Final health: no active/queued work, no fatal error. Graceful SIGTERM: cache
  cleared, Scheduler shutdown completed, application shutdown complete, owned
  PID absent and port released. Same-process model destruction is not claimed.
- One preliminary launch with diagnostics enabled was refused before loading
  (`diagnostics/custom model configuration unavailable in MTP`); its log is
  retained. Removed the unsupported launch option and used normal configuration.
  No generation/effect was retried, no admission bypass and no native failure.

The existing [C8 PASS](phase4-e2e-native.md) was **not rerun**. No macOS GUI/Windows
Harness run or extensive endurance campaign was required or performed.

## Supported envelope and remaining qualifications

Production is single-operator macOS/M3 Ultra 512 GB, Flash-1017/native GLM/paired
MTP, delivered Recipe 0.2.0/pinned dependencies, official Harness 0.2.1-rc.2 native
Messages **Thinking OFF**. Initial integrated policy: count/quantum **3/1**,
automatic pressure/retention/summary **8192/1024/2048**, ordinary output **1200**,
generic retries **0**, addition-only tools and session-log upload disabled.
Use available current sources within **36,864 aggregate patches**, whole-image-safe
prefix selection and all ordinary byte/pixel/context/output/cache/headroom guards.
One application/model lifetime per process; explicitly trusted LAN or loopback,
no authentication/TLS/Internet/multi-user promise. Other recommended reasoning
and output settings have not inherited integrated qualification.

C2 remains **PARTIAL** (broader case/numerical equivalence and intermediate
boundaries); C7 remains **PARTIAL** (broader peak/cancellation/endurance matrix and
same-process reclamation); broader C6 negatives and C9 recovery remain **PARTIAL**.
Finite C8 and scoped C1/C3/C4/C5/normal C6 are not universal PASS. Historical failures,
full-forward differences and framework-retained model weakrefs remain unchanged.
Files are bounded, volatile and not purged by offload; attachment loss, quota/body
limits and failed/truncated summaries may require operator action. Arbitrary 400
does not trigger official offload. No uncertain request retry, tool replay, hidden
history reconstruction, lost exact Chat-lease recovery or native persistence.

Windows Harness qualification is **post-closure operational work**. General Files
extensions/durable IDs, arbitrary numerical equivalence and further endurance
campaigns are not closure prerequisites. CURRENT owners now agree with the achieved
operational goal and reference this record. Lightweight repository/documentation
checks pass. No push, merge, release tag, Windows modification or further phase.
**Project closed; stop here.**
