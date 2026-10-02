# M20 — oMLX 0.7.0 baseline migration

## Status: blocked at initial identity gate; not promoted

The initial read-only inspection found a release revision discrepancy. No runtime
configuration, runtime source, or external checkout was changed. No real-model
compatibility, A/B, long-session, or qualification gate was run.

Machine-readable evidence: [`../artifacts/m20/initial-identities.json`](../artifacts/m20/initial-identities.json).
It records working-tree file hashes, local-difference identities, native binary
hashes, package metadata, and upstream tag verification.

| Role | Checkout | Exact revision | State |
| --- | --- | --- | --- |
| Qualified baseline / rollback | `~/omlx-0.7.0.dev2` | `b390b31e0c6831225fed0f24d278eb1db7fcb68b` | Detached at `v0.7.0.dev2`, known local differences |
| Candidate as inspected | `~/omlx-0.7.0.release` | `87460f4d50de79aef9b67e99e215c31f0a89b445` | `main`, clean Git working tree, one commit after release |
| Intended upstream release | `v0.7.0` | `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` | Local tag matches `git ls-remote origin refs/tags/v0.7.0` |

The candidate's extra commit is `formula: bump to 0.7.0`, changing only
`Formula/omlx.rb`. Even though runtime sources match the tag, this is **not**
the exact requested release revision. A directory name and `__version__ =
"0.7.0"` do not establish release identity. Resolve the candidate to the
verified tag (or explicitly agree a different qualification identity) before
continuing. No checkout/reset was performed automatically.

## Content and environment identity

- dev2 M17/M18 local-difference identity:
  `f53fa43fd97e930c0f49ecc9f25b55ec5d334630b5ba9943a85e8c7751db73b2`.
- dev2 tracked working-tree content identity:
  `f69c02b7a5a1c78b208d058fe184faff20c3cee6f1e00fd9c7d98df9f98feb8b`.
- Candidate tracked working-tree content identity:
  `96bb4dd3a82b2feb25ce5e60c9f63601f89205a769a377c6cc41151319313811`.
- Candidate has no Git-visible local differences. Ignored native binaries and
  egg-info exist; clean Git status does not prove their provenance or loadability.
- dev2 interpreter: `~/.venvs/omlx-0.7.0.dev2/bin/python`, Python 3.13.15,
  editable oMLX 0.7.0.dev2, MLX 0.32.2, mlx-lm 0.31.3 at
  `ab1806e8f5d6aa035973af194a1b9198ab4754dc`.
- No dedicated `~/.venvs/omlx-0.7.0.release` was found. The candidate contains
  CPython 3.11 native extensions versus dev2's CPython 3.13 extensions. Establish
  the actual candidate interpreter and native ABI before any compatibility run.
- Repository `.venv`: Python 3.13.14, no installed oMLX distribution, MLX
  0.32.2, mlx-lm 0.32.0. This is not demonstrated to be the candidate environment.
- Release requirements pin MLX 0.32.2 and mlx-lm
  `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`; requirement declarations are not
  proof of installed executable identity.

Tracked-content hash construction and per-file hashes are in the artifact;
this identity is separate from the existing M17/M18 local-difference digest.
Native presence/hashing was inspected without importing or rebuilding kernels.

## Historical local patches

The dev2 encoding.py and processing.py modifications and untracked
`tests/test_deepseek_v41_literal_image_token.py` remain intact. Existing
provenance classifies their exact hashes as approved, non-production-reachable
text-serving differences. None was copied to the candidate. Whether upstream
makes them unnecessary has **not yet been audited**.

## Remaining work and qualification policy

After resolving identity, trace actual production imports/calls and classify
all requested seams before adapting code. Keep DENSE_P0_P7 prefill and the P5
handoff unchanged; MTP, DSpark, DFlash, speculation, and CED must remain off.
Use explicit `DS41F_OMLX_PATH=$HOME/omlx-0.7.0.release` during development.
The canonical default remains dev2.

No M18 evidence has been reclassified to qualify this candidate. Preliminary
scope for the later evidence review (not a completed evidence-by-evidence audit):

- Checkpoint/source semantics: RETAINED if source/checkpoint identity is unchanged.
- Dependency-independent prefill evidence: RETAINED only after confirming its
  actual executable dependencies are unchanged; oMLX helpers may still matter.
- Decode, handoff, session/cache lifecycle: FULL_RUNTIME_REQUALIFICATION_REQUIRED.
- Protocol/tool/EOS/cancellation: TARGETED_REQUALIFICATION_REQUIRED, including
  real-model lifecycle confirmation.
- Decode/continuation/long-session performance: PERFORMANCE_REQUALIFICATION_REQUIRED.
- 200K evidence: inspect the actual claim/dependencies before deciding to rerun;
  no blanket rerun or retention decision has been made.

Compatibility must stabilize before bounded dev2/release A/B and long-session
runs. Promotion requires every requested gate and refreshed invalidated evidence;
none has been waived because the candidate is clean or reports version 0.7.0.

## Operational path policy

`~/omlx` remains reserved for a separately constructed operational checkout;
it was neither used as evidence nor modified. Versioned checkouts remain the
qualification/rollback authorities. After promotion, an operational checkout
must match the promoted revision, local content, packages, and executable native
identity. Later changes constitute a new dependency identity, not inherited M20
qualification. No release baseline has been promoted yet.
