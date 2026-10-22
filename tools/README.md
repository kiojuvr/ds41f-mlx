# Development and verification tools

The supported product is package-fixed Flash / Flash-1017, MTP-only, Recipe 0.2,
with a Runtime-only operator TUI. Provision through `python -m ds41f_mlx.mtp_setup`;
[operations](../docs/operations.md) owns prerequisites and publication. Never run
mutation/admission probes against a live or already qualified environment.

## Active delivery and integrity

- `build_execution_package.py`: existing clean staged-wheel builder; selects
  `--checkpoint-variant flash|flash-1017` and native/portable GLM explicitly.
- `build_glm_bundle.py`: publisher-owned native build from retained source inputs.
- `check_glm_build_fidelity.py`, `qualify_glm_admission.py`,
  `qualify_execution_admission.py`: bounded numerical/admission checks, not fallback
  selectors. Installed payload hashing and checkpoint admission remain authoritative.
- `check_repository_self_containment.py`: current documentation owners, live links
  and retained provenance destinations; no model execution.
- `check_legacy_import_integrity.py`, `check_native_import_dependencies.py`,
  `check_source_identity_hashes.py`, `source_identity.py`, `check_authority_labels.py`:
  source identity, attribution and imported evidence integrity. Source checks and
  wording checks execute independently. See the [verification baseline](../docs/verification.md).
  `audit_source_identity_hashes.py` is a historical receipt-normalization utility,
  not a supported verification step; do not use it to rewrite acceptance evidence.
- `promote_release.py` and R1 establish/finalize/qualification tools: retained
  reference/release construction and verification. Not routine cleanup commands.

## Supported serving acceptance and diagnostics

`accept_native_mtp.py`, `accept_generic_mtp_tools.py`, `accept_long_mtp_serving.py`,
`accept_mtp_vision.py`, `accept_generation_opencode.py` and `accept_remote_lan.ps1`
cover bounded, source-scoped serving/operator cases. `observe_native_selection.py`
observes the canonical installed launcher rather than owning another Runtime.
`qualify_production_sessions.py` checks logical-session behavior.

`bench_public_mtp.py`, the MTP startup/verification/attention/head and target-prefill
bench/profile/summarize tools, cache-hit diagnostics, and bit-exact kernel/template
comparators remain optional affected-boundary diagnostics. They load models or
measure performance when invoked; do not run them as a documentation check or
inherit their historical results as current qualification.

## Numerical and retained reproduction tools

`run_official_*_fixture.py`, `run_native_*`, `check_boundary*`, and the M4 same-input
and source-derived validation tools retain independent numerical/reference value.
Older prefill/P5/P6/P8 tools and R1/context-copy verification remain bounded
reproduction aids, not supported serving entry points. Follow their recorded
inputs and qualified interpreter; the tool prefix is not a production selector.

Some old numbered tools remain because executable tests, `ds41f_mlx.qualify`,
retained source archives or release records still consume them. In particular,
`accept_native_off.py` reproduces release-bound acceptance; it does not restore
OFF packaging/serving support. Historical milestone evidence generators and
unconsumed M0/M2 prototypes, old recipe clone drivers and retired server smoke/
soak scripts were pruned instead of repaired. Their source remains in Git at
`c1d1f716:tools/<name>`.
