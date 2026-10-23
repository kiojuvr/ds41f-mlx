# Installed implementation inspection

Installation: `/Users/kioju/.local/share/deepseek-harness/node_modules/@deepseek-ai/`.
All 17 inspected package versions are 0.2.1-rc.2; exact entrypoint hashes are in
`identities.json`. Seven overlapping Phase 0 implementation hashes match exactly
(`audit-results.json`). Package manifests do not establish an upstream Git revision.
The CLI was invoked through its installed `dsh/lib/bin.js` with Node v24.5.0;
`dsh` was not on the shell PATH. No installed file was modified.

Verified interfaces (line numbers in installed `lib/index.js`):

- `dsh-tool-fs`: `write` / `edit` / `read` / `read_image`; path/content schemas
  at 549–566, 690–712, and 986–1018. `read_image` reads filesystem bytes and
  calls official `attachments.saveImage` at 1018–1031. It returns the official
  attachment reference, not synthetic wire content.
- `dsh-attachment-local`: image normalization/storage and request variants;
  `saveImageFile` at 583, service `saveImage` at 1031. No replacement was added.
- `dsh-llm-deepseek`: image preparation at 1419; serialization of prepared
  bytes or file IDs at 1590–1621; official Files-first / `FileResolutionFailure`
  inline branch at 2171–2190. These paths remained configured but were NOT
  reached with an image during this attempt.
- Same adapter at 1993–2019: requires settled blocks plus stop reason at
  `message_stop`, emits `finish`, and rejects premature stream termination.
  Direct observed stops and canonical Runtime responses were compared offline.
- `dsh-sandbox-policy`: only `read-only`, `workspace-write`, and
  `danger-full-access`; no command allowlist setting (97–106).
- `dsh-sandbox-local`: macOS Seatbelt profile at 65–74 uses `(allow default)`,
  denies writes, then grants shared writable roots. This is a file-write sandbox,
  NOT a read/network/command allowlist. No full-access mode was granted.
- `dsh-fs-sandbox`: read operations pass through; mutations are fenced against
  canonical writable roots (including platform temporary areas), 72–92, 159–173.
- `dsh-bash-local`: timeout/cwd/output controls at 69–76 and 87–102; command is
  arbitrary `bash -c` at 140–145. No executable allowlist option was found.
- `dsh-bash-sandbox`: wraps the original shell argv via the official sandbox
  provider, preserving shared per-call policy; no custom executor was installed.
- `dsh-tool-bash`: `enableRunInBackground` / `promoteOnTimeout` at 341–343;
  escalation goes through approval. Both background features were disabled.
- `dsh-user-approval`: `ask`/`never`; `never` rejects escalations, not ordinary
  standing-policy shell operations. It is NOT a command allowlist.
- `dsh-tools`: official monotonic `tools.guard()` at 3008–3029, applied after
  pre-execute at 3337. `ToolExecution.arguments`, name and call ID are declared
  in `lib/types/index.d.ts:217–281`. This permits a narrow deployment policy
  without replacing tool dispatch. The small non-official policy extension
  registered only a denial guard; it registered/executed no tools and changed
  no result or model content. Actual denials appear in official session events.
- `dsh-headless`: one direct official Agent/session, `--json`, optional
  `--session-id`; no headless command-specific approval flag was assumed.
  Checked launcher help, headless help, bundle patches, composed config and
  profile insertion syntax. Initial config dump rejected a missing `insert`
  directive before any generation; it was corrected in the retained setup.

The final tool surface was `bash`, `edit`, `exit_plan_mode`, `read`, `read_image`,
`write`. The policy denied everything except index.html operations, produced
workspace screenshot reads, and the exact foreground renderer command. The
otherwise inherited `exit_plan_mode` was not called and would be denied.

The failure was OUR deployment wrapper/configuration: `PHASE3_EVIDENCE` named the
repository, but the renderer subprocess could write only workspace/temp roots.
Its first archive write was correctly denied before Chromium launch. This does
not show that official Harness cannot support workspace-local rendering; no such
impossibility claim is made. A host-only smoke test missed this sandbox mismatch.

`executed/phase3_harness_guard.mjs` and `executed/phase3_render.py` preserve exactly
the bytes used at the original `tools/` paths. They were archived after the run,
not patched into a successful outcome. The profile/setup/launcher are execution
records with temporary absolute paths, not an installed or ready-to-run feature.
Do not run them automatically. No corrected second conversation was started.
