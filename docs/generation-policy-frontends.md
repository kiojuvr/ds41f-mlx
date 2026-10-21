# Generation policy — external clients and operator settings

Browser Chat and its generation-control frontend, workflow snapshots and Tool
Loop are retired. No application policy manager or executor was moved into Runtime.
Runtime `generation-policy-v1` and Recipe 0.2 semantic processing are unchanged;
[capabilities](capabilities.md) owns request controls and defaults.

The [operator console](operator-chat.md) retains Runtime launch default reasoning
settings only. External clients own per-request controls, conversation history
and tool effects. No replacement Harness is bundled, and successful inference is
not comprehensive Agent/Harness compatibility.

## Historical installed evidence

The unchanged paragraph below records original Browser/TUI/OpenCode observations,
not qualification of the retired application or a current Harness. Removed Web
tests/collectors are preserved through Git history, not repurposed as Runtime tests.

The [frontend acceptance receipt](generation-policy-frontend-acceptance.json)
records exact installed wheel/source scope, TUI/Chrome/OpenCode executions and
canonical policy observations. The
[original frontend edition](archive/editorial-baseline/generation-policy-frontends.md.txt)
retains base `28a7cee`, wheel SHA, all execution/effect/test counts, f32 values,
raw evidence paths and failed attempts. Its unchanged-backend assertion applies
to that integration delta, not later HEAD.
