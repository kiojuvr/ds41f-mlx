# Ordinary production API authority audit

Scope: `mtp-serving-v1` Chat Completions admission. This is an API boundary
repair, not a new-client allowlist, implementation of every OpenAI feature,
release promotion, or numerical MTP requalification.

## Authority and order

```
bounded HTTP body / LocalBoundary
  → strict JSON object (UTF-8, duplicate keys, finite numbers)
  → authoritative ChatCompletionRequest parsing + conversion (once)
  → ds41f converted capability/security admission
  → recipe rendering and tokenization
  → actual prompt + output capacity admission
  → ProductionScheduler
```

`serving/ordinary_admission.py` implements the first conversion boundary.
`server.prepare_request(ordinary=True)` consumes that conversion directly; it
never calls the historical `mtp_profile.validate_chat`. The latter now has only
its pinned singleton qualification contract, with no ordinary grammar switch.
`ProductionMTPBackend.validate_ordinary` is a non-HTTP convenience entry point
using the same conversion/capability function, not another protocol validator.

Audit authority: installed `deepseek_recipe` Python API (`_native.pyi`) and
recipe Rust `deepseek-recipe/src/protocol/openai/chat_completion/request/`
`schema.rs` and `convert.rs`. Dependency identity remains the production startup
responsibility. Tests use the installed authority rather than copying its schema.

## Removed independent grammar

| Previous ds41f rule | Current owner |
|---|---|
| Top-level request field allowlist | Recipe schema; unknown fields follow recipe's ignore semantics |
| Exact message field sets, string-only content | Recipe content/block parsing and conversion |
| Four-role allowlist, user/tool-only history terminator | Recipe conversation semantics, including latest_reminder |
| Assistant content/null/reasoning/tool-call shapes | Recipe |
| Call IDs, ordered results, duplicates, foreign or missing results | Recipe transform_messages |
| Tool name, argument and tool-result fixture byte limits | Whole-body resource budget, converted security checks, actual encoded capacity |
| Stream options must equal include_usage=true | Recipe streaming validation and include_usage projection |
| Wire reasoning_effort must be none | Converted thinking_mode must be off (recipe resolves overrides) |
| Wire max_tokens must be present | Converted default 128; ordinary output ceiling remains 393,216 |

Tools, choice, response format and stop semantics remain recipe-owned. No
serving-layer tool schema parser, text-part grammar, or application-specific
normalization is retained. Text parts are joined/ordered exactly as the recipe
specifies, not flattened independently by ds41f.

## Retained constraints and rationale

- HTTP host/origin/connection/body/time/preparation constraints: existing
  LocalBoundary trusted-network/resource policy. Body ceiling: 16 MiB.
- JSON object, unique keys, finite numbers, UTF-8: ambiguity/security admission.
- Model alias: fixed loaded model, not protocol grammar.
- Temperature unset/zero; top_p unset/one; thinking off: qualified greedy MTP
  execution. Unsupported values are rejected after conversion, not silently
  replaced by Scheduler defaults.
- Images rejected from converted message sources before rendering, expansion,
  fetching or model loading: ordinary MTP is text-only.
- Raw special-token source policy retained for converted message text, reasoning,
  tool names/arguments/descriptions/parameters. This is source-injection policy,
  not role/block grammar. Converted strings use the body ceiling, not historical
  256-byte IDs or 64-KiB tool fixture ceilings.
- Actual encoded prompt plus output reservation uses checkpoint context range,
  qualified 1,048,576 total envelope and 393,216 output ceiling. Defaults are not
  capability ceilings. Existing Scheduler admission/settlement remains unchanged.

The recipe intentionally drops some caller-owned controls. These cannot be
checked only on ConversationRequest: nonzero frequency/presence penalties,
seed, parallel_tool_calls=false, thinking budget_tokens, JSON Schema constrained
output and requested logprobs are explicitly rejected as unsupported runtime
controls after authoritative conversion. Zero penalties, null controls and
parallel_tool_calls=true do not require extra execution behavior. This is a
small deny-list of capability promises, **not** a request-field allowlist.
Unknown extension fields still follow authoritative recipe semantics. Tool
`strict` declaration metadata is recipe-owned; this change adds no constrained
sampling or schema enforcement guarantee.

`max_tokens: "auto"` remains an explicitly ds41f-owned extension: recipe sees a
placeholder 1; actual encoding resolves the reservation. Scheduler and the
shared MTP execution start consume `request_max_tokens`, never that placeholder.

## Boundary acceptance / default-promotion gate

Required evidence is authority-based acceptance, not a count of clients:

- Recipe-accepted supported grammar projects identically through admission:
  string/parts, optional/null fields, reminders, history endings, stream options,
  stop sequences, JSON object output, metadata and tool choices/results.
- Recipe parser/converter failures retain their error class/message and become
  HTTP client errors. Duplicate/foreign/missing tool results are recipe errors.
- Recipe acceptance does not bypass explicit runtime/security/resource limits.
- HTTP performs one recipe parse/conversion; rejected requests never reach
  execution admission, and image rejection precedes rendering/expansion.
- Actual capacity/default/auto resolution and the singleton contract regressions
  pass. Existing lifecycle/checkpoint tests remain applicable without another
  long-context numerical qualification campaign.

Tests: `test_ordinary_text_parts.py`, `test_generic_mtp_tools.py`,
`test_production_mtp.py`, `test_production_context.py`, `test_runtime_capacity.py`,
`test_m41_profile.py`. OpenCode Plan's captured text-plus-reminder request is a
real boundary reproducer, not the production grammar specification.

Validation receipt for this change: the six boundary/capacity/singleton suites
above passed **180 tests**. An extended run including M29/M30/M33 passed 205 and
failed one M30 EOF audit because the installed native recipe lacks
`StreamProcessor.preview_certified_eof_tokens`. That dependency qualification
issue is separate from ordinary admission and remains unresolved; no inference
or promotion claim is made from the boundary tests.

This closes this admission defect's scope. It does not promote `mtp-serving-v1`
to default, qualify all API features, prove Windows end-to-end inference, or
replace separately scoped release/LAN/security gates.
