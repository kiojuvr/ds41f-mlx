# Ordinary production API admission

This page owns **`mtp-serving-v1` Chat Completions admission order and authority**,
not a milestone diary or a new client allowlist. [Capabilities](capabilities.md)
owns limits/defaults; [ordinary serving](mtp-production-serving.md) owns execution;
[production sessions](production-sessions.md) adds retained-history/correlation
rules. This contract neither implements every OpenAI feature nor promotes MTP.

## Authority and order

```text
bounded HTTP ingress / LocalBoundary
  -> strict JSON object: UTF-8, unique keys, finite numbers
  -> sampling defaults / explicit Auto placeholder
  -> authoritative ChatCompletionRequest parsing + conversion once
  -> ds41f converted execution/security capability admission
  -> recipe rendering and tokenization
  -> actual prompt + output capacity admission
  -> ProductionScheduler
```

`serving/ordinary_admission.py` performs structure admission and official recipe
conversion. `server.prepare_request(ordinary=True)` consumes that conversion;
it does not call singleton `mtp_profile.validate_chat`. The non-HTTP
`ProductionMTPBackend.validate_ordinary` uses the same conversion/capability
function, not a second protocol validator. Startup identity remains
[installed admission](runtime-admission.md), not checkout equality or local seals.

## Recipe-owned grammar

The installed recipe API and admitted schema/converter are authoritative for:

- optional/unknown fields and null semantics;
- message roles, history endings/reminders and text/content-part ordering;
- assistant content/reasoning/tool-call shapes;
- tool declarations/choices, IDs and ordered/missing/duplicate/foreign results;
- supported response formats, stop parsing and stream/include-usage projection.

No serving-layer field allowlist, tool schema parser, text-part grammar or
application-specific normalization replaces them. Unknown extensions follow
recipe behavior; successful OpenCode or fixture requests are not the schema.
Tool `strict` metadata does not grant constrained decoding/schema enforcement.

Ordinary text may legitimately contain `<|` and fullwidth `｜` in code, regexes,
quoted token documentation or tool results. Do not inherit singleton's blanket
string filter, scrub text or rewrite encoding. Recipe unbacked-image-placeholder
rejection and the encoded missing-image-source guard remain. This does not claim
role-delimiter escaping or prompt-injection protection; clients treat tool text
as untrusted and enforce permissions. Accepting text does not add raw-token execution.

## ds41f-owned constraints and rationale

| Constraint | Current reason/owner |
| --- | --- |
| Host/origin/body/connection/time/preparation bounds | Trusted-network/resource admission; [operations](operations.md) |
| Strict JSON object, UTF-8, unique keys, finite numbers | Ambiguity/security rejection before conversion |
| Model alias | Fixed admitted model, not protocol grammar |
| Sampling normalization/validation | `generation-policy-v1`; unsupported or invalid controls reject, not silent Scheduler defaults |
| Converted image sources | Existing bounded inline preparation/expansion; remote/file URLs reject without fetching; paired hashes bind image identity; historical misses reject without replay |
| Actual encoded prompt/output reservation | Checkpoint range and ordinary total/output capability; [capacity](capabilities.md#capacity-resource-policy-and-defaults) |
| Retained reasoning/tools/history | Session continuity, not a universal stateless grammar rule |

Wire-only controls intentionally dropped by recipe still need explicit capability
checks. `parallel_tool_calls=false`, thinking `budget_tokens`, JSON Schema
constrained output and requested logprobs are unsupported promises and reject
after authoritative conversion. `parallel_tool_calls=true` does not add an
execution guarantee. This is a small capability deny-list, not a field allowlist.
Penalties and seed are preserved by the generation policy; seed is best-effort,
not deterministic reproduction. Defaults and advertised efforts live in
[capabilities](capabilities.md#generation-controls).

`max_tokens: "auto"` is an explicit ds41f extension. Recipe receives placeholder
`1`; actual tokenization resolves the real reservation. Scheduler/execution use
`request_max_tokens`, never the placeholder. Missing `max_tokens` resolves to the
ordinary default, not Auto or the output ceiling.

## Boundary acceptance / default-promotion gate

Acceptance must establish authority, not count clients:

1. Supported recipe-accepted grammar projects identically through admission.
2. Recipe parser/converter failures retain their class/message as client errors.
3. Recipe acceptance cannot bypass explicit execution/security/resource limits.
4. HTTP performs one conversion; rejected requests do not start native execution,
   and image rejection precedes rendering/expansion.
5. Actual capacity/default/Auto resolution and unchanged singleton regressions pass.

Focused owners are `test_ordinary_text_parts.py`, `test_generic_mtp_tools.py`,
`test_production_mtp.py`, `test_production_context.py`, `test_runtime_capacity.py`
and `test_m41_profile.py`. Existing lifecycle/checkpoint evidence applies at its
scope; admission repair alone does not require numerical core requalification.

## Historical audit and deployment follow-ups

The [original audit edition](archive/editorial-baseline/ordinary-api-boundary.md.txt)
retains removed-rule comparisons, original acceptance counts, the M30 native-API
environment failure, source-inventory/seal refresh instructions and real OpenCode
source-read reproducer. Those local paths and refresh steps are history, not
current installation instructions. The unresolved status recorded there describes
that dependency environment at that time, not every present runtime.

[Mac production review](opencode-production-review.md) owns its source-scoped review
disposition, including the **CLOSED** Windows OpenCode / Chat Server access
follow-up based on user confirmation. That access confirmation does not qualify
all API features or establish full Windows end-to-end inference, default-MTP
approval or general-runtime release. Historical unverified Windows status in the
archived audit is superseded for this access follow-up only. Current production
changes follow the
[serving decision policy](production-serving-parity-closure.md).
