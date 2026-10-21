"""Ordinary boundary: recipe owns grammar; ds41f owns execution capabilities.

Never add message/field allowlists here. Wire-only controls below are explicit
runtime limitations for values the recipe validates but does not preserve.
"""
import json

from deepseek_recipe import ChatCompletionRequest, ConversionOptions

from ds41f_mlx.mtp_profile import ALIASES, strict_json
from .capacity import ORDINARY_BODY_BYTES, ORDINARY_OUTPUT_CEILING, validate_ordinary_output


def admit_structure(raw):
    if len(raw) > ORDINARY_BODY_BYTES:
        raise ValueError('body limit exceeded')
    return strict_json(raw)


def admit_capabilities(converted, payload):
    if payload.get('model') not in ALIASES:
        raise ValueError('fixed model alias required')
    options = converted.inference_options
    from .generation_policy import normalize_policy
    normalize_policy(converted, payload)
    # Validate explicit limits here; resolve_output_budget owns omission.
    # Actual capacity is checked after recipe-expanded prompt accounting.
    if options.max_tokens is not None:
        validate_ordinary_output(options.max_tokens)

    # These fields are intentionally omitted by the authoritative converter.
    # Do not silently claim unsupported sampling/constrained decoding controls.
    if payload.get('parallel_tool_calls') is False:
        raise ValueError('ordinary MTP does not enforce parallel_tool_calls=false')
    thinking = payload.get('thinking')
    if isinstance(thinking, dict) and thinking.get('budget_tokens') is not None:
        raise ValueError('ordinary MTP does not support thinking budget_tokens')
    response_format = payload.get('response_format')
    if isinstance(response_format, dict) and response_format.get('type') in ('json_schema', 'regex'):
        raise ValueError('ordinary MTP does not support constrained JSON Schema or regex output')
    # Recipe describes constraints; the native sampler does not enforce them.
    # Existing prompt-based JSON/tool selection is not constrained decoding.
    if payload.get('logprobs') or payload.get('top_logprobs'):
        raise ValueError('ordinary MTP does not support logprobs')

    # Inline Vision uses the existing bounded multimodal preparation authority.


def resolve_output_budget(request):
    """Ordinary Chat Completions semantics after canonical reasoning resolution.

    Explicit Auto is handled separately after actual prompt expansion; its
    recipe placeholder must never be interpreted as an omitted output limit.
    """
    # Ask Recipe for its semantic default without silently lowering an omitted
    # reservation to the physical context left. Existing capacity admission
    # below remains the owner of prompt + resolved-output enforcement.
    prompt = len(request.token_ids)
    return request.conversation_request.resolve_max_tokens(prompt, prompt + ORDINARY_OUTPUT_CEILING)


def with_reasoning_default(payload, effort):
    """Only fill omitted controls; explicit recipe controls retain authority."""
    from ds41f_mlx.config import REASONING_EFFORTS
    if effort not in REASONING_EFFORTS:
        raise ValueError('unsupported default reasoning effort')
    if not any(key in payload for key in ('reasoning_effort', 'thinking')):
        return payload | {'reasoning_effort': effort}
    return payload


def convert_ordinary(raw, *, default_reasoning_effort='none'):
    from deepseek_recipe import ModelVersion
    payload = admit_structure(raw)
    automatic = payload.get('max_tokens') == 'auto'
    # Defaults are resolved only AFTER strict wire admission, then converted by
    # recipe. Qualification callers must explicitly request their greedy policy.
    from .generation_policy import with_sampling_defaults
    recipe_payload = with_reasoning_default(with_sampling_defaults(payload), default_reasoning_effort)
    if automatic:
        recipe_payload = recipe_payload | {'max_tokens': 1}
    recipe_body = json.dumps(recipe_payload).encode()
    request = ChatCompletionRequest(recipe_body)
    converted = request.convert(ConversionOptions(default_thinking_mode=False).with_model_versions_by_name(
        {alias: ModelVersion.DeepseekV41Flash for alias in ALIASES}))
    # The generator reads authoritative response options from `converted`.
    include_usage = bool((recipe_payload.get('stream_options') or {}).get('include_usage', False))
    admit_capabilities(converted, payload)
    return converted, include_usage, automatic
