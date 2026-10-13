"""Ordinary boundary: recipe owns grammar; ds41f owns execution capabilities.

Never add message/field allowlists here. Wire-only controls below are explicit
runtime limitations for values the recipe validates but does not preserve.
"""
import json

from deepseek_recipe import ChatCompletionRequest, ConversionOptions

from ds41f_mlx.mtp_profile import ALIASES, strict_json
from .capacity import ORDINARY_BODY_BYTES, validate_ordinary_output


def admit_structure(raw):
    if len(raw) > ORDINARY_BODY_BYTES:
        raise ValueError('body limit exceeded')
    return strict_json(raw)


def admit_capabilities(converted, payload):
    if converted.model not in ALIASES:
        raise ValueError('fixed model alias required')
    options = converted.inference_options
    from .generation_policy import normalize_policy
    normalize_policy(converted, payload)
    validate_ordinary_output(128 if options.max_tokens is None else options.max_tokens)

    # These fields are intentionally omitted by the authoritative converter.
    # Do not silently claim unsupported sampling/constrained decoding controls.
    if payload.get('parallel_tool_calls') is False:
        raise ValueError('ordinary MTP does not enforce parallel_tool_calls=false')
    thinking = payload.get('thinking')
    if isinstance(thinking, dict) and thinking.get('budget_tokens') is not None:
        raise ValueError('ordinary MTP does not support thinking budget_tokens')
    response_format = payload.get('response_format')
    if isinstance(response_format, dict) and response_format.get('type') == 'json_schema':
        raise ValueError('ordinary MTP does not support constrained JSON Schema output')
    if payload.get('logprobs') or payload.get('top_logprobs'):
        raise ValueError('ordinary MTP does not support logprobs')

    # Inline Vision uses the existing bounded multimodal preparation authority.


def with_reasoning_default(payload, effort):
    """Only fill omitted controls; explicit recipe controls retain authority."""
    from ds41f_mlx.config import REASONING_EFFORTS
    if effort not in REASONING_EFFORTS:
        raise ValueError('unsupported default reasoning effort')
    if not any(key in payload for key in ('reasoning_effort', 'thinking')):
        return payload | {'reasoning_effort': effort}
    return payload


def convert_ordinary(raw, *, default_reasoning_effort='none'):
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
    include_usage = request.include_usage()
    converted = request.convert(ConversionOptions(default_thinking_mode=False))
    admit_capabilities(converted, payload)
    return converted, include_usage, automatic
