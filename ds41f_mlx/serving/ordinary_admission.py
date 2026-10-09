"""Ordinary boundary: recipe owns grammar; ds41f owns execution capabilities.

Never add message/field allowlists here. Wire-only controls below are explicit
runtime limitations for values the recipe validates but does not preserve.
"""
import json

from deepseek_recipe import ChatCompletionRequest, ConversionOptions

from ds41f_mlx.mtp_profile import ALIASES, strict_json, string
from .capacity import ORDINARY_BODY_BYTES, validate_ordinary_output


def admit_structure(raw):
    if len(raw) > ORDINARY_BODY_BYTES:
        raise ValueError('body limit exceeded')
    return strict_json(raw)


def admit_capabilities(converted, payload):
    if converted.model not in ALIASES:
        raise ValueError('fixed model alias required')
    options = converted.inference_options
    if options.temperature not in (None, 0):
        raise ValueError('ordinary MTP supports greedy temperature zero only')
    if options.top_p not in (None, 1):
        raise ValueError('ordinary MTP does not support top_p sampling')
    conversation = converted.conversation
    if conversation.thinking_mode:
        raise ValueError('ordinary MTP supports thinking-off only')
    validate_ordinary_output(128 if options.max_tokens is None else options.max_tokens)

    # These fields are intentionally omitted by the authoritative converter.
    # Do not silently claim unsupported sampling/constrained decoding controls.
    for field in ('frequency_penalty', 'presence_penalty'):
        if payload.get(field) not in (None, 0):
            raise ValueError(f'ordinary MTP does not support {field}')
    if payload.get('seed') is not None:
        raise ValueError('ordinary MTP does not support seed')
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

    # Text-only capability is checked BEFORE rendering/image expansion or I/O.
    for message in conversation.messages:
        if message.image_sources:
            raise ValueError('ordinary MTP supports text Chat Completions only')
        string(message.content, limit=ORDINARY_BODY_BYTES)
        if message.reasoning_content is not None:
            string(message.reasoning_content, limit=ORDINARY_BODY_BYTES)
        for call in message.tool_calls or ():
            for text in (call.name, call.arguments):
                string(text, limit=ORDINARY_BODY_BYTES)
    for tool in conversation.tools:
        for text in (tool.name, tool.description or '', json.dumps(tool.parameters, ensure_ascii=False)):
            string(text, limit=ORDINARY_BODY_BYTES)


def convert_ordinary(raw):
    payload = admit_structure(raw)
    automatic = payload.get('max_tokens') == 'auto'
    # ds41f's sole wire extension: resolve auto using actual encoded length later.
    recipe_body = raw
    if automatic:
        recipe_body = json.dumps(payload | {'max_tokens': 1}).encode()
    request = ChatCompletionRequest(recipe_body)
    include_usage = request.include_usage()
    converted = request.convert(ConversionOptions(default_thinking_mode=False))
    admit_capabilities(converted, payload)
    return converted, include_usage, automatic
