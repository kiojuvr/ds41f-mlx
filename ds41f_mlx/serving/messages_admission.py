"""Wire capability rejection only; Recipe owns Messages grammar and conversion.

Recipe intentionally ignores extensions and degrades documents. Reject those
before conversion/executable admission, including nested ignored controls.
"""
from .ordinary_admission import admit_structure, admit_capabilities


def fields(value, supported):
    if isinstance(value, dict) and set(value) - set(supported.split()):
        raise ValueError('unsupported Messages fields: ' + ', '.join(sorted(set(value) - set(supported.split()))))


def blocks(content):
    if not isinstance(content, list):
        return
    supported = {'text': 'type text', 'thinking': 'type thinking signature',
                 'image': 'type source', 'tool_use': 'type id name input',
                 'tool_result': 'type tool_use_id content is_error',
                 'tool_addition': 'type tool'}
    for block in content:
        if not isinstance(block, dict):
            continue  # Native schema reports malformed input.
        kind = block.get('type')
        if kind not in supported:
            raise ValueError(f'unsupported Messages block: {kind}')
        fields(block, supported[kind])
        if kind == 'image':
            source = block.get('source')
            fields(source, 'type media_type data')
            if isinstance(source, dict) and source.get('type') != 'base64':
                raise ValueError('Messages images require inline base64; Files/URL transport unsupported')
        elif kind == 'tool_result':
            blocks(block.get('content'))
        elif kind == 'tool_addition':
            fields(block.get('tool'), 'type name')


def convert_messages(raw, *, default_reasoning_effort='none', _resolved_payload=None):
    import json
    import deepseek_recipe as d
    from ds41f_mlx.mtp_profile import ALIASES
    from .generation_policy import with_sampling_defaults
    payload = admit_structure(raw)
    if _resolved_payload is not None:
        # Only the server's admitted, bounded file resolver supplies this view.
        payload = _resolved_payload
    # Harness inventory is non-semantic metadata, not a Recipe extension.
    # Remove only after bounded, strict JSON admission has checked the full body.
    payload.pop('dsh_plugin_packages', None)
    recipe_payload = with_sampling_defaults(payload)
    if 'thinking' not in payload and 'output_config' not in payload:
        recipe_payload['output_config'] = {'effort': default_reasoning_effort}
    automatic = payload.get('max_tokens') == 'auto'
    if automatic:
        recipe_payload['max_tokens'] = 1
    # Deserialize with the official schema first; capability checks are not a
    # replacement grammar (nor a source of 500s for malformed native fields).
    request = d.MessagesRequest(json.dumps(recipe_payload).encode())
    fields(payload, 'model messages system max_tokens temperature top_p stop_sequences stream tools tool_choice thinking output_config metadata')
    for message in payload.get('messages') or []:
        fields(message, 'role content')
        if isinstance(message, dict):
            blocks(message.get('content'))
    blocks(payload.get('system'))
    for tool in payload.get('tools') or []:
        fields(tool, 'name description input_schema defer_loading')
    fields(payload.get('thinking'), 'type budget_tokens')
    fields(payload.get('output_config'), 'effort')
    fields(payload.get('metadata'), 'user_id')
    fields(payload.get('tool_choice'), 'type name disable_parallel_tool_use')
    choice = payload.get('tool_choice') or {}
    if choice.get('disable_parallel_tool_use') is True:
        raise ValueError('disable_parallel_tool_use is not enforced')
    if choice.get('type') == 'any':
        raise ValueError('Messages tool_choice=any is not enforced')
    converted = request.convert(
        d.ConversionOptions(default_thinking_mode=False).with_model_versions_by_name(
            {alias: d.ModelVersion.DeepseekV41Flash for alias in ALIASES}))
    admit_capabilities(converted, payload)
    return converted, False, automatic
