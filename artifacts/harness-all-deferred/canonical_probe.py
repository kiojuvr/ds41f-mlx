"""Run with old and candidate wheels; compare previously accepted canonical states."""
import hashlib
import json
from pathlib import Path
import sys
import deepseek_recipe as d

options = d.ConversionOptions(default_thinking_mode=False).with_model_versions_by_name(
    {'deepseek-flash': d.ModelVersion.DeepseekV41Flash})
tok = d.Tokenizer.from_file(sys.argv[1])
encoding = d.DeepseekV41Encoding().with_tokenizer(tok)
rows = {}
for mode in ('absent', 'empty', 'immediate', 'mixed', 'repeated', 'none', 'named'):
    value = dict(model='deepseek-flash', max_tokens=64, thinking={'type': 'disabled'},
                 messages=[dict(role='user', content='before')])
    if mode != 'absent':
        value['tools'] = [] if mode == 'empty' else [dict(name='first', input_schema={'type': 'object'})]
    if mode in ('mixed', 'repeated', 'none', 'named'):
        value['tools'].append(dict(name='second', input_schema={'type': 'object'}, defer_loading=True))
        value['messages'].append(dict(role='system', content=[
            dict(type='text', text='instructions'),
            dict(type='tool_addition', tool=dict(type='tool_reference', name='second'))]))
    if mode == 'repeated':
        value['messages'] += [dict(role='user', content='middle'), dict(role='system', content=[
            dict(type='tool_addition', tool=dict(type='tool_reference', name='second'))])]
    if mode == 'none': value['tool_choice'] = dict(type='none')
    if mode == 'named': value['tool_choice'] = dict(type='tool', name='second')
    value['messages'].append(dict(role='user', content='after'))
    converted = d.MessagesRequest(value).convert(options)
    prompt = encoding.render_conversation(converted.conversation).prompt
    tokens = list(encoding.encode(converted.conversation))
    rows[mode] = dict(prompt=prompt, tokens=tokens,
                      token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
print(json.dumps(rows, indent=2))
